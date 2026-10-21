"""Read-only source/provenance audit plus artificial CPU forward checks."""
import ast
import csv
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import torch

from mkp_config import ModelConfig, ROOT
from mkp_env import MKPEnv
from mkp_Model import MKPModel
from mkp_routing import routing_geometry
from mkp_attention import candidate_relations
from mkp_utils import file_sha256, load_checkpoint


def source_contracts():
    unchanged = {}
    for filename, names in {
        "mkp_Model.py": ("StrategyTheta", "WeakDecoder", "state_features", "BipartiteBlock", "BipartiteEncoder"),
        "mkp_Trainer.py": ("generalized_advantage", "collect_rollout"),
        "mkp_basis.py": ("interaction_loads", "companion_basis", "feasible_means"),
    }.items():
        original = subprocess.check_output(["git", "show", "HEAD:" + filename], cwd=ROOT).decode("utf-8")
        current = (ROOT / filename).read_text(encoding="utf-8-sig")
        def nodes(text):
            return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(text).body
                    if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
        before, after = nodes(original), nodes(current)
        for name in names:
            unchanged[filename + ":" + name] = before[name] == after[name]
    if not all(unchanged.values()):
        raise AssertionError(unchanged)
    return unchanged


@torch.no_grad()
def audit():
    torch.set_num_threads(2)
    torch.manual_seed(204)
    result = dict(scope="No training, no real benchmark evaluation", unchanged_ast=source_contracts())
    baseline_path = ROOT / "results" / "20260922_083444_537744"
    manifest = json.loads((baseline_path / "manifest.json").read_text(encoding="utf-8-sig"))
    with (baseline_path / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        result["historical_mean_gap_pct"] = next(float(row["mean_gap_pct"]) for row in csv.DictReader(stream)
            if row["method"] == "learned_dynamic" and row["M"] == "all" and row["N"] == "all")
    result["checkpoint_provenance"] = {}
    for m in (5, 10):
        path = ROOT / "checkpoints" / "combination_v5" / f"mkp_M{m}_best.pt"
        before_hash = file_sha256(path)
        model, payload = load_checkpoint(path, "cpu", m)
        env = MKPEnv(torch.rand(1, 7) + .1, torch.rand(1, 7, m), torch.full((1, m), 2.))
        policy, _, _ = model.policy(env.state)
        if not torch.isfinite(policy.logits).all():
            raise AssertionError("Legacy checkpoint forward is non-finite")
        after_hash = file_sha256(path)
        if after_hash != before_hash:
            raise AssertionError("Source checkpoint changed")
        historical = manifest["checkpoints"][str(m)]
        result["checkpoint_provenance"][str(m)] = dict(path=str(path), sha256=before_hash,
            historical_sha256=historical["sha256"], historical_update=historical["update"],
            current_update=payload["update"], matches_historical=before_hash == historical["sha256"],
            restored_actor_encoder=model.config.actor_encoder, source_unchanged=True)
    current_c = ModelConfig()
    legacy_c = replace(current_c, actor_encoder="legacy", actor_strategy_readout=False)
    current, legacy = MKPModel(current_c), MKPModel(legacy_c)
    result["parameters"] = {name: dict(total=sum(p.numel() for p in model.parameters()),
        actor=sum(p.numel() for p in model.actor.parameters())) for name, model in (("routing", current), ("legacy", legacy))}
    current.critic_encoder.load_state_dict(legacy.critic_encoder.state_dict(), strict=True)
    current.value_head.load_state_dict(legacy.value_head.state_dict(), strict=True)
    state = MKPEnv(torch.rand(2, 8) + .1, torch.rand(2, 8, 5), torch.full((2, 5), 2.)).state
    torch.testing.assert_close(current.value(state), legacy.value(state), rtol=0, atol=0)
    result["independent_critic_same_weights_exact_output"] = True
    result["actor_geometry_bytes"] = []
    for n in (100, 250, 500):
        state = MKPEnv(torch.rand(1, n) + .1, torch.rand(1, n, 10), torch.full((1, 10), 3.)).state
        routing = routing_geometry(state, current_c)
        relations = candidate_relations(state, legacy_c)
        size = lambda values: sum(x.numel() * x.element_size() for x in vars(values).values())
        result["actor_geometry_bytes"].append(dict(N=n, M=10, routing=size(routing), legacy_relations=size(relations)))
    out = ROOT / "outputs" / "resource_routing_v1" / "source_and_provenance.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
