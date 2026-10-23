"""Synthetic same-state diagnostics, separate from training and Chu evaluation.

Run phases individually or `all`. No diagnostic actor is a benchmark checkpoint.
Full experiments are explicitly user-run; importing this module starts nothing.
"""
import argparse
import json
from pathlib import Path

from mkp_utils import file_sha256, resolve_device
from mkp_audit_common import check_report, write_json


def make_report(bank_path, reach_path, output, learning_path=None, retention_path=None):
    from mkp_audit_bank import load_bank
    bank = load_bank(bank_path)
    reach = json.loads(Path(reach_path).read_text(encoding="utf-8"))
    check_report(bank_path, reach)
    counts = reach["counts"]
    hints = []
    if counts["analytic_unreachable"]:
        hints.append("Some targets have explicit local fixed-score exclusions. Inspect their value and stage; do not extrapolate to a benchmark gap floor.")
    if counts["unknown"]:
        hints.append("Unresolved theta searches remain. Increase search budget or improve bounds; unknown is not unreachable.")
    learning = retention = None
    if learning_path:
        learning = json.loads(Path(learning_path).read_text(encoding="utf-8"))
        check_report(bank_path, learning)
        for m, result in learning["models"].items():
            if result["status"] != "complete":
                hints.append(f"M{m}: no reachable fit states; the learning experiment is inconclusive.")
                continue
            for split in ("fit", "heldout"):
                summary = result["after_summary"].get(split)
                if not summary:
                    continue
                if split == "fit" and summary["target_hit_rate"] < 1:
                    hints.append(f"M{m}: reachable fit targets remain unlearned. Inspect fit curves and action margins before changing the score family.")
                if split == "heldout" and summary["greedy_worsened_states"]:
                    hints.append(f"M{m}: heldout same-prefix greedy continuations regressed after fit; inspect generalization and policy-dependent labels.")
    if retention_path:
        retention = json.loads(Path(retention_path).read_text(encoding="utf-8"))
        check_report(bank_path, retention)
        for name, result in retention["experiments"].items():
            if not result["history"]:
                continue
            last = result["history"][-1]
            for split, measurements in last["retention"].items():
                rate = measurements["target_retention_rate"]
                if rate is not None and rate < 1:
                    hints.append(f"{name}/{split}: some initially correct bank decisions were lost during production PPO. Compare arm, true future returns, gradients and exact KL stop events.")
    groups = {}
    by_id = {r["id"]: r for r in bank["records"]}
    for row in reach["records"]:
        key = f"M{row['M']}/N{row['N']}/r{row['capacity_ratio']:g}/{row['split']}/{row['label_kind']}/{row['stage']}"
        group = groups.setdefault(key, dict(states=0, originally_wrong=0, recovered_by_theta_search=0,
                                           reachable=0, analytic_unreachable=0, unknown=0))
        group["states"] += 1
        group[row["status"]] += 1
        record = by_id[row["id"]]
        group["originally_wrong"] += int(record["baseline_action"] not in record["target_actions"].tolist())
        group["recovered_by_theta_search"] += int(row.get("witness_source") == "legal_theta_search")
    report = dict(schema_version=1, data_source="synthetic_only", bank_sha256=file_sha256(bank_path),
        scope="Local state/action diagnostics; neither a production model nor measured benchmark gap",
        reachability=counts, reachability_by_group=groups, interpretations=hints,
        files=dict(bank=str(Path(bank_path).resolve()), reachability=str(Path(reach_path).resolve()),
            learning=str(Path(learning_path).resolve()) if learning_path else None,
            retention=str(Path(retention_path).resolve()) if retention_path else None))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "diagnosis.json", report)
    lines = ["# 同状态决策诊断", "", "仅使用独立合成状态。此报告不是 Chu gap 成绩，也不证明整条研究路线的上限。", "",
             "| 状态组 | 状态数 | 原目标未命中 | 可执行 θ 见证 | 搜索改对 | 解析局部不可达 | 搜索未决 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, g in sorted(groups.items()):
        lines.append(f"| {name} | {g['states']} | {g['originally_wrong']} | {g['reachable']} | {g['recovered_by_theta_search']} | {g['analytic_unreachable']} | {g['unknown']} |")
    lines += ["", "可达目标必须实际胜过整个可行动作集合，并按真实 tie-break 执行；解析排除有明确的局部作用域。搜索未决不能写成不可表达。", ""]
    if learning:
        lines += ["## Actor 拟合", "", "| M | split | 状态数 | 拟合前目标命中 | 拟合后目标命中 | 拟合后平均精确动作 regret | 平均真实 greedy 后缀收益变化 |", "|---|---|---:|---:|---:|---:|---:|"]
        for m, result in learning["models"].items():
            if result["status"] != "complete":
                continue
            for split in ("fit", "heldout"):
                before, after = result["before_summary"].get(split), result["after_summary"].get(split)
                if after:
                    lines.append(f"| {m} | {split} | {after['states']} | {before['target_hit_rate']:.3%} | {after['target_hit_rate']:.3%} | {after['mean_exact_action_regret']} | {after['mean_greedy_future_delta']} |")
        lines += ["", "| M | split | 原错误数 | 原错误改对数 | 原正确丢失数 |", "|---|---|---:|---:|---:|"]
        for m, result in learning["models"].items():
            for split, values in result.get("target_transitions", {}).items():
                lines.append(f"| {m} | {split} | {values['initial_errors']} | {values['corrected_initial_errors']} | {values['lost_initial_correct']} |")
        lines += ["", "目标命中率混合展示仅是诊断摘要；exact 与 finite_greedy 的细分在 learning.json 中。有限候选目标不是全局最优标签。heldout 标签从未参与拟合。", ""]
        for m, result in learning["models"].items():
            if "selected_step" in result:
                lines.append(f"- M{m}: selected_step={result['selected_step']}; stop_reason={result['stop_reason']}; {result['selection']}。")
        lines += ["", "finite_greedy 已知动作另存 frozen_continuation_gain（旧续行下第一步的收益变化）与 continuation_policy_drift（第一步相同时新旧续行的差异）；unknown 动作不虚构这两项。", ""]
    if retention:
        lines += ["## PPO 后保持", "", "| 实验组 | split | 初始正确数 | 末次保持率 | 相对拟合后平均后缀收益变化 |", "|---|---|---:|---:|---:|"]
        for name, result in retention["experiments"].items():
            if not result["history"]:
                continue
            for split, values in result["history"][-1]["retention"].items():
                lines.append(f"| {name} | {split} | {values['initially_correct']} | {values['target_retention_rate']} | {values['mean_future_delta_from_fit']} |")
        lines += ["", "两个分组从同一个拟合 Actor 和原 Critic 起步，使用新建诊断优化器及相同数据种子；策略分歧后并非相同轨迹。梯度方向只采样每次更新的首个实际 minibatch。", ""]
    lines += ["## 下一步判读", ""] + [f"- {hint}" for hint in hints]
    lines += ["", "- 可达但 fit 仍难学：定位表示、学习目标或优化；拟合目标可以执行，不应直接归因为评分不可表达。",
              "- fit 命中而 heldout 退步：检查状态覆盖、分布和策略依赖标签；小审计集不足以外推到 Chu。",
              "- 拟合后正确而 PPO 后丢失：比较 ppo_only / ppo_rank，检查同状态收益及梯度、KL；局部观察不自动证明唯一因果。",
              "- 高价值目标有解析排除：依据具体反例最小化修改固定评分；单个前缀不可达不等于整个实例不可达。",
              "- 全部诊断良好而外部评估停滞：才进一步检查训练分布、验证选择和跨规模泛化。", ""]
    (output / "diagnosis.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Report: {output / 'diagnosis.md'}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="phase", required=True)
    for name in ("build", "reach", "learn", "retention", "report", "all"):
        p = commands.add_parser(name)
        p.add_argument("--output", type=Path, required=True)
        if name != "report":
            p.add_argument("--device", default="auto")
        if name in ("build", "all"):
            p.add_argument("--checkpoint-dir", type=Path, default=Path(__file__).resolve().parent / "checkpoints/resource_strategy_v2")
            p.add_argument("--resources", type=int, nargs="+", default=(5, 10))
            p.add_argument("--items", type=int, nargs="+", default=(100,))
            p.add_argument("--instances-per-group", type=int, default=2)
            p.add_argument("--bank-seed", type=int, default=2729)
            p.add_argument("--exact-limit", type=int, default=16)
            p.add_argument("--exact-node-budget", type=int, default=200000)
            p.add_argument("--candidates", type=int, default=8)
        else:
            p.add_argument("--bank", type=Path, required=True)
        if name in ("reach", "all"):
            p.add_argument("--theta-restarts", type=int, default=4)
            p.add_argument("--theta-steps", type=int, default=100)
        if name in ("learn", "retention", "report"):
            p.add_argument("--reach", type=Path, required=True)
        if name in ("learn", "all"):
            p.add_argument("--fit-steps", type=int, default=100)
            p.add_argument("--fit-learning-rate", type=float, default=1e-4)
            p.add_argument("--fit-batch-size", type=int, default=8)
            p.add_argument("--label-scope", choices=("both", "exact", "finite_greedy"), default="both")
            p.add_argument("--fit-objective", choices=("regret_margin", "set_ce"), default="regret_margin")
            p.add_argument("--fit-anchor-coefficient", type=float, default=.1)
            p.add_argument("--fit-max-kl", type=float, default=.03)
            p.add_argument("--fit-evaluate-every", type=int, default=10)
        if name in ("retention", "report"):
            p.add_argument("--learning", type=Path, required=name == "retention")
        if name in ("retention", "all"):
            p.add_argument("--ppo-updates", type=int, default=5)
            p.add_argument("--ppo-seed", type=int, default=5729)
            p.add_argument("--control-mode", choices=("original", "guarded", "matched"), default="original")
            p.add_argument("--actor-learning-rate", type=float)
            p.add_argument("--actor-continuation-guard", action=argparse.BooleanOptionalAction, default=False)
            p.add_argument("--actor-guard-states", type=int, default=8)
            p.add_argument("--actor-guard-tolerance", type=float, default=0.0)
            p.add_argument("--validation-bank", type=Path)
            p.add_argument("--arms", choices=("ppo_only", "ppo_rank", "ppo_regret"), nargs="+",
                           default=("ppo_only", "ppo_rank", "ppo_regret"))
        if name == "report":
            p.add_argument("--retention", type=Path)
    args = parser.parse_args(argv)
    if args.phase == "report":
        return make_report(args.bank, args.reach, args.output, args.learning, args.retention)
    device = resolve_device(args.device)
    if args.phase == "all" and args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Full diagnostic output already contains files; use a new directory to preserve prior evidence")
    from mkp_audit_theta import run_reachability
    from mkp_audit_learning import run_learning, run_retention
    retention_options = (dict(seed=args.ppo_seed, control_mode=args.control_mode,
                              actor_learning_rate=args.actor_learning_rate,
                              actor_continuation_guard=args.actor_continuation_guard,
                              actor_guard_states=args.actor_guard_states,
                              actor_guard_tolerance=args.actor_guard_tolerance,
                              validation_bank_path=args.validation_bank)
                         if args.phase in ("all", "retention") else {})
    if args.phase == "all":
        from mkp_audit_bank import build_bank
        args.output.mkdir(parents=True, exist_ok=True)
        build_bank(args.checkpoint_dir, args.output / "bank", resources=tuple(args.resources), items=tuple(args.items),
            instances_per_group=args.instances_per_group, seed=args.bank_seed, exact_limit=args.exact_limit,
            exact_node_budget=args.exact_node_budget, candidates=args.candidates, device=device)
        bank = args.output / "bank/bank.pt"
        run_reachability(bank, args.output / "reach", device=device, restarts=args.theta_restarts, steps=args.theta_steps)
        reach = args.output / "reach/reachability.json"
        run_learning(bank, reach, args.output / "learn", device=device, steps=args.fit_steps,
            learning_rate=args.fit_learning_rate, batch_size=args.fit_batch_size, scope=args.label_scope,
            objective=args.fit_objective, anchor_coefficient=args.fit_anchor_coefficient,
            max_kl=args.fit_max_kl, evaluate_every=args.fit_evaluate_every)
        learning = args.output / "learn/learning.json"
        run_retention(bank, reach, learning, args.output / "retention", device=device,
                      updates=args.ppo_updates, arms=tuple(args.arms), **retention_options)
        return make_report(bank, reach, args.output, learning, args.output / "retention/retention.json")
    if args.phase == "build":
        from mkp_audit_bank import build_bank
        return build_bank(args.checkpoint_dir, args.output, resources=tuple(args.resources), items=tuple(args.items),
            instances_per_group=args.instances_per_group, seed=args.bank_seed, exact_limit=args.exact_limit,
            exact_node_budget=args.exact_node_budget, candidates=args.candidates, device=device)
    if args.phase == "reach":
        return run_reachability(args.bank, args.output, device=device, restarts=args.theta_restarts, steps=args.theta_steps)
    if args.phase == "learn":
        return run_learning(args.bank, args.reach, args.output, device=device, steps=args.fit_steps,
            learning_rate=args.fit_learning_rate, batch_size=args.fit_batch_size, scope=args.label_scope,
            objective=args.fit_objective, anchor_coefficient=args.fit_anchor_coefficient,
            max_kl=args.fit_max_kl, evaluate_every=args.fit_evaluate_every)
    return run_retention(args.bank, args.reach, args.learning, args.output, device=device,
                         updates=args.ppo_updates, arms=tuple(args.arms), **retention_options)


if __name__ == "__main__":
    main()
