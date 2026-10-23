# 50次审计后的最小保持修复

本次只实现接口与回归检查，未运行下面的拟合、保持实验、正式训练或 Benchmark。新方案没有已测的收益或 Chu gap 改善。

已完成CPU回归：179项，175通过、4项CUDA检查因本次禁用GPU而跳过；另通过语法与diff空白检查。测试使用tiny fixture与mock实验入口，不代表新保持方案已有效。

50次审计的主要问题是后段收益回吐：M10 regret 虽有41/50次优于 PPO-only，第45–50次 heldout 收益却连续为负；M5 最近10次也落后于 PPO-only。原有少量正收益不能替代其余独立实例的验证。

## 具体修改与作用域

1. `--actor-learning-rate 1e-5` 仅降低 Actor 的 AdamW 学习率；Critic 仍使用原 `learning_rate=1e-4`。不改网络、theta、固定评分公式、ranking margin/权重、环境奖励、GAE、Critic 目标或单轨迹 greedy 推理。
2. `--actor-continuation-guard` 在**每个生产 PPO update**开始时，从本次训练 rollout 按实例和早/中/后阶段确定性选择最多8个可行动作前缀；没有 heldout 输入。更新前后都从相同物理状态执行完整动态 greedy 续行，float64 累计新选物品利润，再除以原实例全部物品利润之和。
3. 候选 Actor 的平均归一化后缀收益下降超过 `--actor-guard-tolerance`（建议0），或同一组前缀上全可行动作的 `KL(old || new)` 最大值超过原 `target_kl=0.03`，则恢复本次更新前的 Actor 权重和其 Adam step/momentum。Critic 已执行的更新保留。没有额外搜索、修复、oracle 标签或候选择优推理。
4. 护栏只约束本次少量训练前缀，**不保证每个前缀、独立验证集或长期收益不退**。记录最差单状态变化、下降状态数和拒绝原因。独立 Critic 仍参与既有全模型梯度裁剪，不能声称各组 Critic 的训练轨迹完全相同。

默认 `actor_learning_rate=None`、护栏关闭，保留原单组 AdamW 与原 PPO 行为；新方案通过显式开关启用。开启单独 Actor 学习率时才建立 Actor/Critic 两个参数组。正式 resume 要求这些设置与原 checkpoint 一致，旧单组运行不能静默切成新方案。

`PPO_seconds` 包含更新后的护栏检查；护栏开始时的基线续行也计入该字段。增加了额外训练期开销，本次没有测量端到端速度。

## 原证据保留，新目录重建共同起点

源码 SHA 校验继续严格执行。新增训练字段后旧 bank 会拒绝被当前源码直接用于实验。不要改写旧元数据或关闭校验。

下面重建相同 seed/状态/标签和相同小步拟合，再逐值比较 bank 和 Actor。比较工具只读取张量与 JSON；新旧 `.pt` 文件 SHA 可以不同，但 Actor tensor 必须完全相同。每条命令失败时停止，保留原产物。

在仓库目录执行：

```powershell
$mkpPython = 'D:\Zlb08\.envs\RL_MKPtest\Scripts\python.exe'

& $mkpPython diagnose_mkp.py build --checkpoint-dir checkpoints/resource_strategy_v2 --output outputs/decision_audit_v3_guard/bank --bank-seed 2729 --instances-per-group 2 --device cuda
if ($LASTEXITCODE -ne 0) { throw 'build failed' }

& $mkpPython diagnose_mkp.py reach --bank outputs/decision_audit_v3_guard/bank/bank.pt --output outputs/decision_audit_v3_guard/reach --device cuda
if ($LASTEXITCODE -ne 0) { throw 'reach failed' }

& $mkpPython diagnose_mkp.py learn --bank outputs/decision_audit_v3_guard/bank/bank.pt --reach outputs/decision_audit_v3_guard/reach/reachability.json --output outputs/decision_audit_v3_guard/learn --fit-steps 100 --fit-learning-rate 1e-5 --fit-evaluate-every 1 --device cuda
if ($LASTEXITCODE -ne 0) { throw 'learn failed' }

& $mkpPython compare_audit_banks.py --before outputs/decision_audit_v2/bank/bank.pt --after outputs/decision_audit_v3_guard/bank/bank.pt --before-learning outputs/decision_audit_v2_smallstep/learn/learning.json --after-learning outputs/decision_audit_v3_guard/learn/learning.json --output outputs/decision_audit_v3_guard/start_comparison.json
if ($LASTEXITCODE -ne 0) { throw 'Bank or Actor differs; do not claim the old and new experiments share the same start' }
```

比较退出0才表示共同起点核验通过；退出2表示不匹配或证据无效。旧 Actor 不直接 rebind 到新 bank，也不用于正式 resume。保留旧 `retention_50` 不变。

## 独立合成实例，只测量不更新

用相同源 checkpoint、M/N/容量分组，换独立 bank seed `62729`，每组实例数由2增至4：

```powershell
& $mkpPython diagnose_mkp.py build --checkpoint-dir checkpoints/resource_strategy_v2 --output outputs/decision_audit_v3_guard/independent --bank-seed 62729 --instances-per-group 4 --device cuda
if ($LASTEXITCODE -ne 0) { throw 'Independent bank build failed' }
```

默认 M5/M10、N100、3种容量比例，两种生成 split 各4实例，共48个独立实例。**该 bank 的所有实例均转为观测专用 heldout**；原 split 只作为元数据保存。不会在该 bank 上运行 fit 或 reach，也不按 reachability/标签好坏筛选观测状态。程序同时拒绝与拟合 bank 重用的 seed、生成 entropy、实例 ID 或物理实例张量 SHA。

除来源 greedy 轨迹前缀，还从每个独立实例的空初态执行一次完整 greedy，直接记录整实例 objective。这些实例不进入 PPO batch、训练前缀护栏、Actor 拟合、Critic 目标、回滚决策或最佳模型选择；测量前后恢复 Torch/NumPy/Python RNG。此验证是合成数据，仍不是 Chu 测试。当前独立测量逐前缀、逐实例完成 greedy，每次更新都测，四组乘三个seed会增加运行时间；本次没有测量耗时或实现批量测量优化。

## 三个 PPO seed、四组匹配保持

以下实验始终使用同一个重新核验的 Actor、原 Critic 与新优化器，以及同一份独立验证实例。每个 seed 固定运行50次，保留完整过程和末次，不能根据结果挑最好 update。

```powershell
foreach ($ppoSeed in @(5729, 5730, 5731)) {
    & $mkpPython diagnose_mkp.py retention --bank outputs/decision_audit_v3_guard/bank/bank.pt --reach outputs/decision_audit_v3_guard/reach/reachability.json --learning outputs/decision_audit_v3_guard/learn/learning.json --validation-bank outputs/decision_audit_v3_guard/independent/bank.pt --control-mode matched --arms ppo_only ppo_regret --actor-learning-rate 1e-5 --actor-continuation-guard --actor-guard-states 8 --actor-guard-tolerance 0 --ppo-seed $ppoSeed --ppo-updates 50 --output "outputs/decision_audit_v3_guard/retention_seed_$ppoSeed" --device cuda
    if ($LASTEXITCODE -ne 0) { throw "Retention seed $ppoSeed failed" }
}
```

`matched` 固定运行四组；每个 M 输出：

| 组名 | ranking | Actor 学习率 | 续行护栏 |
|---|---|---:|---|
| `original_only` | 关闭 | 1e-4 | 关闭 |
| `original_regret` | 原 regret_margin、系数0.1 | 1e-4 | 关闭 |
| `guarded_only` | 关闭 | 1e-5 | 开启 |
| `guarded_regret` | 原 regret_margin、系数0.1 | 1e-5 | 开启 |

这是“较小 Actor 步幅 + 训练续行护栏”的组合修复对照；不能分别归因两个组件。需要单因素消融时可用 `--control-mode guarded --actor-learning-rate 1e-5 --no-actor-continuation-guard` 单独测步幅，或省略 Actor 学习率而开启护栏；使用新的输出目录。

`--control-mode original` 保留旧臂名；它拒绝额外 Actor 学习率/护栏设置，防止参数被悄悄忽略。`matched` 的组固定为上表，`--arms` 不改变四组定义。每个输出目录必须为空。不同组分歧后轨迹不同，guard回滚也会改变实际 Actor 更新量；需要报告实际接受情况和成本。

## 查看哪些字段，如何决定是否值得全训

每个 seed 的 `retention.json` 保留：

- `experiments["M*/组名"].training_controls`：Actor/Critic 实际学习率、guard开关、阈值；同一实验对象另存 `source_checkpoint_sha256` 和 `fitted_actor_sha256`。
- `history[*].training_metrics.actor_guard_*`：候选策略的收益变化、最大 KL、拒绝原因、最差前缀及退步状态数。`after_return/delta` 是**回滚前候选**观测；拒绝后实际 Actor 已回到更新前。
- `minibatches`：尝试执行次数，Critic 已执行；`actor_minibatches_applied`：本次 Actor 最终保留的次数，整次回滚为0。Loss、梯度和 ranking minibatches 仍描述尝试过程。
- `history[*].validation.instances` / `instance_summary`：独立实例整局收益，相对 source 与拟合后起点的变化，按实例等权。
- `history[*].validation.states`：独立前缀标签命中、精确/有限 regret 及真实续行收益。未知动作价值继续留空，不虚构 Q*。
- `validation.final`、`whole_trajectory_instance_mean`、`last10_instance_mean`、`last20_instance_mean`：固定末次和时间窗口摘要；窗口实际 update 数也保存。

先在每个 seed、每个 M 下比较 `guarded_regret` 与 `original_regret`，判断修复是否改善后段；再比较 `guarded_regret` 与 `guarded_only`，判断在相同修复下 regret 是否有额外价值。联合看容量分组、逐独立实例分布和旧前缀回吐，不能只平均所有状态或只挑一个 seed。

若 guard 几乎每次都拒绝、Actor 接受步数接近0，即便训练 probe 不退，也只能说明更新被挡住，不能称学习有效。应看到独立整实例收益和最后10/20次走势改善，同时保有实质接受更新；再决定是否投入全训。没有从本次接口测试推定任何固定成功阈值。

## 后续正式训练（验证通过后由你执行）

下面是独立从头实验，不是载入诊断 Actor，也不是旧 soft_bce checkpoint 切换语义。先以同seed和同预算比较两组；稳定性仍需多seed。

```powershell
& $mkpPython train_ConstrainedMKP.py --resources 5 10 --updates 2000 --seed 20260910 --ranking-loss-type regret_margin --actor-learning-rate 1e-5 --actor-continuation-guard --actor-guard-states 8 --actor-guard-tolerance 0 --checkpoint-dir checkpoints/actor_guard_v3_regret_s20260910 --runs-dir runs/actor_guard_v3_regret_s20260910 --device cuda

& $mkpPython train_ConstrainedMKP.py --resources 5 10 --updates 2000 --seed 20260910 --ranking-coefficient 0 --actor-learning-rate 1e-5 --actor-continuation-guard --actor-guard-states 8 --actor-guard-tolerance 0 --checkpoint-dir checkpoints/actor_guard_v3_only_s20260910 --runs-dir runs/actor_guard_v3_only_s20260910 --device cuda
```

正式训练仍只按独立 synthetic validation 保存 best，不使用此审计 bank 选模型。单轨迹 Benchmark 对两个目录分别使用 `--checkpoint-tag best`（以及报告 last 以查看回吐）：

```powershell
& $mkpPython Benchmark_zeroShot.py --checkpoint-dir checkpoints/actor_guard_v3_regret_s20260910 --checkpoint-tag best --device cuda
& $mkpPython Benchmark_zeroShot.py --checkpoint-dir checkpoints/actor_guard_v3_only_s20260910 --checkpoint-tag best --device cuda
```

Benchmark 自动输出结果目录；随后把实际路径填入比较命令，不要将诊断 Actor 用于 Benchmark：

```powershell
& $mkpPython compare_mkp_results.py --baseline results/替换为本次PPO_only结果目录 --candidate results/替换为本次regret结果目录
```

完整训练、独立保持和正式评估都由你执行。`Actor → theta → 固定弱解码器 → 可行动作` 的推理边界保持不变。
