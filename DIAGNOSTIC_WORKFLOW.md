# 同状态决策诊断：先区分瓶颈，再决定是否重训

50次保持审计后的最小修复与独立实例保持验证请使用 [ACTOR_RETENTION_V3.md](ACTOR_RETENTION_V3.md)，包括固定共同起点、四组对照及三个PPO seed。旧 bank 的源码 SHA 不作绕过。

**v1已完成后的更新：** 当前学习目标与模型选择已修复，默认fit使用已知收益margin、行为KL和fit真实收益选择，PPO保持新增 `ppo_regret` 组。已完成v1的分析和新命令以 [LEARNING_REPAIR_V2.md](LEARNING_REPAIR_V2.md) 为准；请使用新的 `outputs/decision_audit_v2` 目录，下面v1命令保留作原流程记录。全流程现在拒绝覆盖非空目录。

这套工具用于判断当前 v5 固定评分族、高层控制器学习和 PPO 更新保持分别有什么限制。默认读取现有 `checkpoints/resource_strategy_v2/mkp_M5_best.pt` 与 `mkp_M10_best.pt`，无需先启动新一轮全量训练。

正式推理只保留 `learned_dynamic`：每次选择前重新生成 θ，再由原固定弱解码器选一个可行物品。`fixed`、`learned_static`、采样择优、对应比较列及训练固定基线均已移除。诊断中的 θ 搜索、精确标签、Actor 拟合和额外续行只存在于独立工具，不进入 Benchmark。

## 建议首先运行

在项目目录使用原 Python 环境。以下命令由你执行；它会进行实际诊断搜索和局部拟合，但不是 2000-update 正式训练，也不会写回原 checkpoint。

```powershell
$mkpPython = 'D:\Zlb08\.envs\RL_MKPtest\Scripts\python.exe'
& $mkpPython diagnose_mkp.py all --checkpoint-dir checkpoints/resource_strategy_v2 --output outputs/decision_audit_v1 --device cuda
```

每条命令完成后检查退出码；异常时保留输出目录和错误日志，不把部分阶段当成完整结论。再次从头构建请使用新的输出目录，避免混用不同状态集或预算。

默认预算为：M5/M10、N100、容量比例 0.25/0.5/0.75，fit/heldout 各每组 2 个不同实例；每条 greedy 轨迹选早/中/后段及实际出现的精确尾部，重复索引去重。θ 搜索每个待查状态至多 4 次重启、每次 100 步；Actor 仅在 fit 状态拟合 100 步；随后分别进行 5 次真实 PPO-only 和 PPO+ranking 更新，检查保持。实际状态数、缺失尾部分组和实际计算量写入报告，不保证每条轨迹都有指定的精确尾部。

工具会校验源 checkpoint、状态集和相关代码的 SHA，fit 与 heldout 来自不同合成实例；没有 Chu 数据入口。输出的诊断 Actor 文件有专门 artifact 类型，不是正式 checkpoint，不能用于 Benchmark 或冒充新的 gap 成绩。

## 分阶段运行与文件

需要分开运行或改变后续预算时使用以下命令，保持同一份 bank：

```powershell
& $mkpPython diagnose_mkp.py build --checkpoint-dir checkpoints/resource_strategy_v2 --output outputs/decision_audit_v1/bank --device cuda
& $mkpPython diagnose_mkp.py reach --bank outputs/decision_audit_v1/bank/bank.pt --output outputs/decision_audit_v1/reach --device cuda
& $mkpPython diagnose_mkp.py learn --bank outputs/decision_audit_v1/bank/bank.pt --reach outputs/decision_audit_v1/reach/reachability.json --output outputs/decision_audit_v1/learn --device cuda
& $mkpPython diagnose_mkp.py retention --bank outputs/decision_audit_v1/bank/bank.pt --reach outputs/decision_audit_v1/reach/reachability.json --learning outputs/decision_audit_v1/learn/learning.json --output outputs/decision_audit_v1/retention --device cuda
& $mkpPython diagnose_mkp.py report --bank outputs/decision_audit_v1/bank/bank.pt --reach outputs/decision_audit_v1/reach/reachability.json --learning outputs/decision_audit_v1/learn/learning.json --retention outputs/decision_audit_v1/retention/retention.json --output outputs/decision_audit_v1
```

| 阶段 | 主要文件 | 需要分析的内容 |
|---|---|---|
| build | `bank/bank.pt`、`bank_summary.json` | M/N/容量/阶段/分割覆盖；精确与有限标签数量；精确计算预算是否耗尽 |
| reach | `reach/reachability.json`、`witnesses.pt` | 当前控制器已正确、搜索找到可执行 θ、解析局部排除、未决分别有多少 |
| learn | `learn/learning.json`、`actor_M*.pt` | 只对可达目标拟合；fit 和 heldout 前后命中、精确动作 regret、真实 greedy 后缀收益、原错误改对和原正确丢失 |
| retention | `retention/retention.json` | PPO-only/PPO+ranking 后的目标保持率、同前缀真实收益、分支梯度范数/夹角、实际 KL 停止原因 |
| report | `diagnosis.md`、`diagnosis.json` | 各分组摘要与下一步判读提示；仍应查看逐状态记录 |

`witnesses.pt` 中的 θ 仅是诊断见证，不是推理时使用的策略。搜索对比整个可行动作集合，并核对真正执行的 argmax 和 tie-break；合法 θ 的成功说明这个状态的目标可以执行，不说明控制器已学会。

## 标签是什么

精确尾部要求当前可行物品数不超过 `--exact-limit 16`，并在 `--exact-node-budget 200000` 内完成完整求解。只要预算耗尽，全部部分精确结果作废，记录预算失败，再退回有限候选标签；不会把尚未穷尽的搜索称为精确最优。

精确标签提供所有可行动作的最优剩余收益，接受全部最优动作中的任何一个。有限标签只比较最多 `--candidates 8` 个提议动作在冻结旧策略 greedy 续行下的收益，其余动作价值未知；它不是 Q*，也不把未知值补成零。实际旧 greedy 后缀收益另行记录，避免混淆精确动作价值与旧策略真正获得的终局收益。

解析排除给出固定评分族在所有合法 θ 下的分差上界，包括价格/稀缺的正加权成本比、标量范围与交互/搭配的总质量约束。报告注明实数评分公式及数值安全余量的作用域；它是某个状态/目标集合的局部排除，不是整实例最优性证书。其他非凸搜索未找到 θ 一律为 `unknown`，不能写成不可表达。

Actor 拟合是诊断：每个状态独立可达，不保证同一个网络同时学会所有目标。只在 fit 状态更新 Actor，不更新 Critic；heldout 仅测量。`finite_greedy` 使用候选改进目标的模仿损失，不会将其称为全动作最优监督。若只想先检查严格精确目标，可给 all/learn 加 `--label-scope exact`。

PPO 保持试验直接调用当前 `ppo_update`，不复制另一份近似训练实现。两个分组从相同的诊断 Actor、原始 Critic和新建优化器起步，按相同数据种子运行。策略分歧后轨迹并非逐位相同；它不是中断训练的 resume。每次更新仅对首个执行 minibatch 分解 Actor 梯度，KL 停止事件另存其真实触发值。梯度观察使用 `autograd.grad`，不写入参数 `.grad`、不改变默认训练目标。

## 结果对应的明确推论

| 观察结果 | 支持的判断 | 后续行动 |
|---|---|---|
| 有价值的精确最优目标存在解析排除 | 该状态的固定评分族有表达缺口 | 检查排除项、阶段与损失权重，最小化修改固定评分基/状态级控制，再单独消融 |
| 多数失败状态仍是 unknown | 当前搜索/上界不足以判断可达性 | 对同一 bank 增加搜索预算或改进界；暂不据此否定路线 |
| 找到 θ 见证，但 fit 原错误仍改不对 | 独立可达性成立，当前共享网络/学习或优化仍受限 | 检查 set loss、动作 margin、梯度及状态特征区分度；不先继续扩大评分族 |
| fit 改对、heldout 命中/收益未改善或变差 | 审计集拟合与泛化存在落差 | 分层增加独立合成状态覆盖，检查策略依赖标签；不把 fit 成功当成 Chu 改善 |
| Actor 拟合后正确，PPO 后明显丢失 | 更新保持是实测问题 | 比较两组的丢失状态、真实收益、梯度夹角与 KL，考虑隔离更新或保持机制的受控试验 |
| PPO-only 保持、PPO+ranking 丢失 | ranking 参与后的差异值得优先排查 | 查标签目标、梯度及触发位置；样本量和策略分歧仍限制因果外推 |
| 两组都丢失 | 不能只把问题归给 ranking | 检查 PPO/entropy/共享裁剪、原 Critic 与策略变化，以及采样状态覆盖 |
| 可达、fit、heldout 与保持都良好，但正式 gap 仍平台 | 优先怀疑审计覆盖、分布与验证选择 | 扩展审计分层并核对正式单轨迹评估；再做多种子同预算消融 |

成功率需按 exact/finite、M、容量和阶段分别看；小审计集不设一个万能百分比来裁决研究路线。单个已访问前缀不可达，也不证明这个实例的其他选择路径都不可达。各状态 regret/后缀收益不能直接相加冒充平均 Chu gap，因为这些前缀可能属于同一条轨迹。

## 增加诊断预算与后续正式实验

例如，仅增加 θ 搜索预算，保留原 bank：

```powershell
& $mkpPython diagnose_mkp.py reach --bank outputs/decision_audit_v1/bank/bank.pt --output outputs/decision_audit_v1/reach_larger --theta-restarts 8 --theta-steps 300 --device cuda
```

后续 learn/retention/report 必须显式传入这份新的 reachability.json，输出新目录，避免旧结果混入。

当前没有根据未知诊断结果预先改网络或评分公式。先把 `diagnosis.md` 和对应 JSON 返回分析，再决定最小代码改动。确定改动后，完整训练与单轨迹评估由你执行，例如独立新目录：

```powershell
& $mkpPython train_ConstrainedMKP.py --resources 5 10 --updates 2000 --device cuda --checkpoint-dir checkpoints/诊断后方案
& $mkpPython Benchmark_zeroShot.py --checkpoint-dir checkpoints/诊断后方案 --device cuda
& $mkpPython compare_mkp_results.py --candidate results/新评估目录
```

不要直接将诊断 actor_M*.pt 用于正式评估。默认 N100 训练的跨 N 测试边界保持明确；如果将 build/fit 扩展至 N250/500，它是新的诊断覆盖范围，不能将该拟合 Actor 宣称为 N100 训练得到的零样本策略。

全量接口回归由你运行：

```powershell
& $mkpPython -m unittest discover -s tests -v
```
