# v1 诊断结论与学习目标修复

50次保持审计后的 Actor 小步与续行护栏修复、独立合成实例验证及三seed运行命令见 [ACTOR_RETENTION_V3.md](ACTOR_RETENTION_V3.md)。新功能需显式开启；旧证据保留，按新文档重建并核验共同起点。本次只完成接口和回归检查。

当前优先级是**控制器学习目标、共享策略变化和更新幅度**。本次证据不支持先扩大 θ 或先改网络，也不支持将主要退步归因于 PPO。

完整只读聚合见 `outputs/decision_audit_v1_analysis/summary.md` 和 `summary.json`，可用同目录 `summarize_json.py` 重建；未修改原 v1 产物。

| 已测结果 | 含义与限制 |
|---|---|
| 40 个原错误，38 个有合法 θ 见证，2 个 unknown；无解析排除 | 95% 原错误的局部目标已证可达；不证明共享网络能同时学会，也不证明全局最优目标可达 |
| 12 个 exact 全部是原本已正确的尾部 | 本次错误全为有限候选/旧策略续行代理目标，不能裁决全局表达上限 |
| 拟合目标命中 44/82→40/82；改对8个、丢失12个；58个后缀退步 | 100步拟合没有形成可靠策略改进 |
| M5 loss前10/后10均值2.2695→1.5515；M10 2.6383→1.9850 | 反向优化在工作，但代理loss与greedy终局收益不一致 |
| 拟合后19个动作价值未知，其中17个后缀退步；40个目标命中中仍有22个后缀退步 | 未知动作不能补零；第一步选对旧目标，也不能保证更新后的整条续行更好 |
| PPO+ranking相对拟合后恢复：M5 +130.125、M10 +72.214；PPO-only +66.975、+7.738 | 本次主要崩塌先发生于拟合；5次更新、原Critic、新优化器不足以排除长期PPO问题 |

原始收益数值来自合成前缀，不能直接相加或换算为 Chu gap。新版本尚未完成实际拟合、训练或评估，不承诺达到0.1%。

## 具体修改

1. **收益差加权的有限 margin**：生产 ranking 增加 `regret_margin`，旧 `soft_bce` 保留为消融。令 `x=(score_i-score_j)/score_scale`、`u=abs(delta)/return_scale`，每对损失为
   `min(u,1) * Huber(relu(min(u,margin_cap)-sign(delta)*x))`，Huber阈值为1。
   达到所需分差后不继续推动；收益差权重、所需margin及单对损失对分差的导数有界，避免极端错误分差放大二次梯度。网络参数梯度仍使用原全局裁剪。仍保留状态均衡、置信度、全可行动作KL和温度不变性。`delta` 仍是冻结策略续行估计，不升级成Q*。
2. **诊断只监督已知价值关系**：finite标签只比较已查询的较优目标与已查询的较差动作；未查询动作没有虚构的负价值标签。旧 `--fit-objective set_ce` 精确保留原全候选CE/最后迭代保存行为，以便匹配对照。
3. **行为锚定与fit选择**：新诊断损失加入源策略全可行动作KL；默认系数0.1。每10步检查全部fit状态的最大KL，超过0.03则停止；这是周期性候选检查与最终选择约束，不是每一步优化的硬信任域。候选按fit的真实归一化greedy后缀收益、命中率和label loss选择，包含未更新源Actor作为候选，回载最佳候选后才测heldout。这个保护只针对已测fit状态平均值，不保证heldout或Chu收益。
4. **保持试验分组清晰**：`ppo_only` 无ranking；`ppo_rank` 使用源checkpoint原ranking损失（旧checkpoint缺字段时识别为soft_bce）；`ppo_regret` 使用新margin。三组使用同一拟合Actor、原Critic及新优化器。正式PPO目标、环境奖励、Critic和推理解码均未改。

模型结构、θ范围与固定弱解码器评分保持一致。所有新增标签、搜索、KL拟合和续行检查仅用于训练/独立诊断；Benchmark仍只执行learned_dynamic单轨迹greedy，无物品学习打分头、搜索或修复。

## 首先由你运行匹配诊断

旧bank的源码SHA包含config，本次新增训练字段后会拒绝直接复用。保持严格校验，用相同checkpoint/seed重建新bank，再逐值核对状态和标签；不要改写旧bank元数据。

```powershell
$mkpPython = 'D:\Zlb08\.envs\RL_MKPtest\Scripts\python.exe'
& $mkpPython diagnose_mkp.py all --checkpoint-dir checkpoints/resource_strategy_v2 --output outputs/decision_audit_v2 --device cuda
& $mkpPython compare_audit_banks.py --before outputs/decision_audit_v1/bank/bank.pt --after outputs/decision_audit_v2/bank/bank.pt --output outputs/decision_audit_v2/bank_comparison.json
```

每条命令完成后检查退出码。全流程拒绝覆盖非空目录。比较脚本只读两个bank、不执行推理；状态/标签或源checkpoint不一致时退出2，不能将两轮结果写成同状态消融。

可在同一份v2 bank上再跑旧损失作为严格匹配控制，优化步数/学习率/种子均使用同样默认值：

```powershell
& $mkpPython diagnose_mkp.py learn --bank outputs/decision_audit_v2/bank/bank.pt --reach outputs/decision_audit_v2/reach/reachability.json --output outputs/decision_audit_v2/learn_ce_control --fit-objective set_ce --device cuda
```

重点看 `learn/learning.json` 中每个M的 `fit_history`、`selected_step`、`stop_reason`、原错误改对/原正确丢失、未知动作比例和heldout真实收益；保持数据在 `retention/retention.json`。不要把强保护后的fit不退步本身当作方法有效证据；需要确认确实改对错误并能泛化。

finite已知动作同时记录 `frozen_continuation_gain`（只换第一步、继续使用旧策略的收益变化）和 `continuation_policy_drift`（固定第一步后，新旧续行收益之差）。前者为正但后者负得更多，说明第一步模仿收益被后续策略变化抵消；未知动作不填充这些数值，exact Q*也不冒充旧策略续行。

| 新结果 | 下一步推论 |
|---|---|
| 错误修正增多、原正确丢失减少，heldout真实收益也改善 | 支持目标/漂移修复，进入完整独立训练 |
| selected_step=0 | 保存了原模型，修复尚未产生可接受改进；检查fit_history是KL过紧还是收益候选均退步 |
| KL过早超限 | 用新输出目录降低fit学习率至1e-5并将fit检查间隔设为1；不要直接放宽预算接受退步 |
| loss/命中改善但整段收益不升 | 仍有策略依赖续行目标错配，下一步检查更密的多步状态覆盖与刷新标签 |
| fit改善但heldout退步 | 小样本/覆盖或共享控制泛化问题；扩大独立合成覆盖，再考虑网络表示 |
| ppo_regret比原ranking更好保持且真实收益改善 | 支持新生产ranking；5更结果仍不能替代长期多种子训练 |

## 完整训练和评估由你执行

确认局部实验有实质收益后，使用独立目录从头训练新损失；诊断Actor不能用于正式Benchmark。

```powershell
& $mkpPython train_ConstrainedMKP.py --resources 5 10 --updates 2000 --seed 20260910 --ranking-loss-type regret_margin --checkpoint-dir checkpoints/regret_margin_v2 --runs-dir runs/regret_margin_v2 --device cuda
& $mkpPython Benchmark_zeroShot.py --checkpoint-dir checkpoints/regret_margin_v2 --device cuda
& $mkpPython compare_mkp_results.py --baseline results/20261022_090713_537783 --candidate results/替换为新评估目录
```

新旧损失完整消融：上面的训练命令改为 `--ranking-loss-type soft_bce` 并使用另外的checkpoint/runs目录，保持其余设置相同。需要多种子和同预算评估才能判断稳定改善。旧soft_bce训练的中断恢复必须显式带 `--ranking-loss-type soft_bce`；新损失应开始独立实验，不能静默切换旧optimizer运行的语义。

训练和Benchmark默认目录也已改为 `checkpoints/regret_margin_v2`，避免默认新训练覆盖原 `resource_strategy_v2` 权重。诊断命令仍显式读取原权重，网络/checkpoint schema版本保持兼容。

全量接口回归也由你运行：

```powershell
& $mkpPython -m unittest discover -s tests -v
```
