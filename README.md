# Transformer 高层策略控制的强化学习 MKP

实现路径：**动态 MKP 状态 → 二分图 Transformer → 高层策略 θ → 固定弱解码器 → 可行物品 → 状态更新**。

目标是在 `MKP_dataset/Benchmarks_Chu.mat` 上研究接近或低于 **0.1% gap** 的解。该目标需要实验验证，代码交付本身不代表达到该精度。默认训练与测试均由使用者运行。

当前为 **combination θ v5**：在上一版的关系感知 Actor 上，扩展 `StrategyTheta` 和透明评分基函数，使网络识别到的资源联动与两物品搭配能够直接影响评分。checkpoint schema 为 5，控制器标识 `combination_theta_v5`，解码器标识 `second_order_density_v5`；v1–v4 checkpoint 明确拒绝加载，需要重新训练，默认保存至 `checkpoints/combination_v5/`。

## PPO + 训练期 θ 候选反事实 rollout + 动作排序监督

默认训练机制现为 **`theta_action_rollout_v1`**。本次只修改训练监督的生成方式，沿用现有 v5 网络、θ 定义和固定弱 decoder。**监督目标是 θ 所诱导动作的后续利润顺序，不回归候选 θ 数值，也不惩罚 θ 的变化幅度。** 多组 θ 只要产生同一种好动作，便属于同一个动作监督目标。

每次 PPO 更新前，使用尚未更新参数的当前策略作为冻结后续策略 μ，按以下流程生成标签：

1. 从本轮独立生成的合成实例的正常 PPO 轨迹中抽取最多 64 个已访问状态，排除不足两个可行动作的状态。标签不使用 Chu、最佳已知解或固定验证集。
2. 每个状态保留当前 θ，再生成 7 个合法 θ：默认 4 个在当前控制参数的 logit 空间局部扰动，3 个在合法范围内较广采样。资源价格满足下界和归一化；二阶资源系数保持对称、非负及上三角总幅度上界；搭配系数保持非负及总幅度上界；关闭的系数始终为 0。温度固定，因为它不改变当前 greedy 动作。
3. 每组 θ 都经过**同一个原 decoder**得到 greedy 动作。仅保留确实诱导出不同动作的状态，最多选择 16 个：约 3/4 优先取当前评分差较小的状态，其余随机取。按动作去重，保留当前动作以及最多 3 个不同动作；超预算时在不同动作之间随机抽取，发生在读取任何续行收益之前。同一动作有多个 θ 见证不会得到更大权重。
4. 从精确复制的状态分别执行每个动作，更新剩余容量、已选标记和可行集合，再用同一个冻结 μ 续行到终局。**只有第一步动作受候选 θ 影响，后续每步都重新读取真实状态，由 μ 生成新的 θ。** 每个不同动作只执行 1 条 greedy 续行和 4 条随机续行；同状态、同重复、同相对步骤的各动作分支共享按原始物品编号生成的 Gumbel 噪声。
5. 每个备选动作都与当前动作比较，同时保留可靠的改进与退化样本。不会直接把若干次随机续行中的最大收益当标签，也不会为每个候选 θ 重复评估相同动作。标签生成完成后才执行 PPO 参数更新。

原始后续利润定义为：

```text
Q_mu(s,a) = profit(a) + 从真实后继状态按冻结 μ 新获得的利润
          = sum(profits * (terminal_selected & ~selected_at_s))
Delta     = [0.5 * (Q_greedy(s,a0) - Q_greedy(s,a))
             + 0.5 * mean_r(Q_sample_r(s,a0) - Q_sample_r(s,a))] / sum(instance_profits)
```

`a0` 是当前 θ 的动作。直接统计新增物品利润，排除到达 s 之前已获得的共同前缀收益；先用 float64 计算差值，再按该实例的总利润归一化，与 PPO 奖励尺度一致。默认 μ 的比较目标是 greedy 续行与当前策略随机续行的等权平均；后者使用配对差值的标准误。保留 `|Delta| > max(ranking_min_gap, ranking_se_multiplier * SE)` 的样本，并过滤 greedy 与随机均值明显冲突的方向。**这是有限采样下的相对收益估计和过滤规则，不是最优 Q、统计显著性保证或新策略整体改进保证。**

训练使用当前可学习控制器重新计算原 decoder 的分数：

```text
target       = sigmoid(Delta / ranking_return_scale)
margin       = (score_theta(s,a0) - score_theta(s,a)) / ranking_score_scale
ranking_loss = BCEWithLogits(margin, target)
total_loss   = 原 PPO / Value / 可行性辅助 / Entropy 损失 + 0.1 * ranking_loss
```

`Delta < 0` 表示备选动作更好，监督降低当前动作相对它的分数。排序使用温度缩放前的分数，梯度经过固定评分算式、当前 θ 和 Actor encoder；候选 θ、采样动作、续行和标签均无梯度。独立 Critic 仍只使用 PPO 轨迹的目标，反事实分支不进入 PPO buffer/GAE。保留全候选 KL 提前停止机制和独立的标签 RNG。PPO 仍负责整体利润，排序损失帮助识别同一状态下不同选择的后果。

推理仍然是 `状态 → 单个 θ → 原弱 decoder → 一个动作`。新增候选生成、分支续行、标签筛选均只在训练期运行，没有新增可训练的 decoder 模块、逐物品 Q 头、修复或推理搜索。候选控制参数的维度与 N 无关；资源价格/系数按合法域采样，不把物品编号作为学习特征。保留跨 N 的接口与独立测试边界，**泛化是否改善仍需 N=250/500 等正式评估验证**。

### 运行、对照与预算

```powershell
# 新机制；建议使用独立目录，与已有 v5 训练结果区分
python train_ConstrainedMKP.py --ranking-mode theta --checkpoint-dir checkpoints/theta_ranking_v1
python Benchmark_zeroShot.py --checkpoint-dir checkpoints/theta_ranking_v1
# 前一版：只按当前评分选择相近物品对，再做冻结续行
python train_ConstrainedMKP.py --ranking-mode score_pairs --checkpoint-dir checkpoints/score_pair_control
# 保留现有网络/解码器，只用原 PPO 及已有辅助目标
python train_ConstrainedMKP.py --ranking-coefficient 0 --checkpoint-dir checkpoints/ppo_control
# 减少训练标注预算
python train_ConstrainedMKP.py --ranking-states 8 --ranking-theta-candidates 6 --ranking-max-actions 2 --ranking-repeats 2 --checkpoint-dir checkpoints/theta_small_budget
```

候选数量 `--ranking-theta-candidates 8` 包含当前 θ；`--ranking-theta-local-scale 1` 控制局部扰动；`--ranking-max-actions 4` 控制去重后的动作上限。默认最多为 `16*4*(1+4)=320` 条续行轨迹、`16*(4-1)=48` 个比较，实际数量受动作去重和过滤影响。原两动作机制最多为 160 条，因此**此前约 2.9× 的局部计算提速不能用于推断这次新增标注后的总训练时间**。24GB profile、固定特征缓存和 GPU 分块继续生效；候选 θ 重用固定几何特征，但每个 θ 都调用原 decoder 算式。

本次不改 schema/model_config，已有 v5 checkpoint 可 `--resume`；切换监督机制会记录 `RANKING_PROTOCOL_CHANGE`，不会把它描述为相同配置的精确续跑。`config.json`、checkpoint 和 Benchmark manifest 记录 `ranking_protocol`。公平对照应使用相同种子和模型配置，并同时报告训练更新数、续行预算、实际耗时和真实收益/gap；新增候选并不保证一定发现改进。

### 可复核记录

`THETA_ACTIONS` 日志及 CSV 增加：候选 θ 评估数、去重动作数、无替代动作的状态数、实际续行分支数、可靠改进对/状态数，以及相对当前动作的平均归一化/原始利润增量。这些增量来自冻结 μ 的训练标签，不能作为训练后 greedy 策略或 Chu gap 的改善结果。

默认在 update=1 及每 100 次更新保存 `runs/<本次运行>/ranking_audit/M5_update000100.pt` 等快照；`--ranking-audit-every 0` 可关闭。仅当存在不同动作供比较时写文件。快照包括**更新前模型权重、真实状态、θ 候选与动作见证、逐重复的原始 Q、配对差值/标准误/保留掩码、续行 RNG 状态及配置**。有快照即可重新计算动作和完整标签：

```powershell
python verify_ranking_audit.py runs/<本次运行>/ranking_audit/M5_update000100.pt
```

验证默认使用快照原设备类型，CPU/CUDA 随机数后端必须一致；建议保留原 PyTorch/CUDA 环境。验证工具检查当前 θ、候选动作、去重动作见证、greedy/随机续行利润和过滤结果，一项不符即报错；只复现冻结标签，不执行优化器更新。快照额外占用磁盘和周期性复制时间，其写入耗时包含在 `ranking_label_seconds`。

## 24GB GPU 训练速度优化

默认 `--performance-profile auto`：检测到至少 20GiB 的 CUDA 设备时启用 `memory24`，CPU/较小显存设备使用 `reference`。本机检测结果为 **RTX 5090 Laptop GPU，24463MiB 专用显存**，直接运行原训练命令即可启用。该速度优化只调整执行方式，保留 v5 θ/弱解码器、独立 Critic、float32 网络与 float64 容量计算，以及 batch=64、PPO minibatch=128、3 epochs；不会自行修改已配置的排序预算。最新 θ 候选标注预算见上一节。

| 修改               | 具体作用                                                                                                                                                           |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 扩大 GPU 分块      | 关系特征及注意力查询块从 32 提至至少 128；解码器机会项/搭配项的候选块从 128/64 提至至少 4096，减少 Python 循环和小 CUDA kernel 调度；同样用于训练时的冻结策略续行  |
| PPO 状态特征缓存   | 每次更新为有效状态构建一次固定特征，跨 minibatch/epoch 复用；包含基础状态特征、候选关系、机会损失、二阶负载和搭配基函数；Actor/Critic 共用相同基础输入，参数仍独立 |
| 保留梯度通路       | 缓存仅保存无参数、无梯度的几何量；神经网络表示、θ、候选分数和概率每次重新计算。状态张量原地修改或替换后缓存自动失效；切片保持状态与特征对齐                        |
| 减少同步及增加计时 | PPO 指标在 GPU 累加后一次回传；日志拆分 rollout、排序标签、PPO/缓存构建耗时，并记录本进程训练阶段显存峰值                                                          |

缓存默认上限 `--feature-cache-gib 6`，并在当时的空闲显存中预留 `--cache-headroom-gib 6` 给后续计算；按 `--feature-cache-batch-size 64` 批量构建。缓存预算包括固定特征及状态副本，超过预算或缓存分配 OOM 时回退到逐 minibatch 计算。**这不是整个训练进程的显存硬上限**：神经网络激活、梯度、排序续行和其他进程另占显存，N 增大时关系特征仍为二次增长。已有缓存够用时，继续提高上限不会额外分配，也不必填满 24GB。

```powershell
# 默认自动启用；也可显式指定
python train_ConstrainedMKP.py --performance-profile memory24
# 保留原来的小分块执行路径，方便对照或在显存紧张时使用
python train_ConstrainedMKP.py --performance-profile reference
# 只关闭跨 epoch 的整批状态缓存，保留大分块和单次前向内的特征复用
python train_ConstrainedMKP.py --performance-profile memory24 --feature-cache-gib 0
# 从已有 v5 checkpoint 续训；updates 是累计更新次数，原模型参数需保持一致
python train_ConstrainedMKP.py --resources 5 --resume checkpoints/combination_v5/mkp_M5_last.pt --updates 4000
```

**本轮速度优化不更改 checkpoint schema 或 model_config，不需要重训现有 v5 模型。** 执行设置单独保存到 `runtime_M5.json` / `runtime_M10.json` 和 checkpoint 的 `training_runtime`；`--resume` 可更改缓存预算或执行 profile。已启动的 Python 进程需要重启并续训才会使用新代码。扩大矩阵分块可能造成浮点舍入差异，不能承诺长程采样/训练逐位相同。独立 Benchmark 默认仍按 checkpoint 原模型配置推理。

2026-09-21 本机短时测量，PyTorch 2.11.0+cu128，B=128、N=100、M=10，默认网络；预热 2 次、计时 6 次，512 个合成状态，仅执行前向/反向：

| 执行方式               | 每 minibatch 前向/反向 | 计入按 3 epochs 分摊的缓存构建时间 |
| ---------------------- | ---------------------: | ---------------------------------: |
| reference              |                202.7ms |                            202.7ms |
| 大分块、单次前向内复用 |                 75.9ms |                             75.9ms |
| 大分块 + 跨 epoch 缓存 |                 62.7ms |                             69.8ms |

该计算环节计入缓存构建后约 **2.9×**；512 状态缓存及状态副本约 0.157GiB，本进程 allocator 的 reserved 峰值从 2.17GiB 增至 2.69GiB。logit 最大差异约 `5.72e-6`。这是一次局部计时，GPU 负载/频率会影响结果；**没有执行优化器更新、完整训练或 Chu/gap 评估，不能把该倍数视为完整训练提速。** 原始结果在 `outputs/performance_probe_5090.json`，复测建议在 GPU 空闲时运行：

```powershell
python profile_training_speed.py --output outputs/performance_probe.json
```

实际训练可查看 `train.log` 的 `SPEED` 行，及 CSV 中的 `rollout_seconds`、`ranking_label_seconds`、`ppo_seconds`（包括缓存构建，不含标签生成）、`feature_cache_seconds`、`feature_cache_gib`、`gpu_peak_allocated_gib` / `gpu_peak_reserved_gib`；后两者是本进程训练阶段的 PyTorch 显存峰值，不包含其他进程，也不包含随后独立验证阶段。

## 组合 θ 修改：把组合信息传到解码器

本分支上一版实际是 **M+6** 个 θ 条目，而非其他项目旧接口的 M+3。仅增强 attention 后，资源之间的关系仍需压缩进单资源价格和少量全局权重。本轮新增两组可解释的状态级控制参数：

| 新 θ                                                               | 固定基函数                                                           | 用途                                           |
| ------------------------------------------------------------------ | -------------------------------------------------------------------- | ---------------------------------------------- |
| `resource_interactions[B,M,M]`，对称、非负，每个无序资源对只计一次 | 单资源平方 `z_ir²` 和跨资源乘积 `z_ir*z_is`                          | 表达同一候选同时消耗多个资源时的额外成本       |
| `companion_weights[B,K]`，非负，默认 K=3                           | 两物品可行性 × 组合收益质量 × 多尺度余量匹配度，再对搭配物品取最大值 | 让单看普通、但存在高价值互补搭配的候选获得加分 |

新 θ 有 `M+6+M(M+1)/2+K` 个非重复条目，默认 M=5 时为 29，M=10 时为 74；这是输出条目数，并未扣除价格归一化造成的冗余。张量为方便运算保存完整对称矩阵。**θ 维度不随物品数量 N 增长，网络仍不输出逐物品评分或 Q 值。** 资源对头用交换对称的资源表示生成系数，保持资源重排等变；全局头控制搭配尺度的权重。

评分扩展为 `原有 score - 二阶资源成本 + 两物品搭配加分`，详细公式见下文。两个新增系数组各有独立的总幅度上界，默认均为 2，初始总幅度均为 0.02，避免新项在初始化时覆盖原有评分。将两个上界都设为 0，可以精确恢复上一版评分，并跳过新增基函数计算。组合 θ 这一步保留 PPO、奖励、独立 Critic 和冻结策略排序损失形式；梯度通过固定基函数系数回到高层控制器。最新的排序标签来源已改为本文开头的 θ 候选机制。

微型表达能力测试：容量 `[10,10]`，A=`12/[6,6]`，B=`8/[10,0]`，C=`8/[0,10]`。原密度偏好 A，随后 B/C 都不可行；合法的搭配系数可使解码器先取 B、再取 C，合计 16。另一个测试验证交叉成本具有非零混合差分，并能改变相同总消耗候选的排序。**这些是人工设置合法 θ 的接口测试，不是训练所得策略或 gap 改善结果。**

### 运行与独立消融

默认训练已启用两类基函数。建议用相同种子、预算及独立目录做 2×2 消融：

| 对照           | 训练参数                                              |
| -------------- | ----------------------------------------------------- |
| 上一版评分     | `--interaction-weight-max 0 --companion-weight-max 0` |
| 仅二阶资源成本 | `--companion-weight-max 0`                            |
| 仅两物品搭配   | `--interaction-weight-max 0`                          |
| 两类同时启用   | 默认                                                  |

这些开关改变评分表达，与下方 Actor 感知开关相互独立。其他参数为 `--companion-scales 0.5 2 8`、`--combination-initial-weight 0.02`、`--combination-chunk-size 64`。更改尺度会改变基函数及 checkpoint 配置，续训必须使用原设置。

```powershell
# 自行运行：完整 v5
python train_ConstrainedMKP.py --checkpoint-dir checkpoints/combination_v5
python Benchmark_zeroShot.py --checkpoint-dir checkpoints/combination_v5
# 自行运行：同一 Actor、上一版评分的对照，需要单独重新训练
python train_ConstrainedMKP.py --interaction-weight-max 0 --companion-weight-max 0 --checkpoint-dir checkpoints/ablation_v4_score
python Benchmark_zeroShot.py --checkpoint-dir checkpoints/ablation_v4_score
```

训练 `metrics.csv` / `train.log` 和 Benchmark `instances.csv` 增加 `interaction_weight_sum`、`companion_weight_sum`、`interaction_penalty_mean`、`companion_bonus_mean`、`combination_abs_mean`、`combination_to_base_margin_ratio`。最后一个指标是同状态下“新增修正的候选间极差 / 原有评分的候选间极差”，分母下限 1e-3，单候选/终止状态为 0；它不受公共分数偏移或采样温度影响。训练按有效状态步平均；Benchmark 先在每一步对候选统计，再对该轨迹的有效步平均。这些指标帮助检查尺度，不能替代配对终局收益和 gap。

## 保留的 v4 改动：让 θ 看见关键候选的差别

前一轮根据用户提供的 oracle_ceiling 实验，优先增强 Actor 的选后感知、关系注意力和差异聚合。以下网络结构在 v5 中继续使用，并将其资源表示和全局表示送入新增组合控制头；本轮没有重新运行 oracle 实验。

| 改动             | 网络获得的信息                                                                                                      | 代码位置                                                                |
| ---------------- | ------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| 选后物品特征     | 选择 i 后阻断的可行收益质量、平滑兼容收益质量、其他物品仍可行的比例、平滑余量损失，共 4 维                          | `mkp_attention.py:candidate_relations`，`mkp_Model.py:BipartiteEncoder` |
| 选后资源边特征   | 每个资源分别造成的阻断收益质量和余量损失，共 2 维；保留应调整哪个资源价格的信息                                     | `candidate_relations`，`BipartiteBlock` 的双向交叉注意力                |
| 物品关系注意力   | 利润差、负载差、共同装入余量、是否冲突等关系同时进入 attention bias 和 Value；即使节点 Value 接近，也能传递不同消息 | `mkp_attention.py:PairRelationAttention`                                |
| 相似候选差异聚合 | 将候选与相近物品比较，保留有方向的 embedding 差和选后特征差；通过差异强度及可学习查询汇聚关键候选                   | `mkp_attention.py:CriticalContrastPool`                                 |
| 可行候选聚合     | Actor 的物品消息、物品均值及竞争查询仅聚合当前可行物品；全局历史摘要仍保留                                          | `mkp_Model.py:EdgeCrossAttention/BipartiteEncoder`                      |

对当前可行且不同的 i、j，先用 float64 计算 `r_ijm = (remaining_m - w_im) - w_jm`。任一维 `r_ijm < 0` 表示选择 i 后 j 无法再装入。所有收益质量都除以选择前的可行总收益；排除自身、已选及当前不可行物品。资源边上的阻断归因逐资源计算，同一个物品可以在多个资源上冲突，因此各资源归因不应相加解释为总阻断收益。

物品间 7 维关系为：`(p_i-p_j)/max(p_i+p_j,eps)`、平均 `|w_i-w_j|/remaining`、最小相对共同余量、是否阻断、平滑兼容度、相对选择前的平滑余量损失、j 的可行收益质量。实现对零分母作下界保护，对极端相对负载作截断；容量是否超限始终使用原始 float64 余量比较。差异池化按利润差与负载差为其他候选分配相似性权重，不依赖物品序号或硬 top-k；再用候选自身与相似候选的差异构成非线性消息，避免正负差直接求平均后抵消。

例如 A=`(p=10.05,w=[5.2,5])`、B=`(p=10,w=[5,5.2])`、C=`(p=5,w=[5,1])`，剩余容量为 `[10,10]`：A、B 即时属性相近，A 会阻断 C，B 保留 C。新 Actor 可以直接看到这一区别及其资源归因，再通过 θ 改变既有公式的相对评分。**这些是当前状态的静态兼容信息，不是终局收益预测，也不能保证识别所有多步组合效应。** 终局方向仍由 PPO 和已有的冻结策略续行排序损失学习；排序标签仍是当前策略下的收益估计。

数据流：`当前状态 → 选后关系特征 → Actor 二分图与物品关系注意力 → 资源/全局表示 → 扩展 θ → 固定评分基函数`。注意力权重仅用于内部表示。原有 θ 头及新增控制头的最后一层权重从零初始化：策略/排序分支在初始化时首先更新输出层，之后梯度传入上游表示；可行性辅助分支仍可从首步向 Actor 编码器传递梯度。梯度测试使用非零输出层权重验证完整通路，没有执行优化器更新。

### Actor 感知消融与计算开销

四个开关默认开启，可分别关闭并组合做因子消融，使用相同种子、训练预算、独立 checkpoint 目录，最终比较配对收益/gap：

| 消融                                       | 命令参数                           |
| ------------------------------------------ | ---------------------------------- |
| 不向物品输入/资源边注入选后特征            | `--no-actor-postdecision-features` |
| 恢复普通物品自注意力                       | `--no-actor-pair-attention`        |
| 关闭相似候选差异池化                       | `--no-actor-contrast-pooling`      |
| 恢复普通注意力/均值/竞争查询中的全物品聚合 | `--no-actor-feasible-attention`    |

前 3 个开关控制不同使用位置：关闭特征注入时，关系注意力和差异池化仍可使用关系特征；关系模块始终排除自身和无效候选。关闭第 4 项只恢复普通聚合中的全物品读取，不会把无效候选引入关系模块。四项全部关闭恢复 v3 的 Actor 编码结构，v5 组合输出头仍存在；若还要恢复原评分，需同时关闭两类新增系数上界。所有消融均使用 v5 checkpoint 格式。

关系特征计算额外需要 `O(BN²M)`，物品关系注意力约 `O(BN²D)`。`--actor-relation-chunk-size 32` 控制特征计算和注意力的查询分块，避免构造 `[B,N,N,D]` 的边 Value 张量；仍会缓存 `[B,N,N,7]` 关系及训练激活，**分块不意味着总内存对 N 线性增长**。N=250/500 的实际速度、显存及完整续行标注开销需正式运行测量。`--actor-relation-bandwidth 0.05` 控制余量平滑，`--actor-similarity-bandwidth 0.1` 控制相似候选权重。三个关系功能均关闭时跳过新增关系计算。

## 运行

在 PowerShell 中进入本项目，依次执行：

```powershell
& 'D:\Zlb08\.envs\RL_MKPtest\Scripts\python.exe' train_ConstrainedMKP.py
& 'D:\Zlb08\.envs\RL_MKPtest\Scripts\python.exe' Benchmark_zeroShot.py
```

默认分别训练 M=5、M=10 的模型，训练规模 N=100，共享该 M 下所有物品位置的网络参数；随后测试两个维度下 N=100/250/500 的全部 180 个 Chu 实例。原始文件还包含 M=30 的 90 个实例，默认不测试。

```powershell
# 自行运行：只训练一个资源维度
& 'D:\Zlb08\.envs\RL_MKPtest\Scripts\python.exe' train_ConstrainedMKP.py --resources 5 --updates 2000
# 自行运行：从 last checkpoint 继续，updates 是累计更新次数目标
& 'D:\Zlb08\.envs\RL_MKPtest\Scripts\python.exe' train_ConstrainedMKP.py --resources 5 --resume checkpoints/combination_v5/mkp_M5_last.pt --updates 4000
# 可选：另报 128 条随机轨迹中的最佳解，不替代单轨迹 greedy 结果
& 'D:\Zlb08\.envs\RL_MKPtest\Scripts\python.exe' Benchmark_zeroShot.py --sample-rollouts 128
```

其余参数集中在 `mkp_config.py`，包括模型、生成器、PPO、日志、checkpoint、benchmark。`--help` 只显示参数，不运行实验。运行目录不会影响默认数据和产物路径。

## 模块与输出

| 文件                        | 职责                                                      |
| --------------------------- | --------------------------------------------------------- |
| `mkp_config.py`             | 所有可调整默认参数及校验                                  |
| `mkp_data.py`               | 调用现有 `read_mat.py`，统一数组方向；独立随机训练数据    |
| `mkp_env.py`                | float64 容量与收益记账，逐步选择，终止与可行候选集合      |
| `mkp_Model.py`              | 二分图多头 Transformer、高层策略、固定解码器、独立 Critic |
| `mkp_attention.py`          | Actor 选后兼容特征、物品关系注意力、相似候选差异池化      |
| `mkp_basis.py`              | 组合 θ 输出头、固定二阶/搭配基函数、评分贡献统计          |
| `mkp_performance.py`        | 训练执行 profile、无梯度固定特征缓存、显存预算回退        |
| `profile_training_speed.py` | 合成状态前向/反向短时计时，不执行优化器更新               |
| `mkp_Trainer.py`            | 完整 episode 采样、GAE、PPO、合成验证、日志、checkpoint   |
| `mkp_ranking.py`            | 关键候选筛选、固定策略成对续行、收益标签与排序损失        |
| `mkp_theta_candidates.py`   | 训练期合法 θ 提议、原 decoder 动作诱导及去重              |
| `verify_ranking_audit.py`   | 重放冻结权重/状态/随机数，核验候选动作、Q 与标签          |
| `train_ConstrainedMKP.py`   | 简洁训练入口                                              |
| `Benchmark_zeroShot.py`     | 冻结推理、独立解校验、CSV 与物品序列                      |
| `mkp_utils.py`              | 复现设置、checkpoint 合约、共享构造过程                   |
| `tests/test_contracts.py`   | 小张量接口与梯度校验，不执行训练或 benchmark              |

- `runs/<时间戳>/train.log`：每次更新的日志。
- `runs/<时间戳>/metrics.csv`：收益、归一化 return、利用率、PPO 各项损失、归一化 entropy、KL、梯度范数、Critic explained variance、合成验证表现。
- `runs/<时间戳>/config.json`：配置与运行环境。
- `checkpoints/combination_v5/mkp_M5_best.pt`、同目录 `mkp_M10_best.pt`：按固定合成验证集的 greedy 归一化收益选择。
- `checkpoints/combination_v5/mkp_M5_last.pt`、同目录 `mkp_M10_last.pt`：最近一次保存的模型，包含优化器和 RNG 状态；默认每 100 次更新、产生新 best 及训练结束时保存。
- `results/<时间戳>/instances.csv`：逐实例、逐方法报告。
- `results/<时间戳>/summary.csv`：每种方法按 M/N 分组及总体报告。
- `results/<时间戳>/solutions.jsonl`：零基索引物品选择顺序，可独立复核。
- `results/<时间戳>/manifest.json`：数据及 checkpoint SHA256、预算、设备、配置、运行状态。

重新训练会更新同一 M 的标准 checkpoint 文件名；日志目录保留每次配置。继续训练时保留原模型、数据分布、训练 N 和固定验证设置。

## 收敛诊断与训练曲线

直接按上面的两条命令运行训练和 Benchmark 即可。`TrainConfig.updates=2000` 是每个 M 的默认训练预算；诊断不提前停止训练。每个 M 训练结束后自动生成：

- `runs/<时间戳>/convergence_M5.json`、`convergence_M10.json`：最终收敛判定及阈值。
- `runs/<时间戳>/plots/training_M5.png`、`training_M10.png`，以及同名 SVG：12 个子图，包括固定验证收益、训练 return、归一化 policy entropy、policy/value/辅助损失、approx KL、clip fraction、梯度范数、explained variance、容量利用率、KL 提前停止频率。浅色为原始数据，深色为最近 100 updates 的均值；验证曲线不平滑。
- `runs/<时间戳>/diagnostics_history_M*.json`：绘图历史，续训时包含新版本 checkpoint 中继承的记录。
- `results/<时间戳>/training_convergence.json`：Benchmark 完成后，将每个 M 的实际被测 checkpoint update、SHA256、该 checkpoint 的平均 gap 与来源训练的收敛报告关联起来。**该 gap 不代表平台起点模型的 gap**。

### 收敛点的含义

判据使用固定、独立的**合成验证集 greedy 归一化收益**，不使用训练 loss、entropy 或 Chu 测试集选择收敛点。Entropy/KL 等曲线用于解释策略行为，性能进入平台并不要求策略分布完全静止。

默认从最近一次验证向前寻找连续区间，要求每一次验证收益都不低于历史最佳收益的 `99.95%`，即相对差距不超过 `0.05%`。例如最佳 return 为 `0.5`，容许的绝对差为 `0.00025`。该区间须跨越至少 1000 updates、包含至少 20 次验证，且确认时累计 update 不小于 1000。默认每 25 updates 验证一次，起点精度受验证间隔限制。

| TrainConfig 参数               | 默认值 | 含义                                                    |
| ------------------------------ | ------ | ------------------------------------------------------- |
| `convergence_tolerance_pct`    | `0.05` | 相对最佳验证收益的百分比容差，不是 Benchmark gap 百分点 |
| `convergence_patience_updates` | `1000` | 连续平台至少跨越的 update 数                            |
| `convergence_min_evaluations`  | `20`   | 平台至少包含的有效验证次数                              |
| `convergence_min_updates`      | `1000` | 最早允许确认平台的累计 update                           |
| `plot_smoothing_updates`       | `100`  | RL 曲线平滑窗口，仅影响显示                             |

`train.log` 中搜索 `CONVERGENCE_FINAL`，每个 M 分别报告：

- `plateau_start_update`：最终仍成立的平台起点；图中用**红色实线**标记。
- `confirmation_update`：观察长度足够、满足上述条件的确认点；图中用**橙色虚线**标记。以后若考虑早停预算，应参考确认点，因为在平台起点尚无后续证据。
- `best_update`：验证收益最高的 update，与收敛点分开；图中用紫色星号标记，Benchmark 默认仍评估 `best.pt`。
- `status=converged`：截至本次训练结束满足性能平台判据；`not_converged` / `insufficient_history` 表示尚未确认，不会强制画一个收敛点。

训练中 `CONVERGENCE_CANDIDATE` 只表示临时判断；后期提升或退化会使候选失效，最终以 `CONVERGENCE_FINAL` 为准。这是有限训练预算内、阈值明确的性能平台诊断，不能证明继续训练永远不会进步，也不能证明 Chu gap 同时收敛。训练和 checkpoint 选择仍只使用合成数据。

新 checkpoint 保存诊断历史，因此续训会延续判断；旧 checkpoint 没有历史时，会在日志提示仅从续训段积累证据，并保留旧 best 收益作为参照，避免把退化后的稳定误判成收敛。

只重绘已有日志（不训练、不评估）：

```powershell
python mkp_diagnostics.py runs/<时间戳>
```

绘图依赖 `matplotlib`；训练开始前会检查能否加载，避免长时间训练结束后才发现缺少绘图依赖。现有合约测试与新增诊断测试可用 `python -m unittest discover -s tests -v` 执行；新增流程测试使用模拟 rollout 和 PPO 指标，不执行真实优化或 Chu 推理。

## 当前代码中的二分图可以概括为：
- 左侧节点：N 个物品节点。基础特征 8 维，默认 Actor 额外接收 4 维选后兼容特征，共 12 维；Critic 仍用基础 8 维。
- 右侧节点：M 个资源节点。每个节点包含剩余容量比例、资源需求、已选择比例、可行物品比例等 6 维特征。
- 边：每个物品与每个资源相连，即形成完整的物品—资源关系。边特征为：
  1. 初始消耗；
  2. 当前消耗；
  3. 当前相对容量的超额程度。
基础边为 `edge_features[B,N,M,3]`，默认 Actor 拼接 2 维选后资源特征成为 5 维；边同时进入 attention bias 和 Value 消息。Actor 还读取 `[B,N,N,7]` 的物品对关系。Critic 保留原有独立编码器及 3 维资源边。
Transformer 的作用是：
1. 分别将物品节点和资源节点映射到统一的 embedding 空间；
2. 通过物品→资源、资源→物品的交叉多头注意力，让两类节点交换信息，并利用边特征体现资源消耗关系；
3. 通过各自的 self-attention，让物品之间比较利润与负载，资源之间比较紧张程度和瓶颈；
4. 对可行物品及资源池化，再汇聚竞争查询与相似候选差异，形成全局状态表示；
5. 高层策略头根据该表示输出低维策略参数 θ，包括：
   - 资源价格 resource_prices
   - 利润权重 profit_exponent
   - 稀缺性权重 scarcity_exponent
   - 瓶颈惩罚 bottleneck_weight
   - 温度 temperature
   - 后续可行机会损失权重 opportunity_weight
   - 后续资源余量损失权重 reserve_weight
   - 资源交互系数 resource_interactions
   - 两物品搭配基函数权重 companion_weights
随后，固定的弱解码器使用 θ 对当前可行物品打分并选择物品。因此，Transformer 不直接输出每个物品的动作概率，而是根据全局二分图状态调整解码器的策略参数。

## 模型如何感知状态与约束

物品节点包含归一化收益、初始资源占用、已选状态、当前可行性、资源超额程度与当前容量下的占用比例。资源节点包含剩余容量比例、已用比例、未选物品需求和可行物品需求。二分图边包含每个物品对每个资源的初始占用、动态占用与超额量。

每个模块包含：物品→资源与资源→物品的带边偏置多头交叉注意力、物品间多头自注意力、资源间多头自注意力、前馈网络。默认 8 头、128 维、2 层。节点没有位置编码，支持不同 N/M 的输入；默认每个 M 单独训练，避免未经验证地宣称跨资源维度泛化。

物品关系注意力可学习收益、负载与选后兼容关系，资源自注意力可学习瓶颈资源之间的关系，但**不预先宣称每个注意力头已经学到了某种语义**。Actor 在可行物品均值汇聚后使用 4 个可学习查询提取物品/资源竞争摘要，再增加相似候选差异池化。默认已选和不可行物品不作为 Actor 的物品消息来源；累计收益、已选比例等全局摘要及原有资源需求特征仍保留历史与背景信息。Critic 的全物品注意力保持原样。

资源价格使用 `MLP(resource, global, resource * global)` 后再归一化。原来的线性拼接中，全局项对所有资源相同，会在 softmax 中抵消；现在全局状态能通过非线性交互改变资源之间的相对价格。

另有当前可行性的辅助预测损失，帮助编码器学习约束表示；该预测头的输出不参与物品评分。Critic 使用独立编码器，不向弱解码器提供预测收益。

### 关于 mask

默认 Actor 在读取物品时使用可行性 attention mask，物品关系还排除自身；空候选行的注意力消息显式归零，避免全屏蔽 softmax 导致 NaN。资源节点始终可以参与竞争查询。Critic 保持无 attention mask。解码器仍根据环境的真实容量生成可行候选索引，使用分段 softmax 构成 categorical 分布，不构造包含不可行动作的 dense logits；终止行使用动作 `-1`。注意力掩码只控制信息聚合，最终动作合法性仍由环境检查。

这里仍有布尔型“可行集合”和“已选状态”；它们是 MKP 的物理约束检查。完全取消可行性检查无法保证输出合法解。约束状态显式进入模型，合法性由环境再次检查，两者同时保留。评分公式内部的 `-inf` 仅用于表示零资源消耗的对数加法项，不是屏蔽物品 logits。

## 高层策略与固定弱解码器

高层保留 M 个正资源价格及 6 个标量，再增加对称资源交互系数和 K 个搭配系数；共 `M+6+M(M+1)/2+K` 个非重复条目，维度不随 N 增长：

- `resource_prices`：均值为 1，避免价格尺度冗余。
- `profit_exponent`：收益竞争强度 α。
- `scarcity_exponent`：资源稀缺性强度 β。
- `bottleneck_weight`：单一瓶颈占用惩罚 κ。
- `temperature`：随机策略温度 τ；只影响采样概率，在同一步 greedy 排序中不改变 argmax。
- `opportunity_weight`：选后永久失去的可行收益机会的惩罚强度 λ。
- `reserve_weight`：选后资源余量下降的平滑惩罚强度 ρ。
- `resource_interactions`：对称矩阵 A，非负，上三角总和不超过 `interaction_weight_max`；对角线控制平方成本，非对角线控制跨资源联动成本。
- `companion_weights`：长度 K 的非负向量 b，总和不超过 `companion_weight_max`，控制不同剩余容量尺度下的搭配加分。

设 p_i 为收益、w_ij 为占用、C_j 为初始容量、c_j 为当前剩余容量：

```text
q_ij    = N * w_ij / C_j
s_j     = C_j / max(c_j, eps)
cost_i  = mean_j(price_j * q_ij * s_j ** beta)
score_i = alpha * log(max(p_i / mean(p), eps))
          - log(cost_i)
          - kappa * max_j(w_ij / max(c_j, eps))
          - opportunity_weight * blocked_mass_i
          - reserve_weight * margin_loss_i
          - sum_{r<=s}(A_rs * z_ir * z_is)
          + sum_k(b_k * F_ik)
pi_i    = softmax(score_i / temperature)，只在可行候选集合中归一化
```

零总资源消耗的物品用 eps 成本处理；数值实现使用 log-sum-exp。默认固定对照为 price=1、α=1、β=1、κ=0.1、τ=0.2、λ=ρ=0、A=b=0，保留原始密度基线。网络初始化为同样的密度参数，λ=ρ=0.1，A 的上三角系数总和与 b 的系数总和默认各为 0.02。

弱解码器没有可训练参数、物品指针网络、局部搜索、修复、beam search 或优化求解器。设每行可行候选数为 F：原始密度评分约 `O(FM)`，原机会项 `O(FNM)`，新增交互项 `O(FM²)`，搭配项 `O(FN(M+K))`。原机会项按 128 个候选分块，新搭配项按 `combination_chunk_size=64` 分块，临时空间约 `O(64N(M+K))`；没有枚举候选 i 之后所有 j/k 三元组合。`fixed` 对照会跳过所有未使用的机会及组合计算。无参数并不等于无推理开销。

### 新增二阶与搭配基函数

令 μ_r 为当前可行物品在资源 r 上的平均消耗，p̄ 为其平均收益。新增二阶项使用：

```text
z_ir = w_ir / max(w_ir + μ_r, eps)                 # [0,1]
interaction_cost_i = sum_{r<=s}(A_rs * z_ir*z_is)
```

使用当前平均需求为单位，而非直接平方 `w/C`，避免大 N 时新增项仅因容量尺度变化按 `1/N²` 缩小；对所有资源对的系数总幅度作约束，避免随 M 增长累加放大。归一化只能控制尺度，不保证跨规模效果。A 在同一状态内对所有候选共享；非零 A_rs 允许一个维度的边际成本随另一维度的消耗而变，不再仅靠独立资源价格表达。这是 z 空间的二次基函数，不声称 A 是半正定矩阵或完整成本函数具有全局凸性。

对可行且不同的物品 i、j，固定搭配基函数为：

```text
r_ijr = (c_r - w_ir) - w_jr                       # float64
pair_valid_ij = all_r(r_ijr >= 0)
gap_ij = mean_{r: μ_r>0}(max(r_ijr,0) / max(μ_r,eps))
quality_ij = (p_i+p_j) / max(p_i+p_j+2*p̄, eps)   # [0,1]
F_ik = max_{j in feasible, j!=i}(
    pair_valid_ij * quality_ij * exp(-gap_ij / scale_k))
scale = [0.5, 2, 8]                              # 单位是平均需求个数
```

无可行搭配时 F=0；没有活跃需求资源时 gap=0。max 仅汇聚一个固定特征值，不返回或执行第二个动作；实际下一步仍重新生成 θ、选择单个物品。它既看两物品的合计收益，也看装入后留下的容量，避免只按“是否能搭配”区分。使用 max 而非对 N 个物品求均值，可以保留少量高质量搭配的信号。每个 F∈[0,1]，故新增惩罚与加分分别受各自系数总幅度上界约束。

上述设计显式表达当前候选与一个搭配物品的关系，以及资源之间的二阶成本；**不精确计算选 i 后任意其他两物品 j/k 的联合可行性或更长组合的价值，也不保证任意最优动作都可表达**。这些更长后果仍依赖状态条件化的 θ 与终局训练信号。判断改动是否有效，应看同状态排序、最终 objective/gap，而非权重大小或容量填充率本身。

### 机会损失如何区分相似物品

设 F 为当前可行集合，Z 为 F 中所有物品的总收益。对候选 i 和其余可行物品 k，计算：

```text
d_k    = min_j((c_j - w_kj) / max(c_j, eps))
d_ik   = min_j((c_j - w_ij - w_kj) / max(c_j, eps))
blocked_mass_i = sum_{k in F, k != i} p_k * 1[存在 j: c_j - w_ij - w_kj < 0] / Z
margin_loss_i  = sum_{k in F, k != i} p_k * (sigmoid(d_k / h) - sigmoid(d_ik / h)) / Z
h = opportunity_bandwidth = 0.05
```

两项都排除 i 自身以及当前已选/不可行物品。容量边界用 float64 判断；平滑项可以在物品尚未不可行时区分接近临界容量的选择。λ、ρ 由全局状态产生，其他算术固定。

微型合约样例：容量 `(10,10)`，A 的利润/占用为 `10.05/(5.2,5.0)`，B 为 `10/(5.0,5.2)`，C 为 `5/(5.0,1.0)`。A、B 总耗用相同，A 的密度略高，但 A 会同时挤掉 B、C；B 之后仍可放 C。测试验证机会项能把选择从 A 翻到 B，强制续行的总收益分别为 10.05 和 15。**这是可控样例上的表达能力检查，不是训练收益或 Chu gap 结果。**

这些机会项只描述剩余候选集合和资源余量，不能把各物品的可行收益之和解释为可同时获得的终局收益。真正的长程方向来自下面的完整续行监督；有限维 θ 仍不保证所有状态都能表达最佳选择。

### 同状态的长程排序训练

现默认使用本文开头的 θ 候选机制。旧机制保留为 `--ranking-mode score_pairs` 对照：每次 PPO 更新之前，从本轮合成 rollout 抽取最多 64 个状态，以实际 greedy 候选为锚，在最高分的 8 个可行候选中，按利润差、逐资源占用差和原始 score 差选择相近对手。最多保留 16 个状态，约 3/4 优先取难区分的状态，其余随机选取。

对每个状态复制两个分支，分别先选 i、j；后续都使用**本次更新前、参数固定的策略**构造到终点。每对执行 1 组 greedy 续行和 4 组随机续行；随机分支在相同步骤对相同原始物品共享 Gumbel 噪声。标签为归一化终局收益差 Δ，默认各占 50% greedy 和随机均值。过滤 `|Δ| <= max(0.001, SE)` 的小差异/噪声样本，也过滤 greedy 与随机均值明确冲突的方向。该标签是当前策略下的续行估计，不是最优 Q 值或 oracle。

```text
target = sigmoid(delta / ranking_return_scale)
margin = (score_i - score_j) / ranking_score_scale
ranking_loss = BCEWithLogits(margin, target)
total_loss = 原 PPO/Value/可行性/Entropy 损失 + 0.1 * ranking_loss
```

排序使用温度缩放**之前**的 score，模型不能只调低采样温度来满足排序监督。标签及续行均无梯度；梯度由固定 score 经 θ 回到 Actor，不进入 Critic。反事实分支不进入 PPO buffer，不改变 GAE/Value 的目标。每个 PPO minibatch 另外计算整个标签状态集合的全候选 KL，超过 `target_kl` 即停止后续更新，避免只检查已采样动作而遗漏其他候选的概率漂移。标签使用独立 RNG，不消耗主 rollout 随机流。

`metrics.csv` / `train.log` 记录标签提议/接受数、平均终局收益差、标准误、标签耗时、排序损失、排序方向正确率和全候选 KL。`ranking_accuracy` 是当轮被接受训练对上的方向正确率；没有接受对时记 0，应结合 `ranking_pairs` 阅读，不能当作独立验证精度。Benchmark 记录原有 θ 均值及新增组合指标，checkpoint/manifest 保存 v5 控制器、解码器标识与完整模型配置。推理不执行反事实续行。

### 运行与独立消融

默认命令已开启 θ 候选排序。以下是保留的原 v3 机制开关，可与前述 v4 控制器开关组合。建议每组使用相同种子、训练预算及独立 checkpoint 目录，最终比较同一批实例的配对收益/gap：

| 关闭的部分     | 训练参数                                            |
| -------------- | --------------------------------------------------- |
| 长程排序监督   | `--ranking-coefficient 0`                           |
| 选后机会评分   | `--opportunity-weight-max 0 --reserve-weight-max 0` |
| Actor 边 Value | `--no-actor-edge-values`                            |
| 竞争摘要查询   | `--opportunity-tokens 0`                            |

可以组合开关做因子消融；非线性资源价格头在这些组中保持一致。关闭所有开关不等于恢复 v1 网络结构。完整续行标签会增加训练时间，可用 `--ranking-states 8 --ranking-repeats 2` 降低标注预算；样本量和方差也随之改变，应记录配置。

```powershell
# 自行运行：仅关闭长程排序，保留网络与新评分
python train_ConstrainedMKP.py --ranking-coefficient 0 --checkpoint-dir checkpoints/ablation_no_ranking
python Benchmark_zeroShot.py --checkpoint-dir checkpoints/ablation_no_ranking
```

## 强化学习与零样本边界

训练采用通过固定评分公式诱导的 categorical **物品策略**的 PPO：采样的是物品动作，PPO ratio 使用实际物品的 log probability；梯度沿“概率→固定评分→θ→高层 Transformer”传播。θ 本身不是额外采样的高斯动作，不能用另一套 θ 分布的概率计算 PPO ratio。

单步 reward = 被选物品收益 / 该实例全部物品收益之和。默认 gamma=1，所以累计 reward 与原始总收益同序；容量利用率只记录，不作为额外奖励。没有可行物品即终止；GAE 不跨终止和 batch padding bootstrap。PPO 使用 clipped surrogate、value clipping、梯度裁剪、KL 提前停止和可行性辅助损失。归一化 entropy 在候选数不同时便于比较，单候选和终止状态 entropy 为零。

训练实例由随机生成器产生：整数资源占用，三种容量比例，以及以资源平均占用相关收益为主、混合独立收益的分布。这是借鉴 Chu 风格的训练分布，**不是原始数据生成器的精确复现**。固定验证集使用独立随机种子，且只使用训练 N 的合成实例。

训练与 checkpoint 选择不读取 Chu。测试程序只读取冻结 checkpoint；N=100 是未见实例测试，N=250/500 在默认配置下同时是跨规模 zero-shot。CSV 中 `size_zero_shot` 根据 checkpoint 中记录的训练 N 自动标记。如果自己在训练配置加入 N=250，则该规模不再标记为跨规模测试。

### 对照与指标

默认对每个实例分别报告：

1. `fixed`：固定 θ，动态环境仍按当前容量计算评分。
2. `learned_static`：仅在初始状态产生一次 θ，整个构造过程复用。
3. `learned_dynamic`：每步重算 θ，单轨迹 greedy，作为主要方法结果。
4. 可选 `learned_dynamic_sampled`：独立报告 K 条随机轨迹的最好结果及完整时间预算，不与单轨迹合并。

主要比较 dynamic 对 fixed 的提升，及 dynamic 对 static 的提升，后者检验动态状态感知的价值。`gap_pct = 100 * (best - objective) / best`，其中 best 来自 MAT 文件；仅标记为 **best-known reference**，不声称已经验证其全局最优性。保留负 gap 并标记 `reference_exceeded`，避免截断掩盖数据问题或更好解。

报告包括平均/中位/最差 gap、≤0.1% 的实例比例、原始收益、逐资源利用率、选中数量、可行性、独立重算容量违约量、时间、采样预算和策略统计。只看平均 gap≤0.1% 并不意味着每个实例达标，应同时查看 worst gap 和 target hit rate。时间经过 CUDA 同步，不含 checkpoint 加载，含数据搬运和整条解构造；不做预热，首实例可能包含初始化开销。

新增数据集：若遵循现有 `read_mat` 字段布局，可使用 `--dataset`；其他布局应新增读取适配器，统一转换为 `MKPInstance`。不猜测未知 MAT 格式，也不把缺失 best 当成零 gap。

## 思想来源与原创性边界

- [Gasse 等，2019，Exact Combinatorial Optimization with Graph Convolutional Neural Networks](https://arxiv.org/abs/1906.01629)：借鉴变量—约束二分图表示，理由是 MKP 物品与资源的消耗关系自然构成二分图。原论文研究分支定界变量选择，这里没有使用其求解器或训练任务。
- [Kool 等，2019，Attention, Learn to Solve Routing Problems!](https://arxiv.org/abs/1803.08475)：借鉴多头注意力处理可变集合与构造式优化的思路；这里的 Transformer 输出低维资源策略，物品评分保持固定。
- [Schulman 等，2017，Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)：使用 clipped PPO 提高策略更新稳定性；这是已有算法，不作为原创贡献。

本实现的待验证设计是：动态资源占用边偏置 + 双向物品/资源竞争表示 + MKP 高层资源策略 + 无参数弱解码器的组合。组合是否具有论文层面的新颖性仍需进一步文献比较；消融对照和真实 gap 结果是论证贡献的必要证据。此实现采用直接 PyTorch 环境与 PPO，便于审查状态/动作/奖励合约，无需依赖 RL4CO 的特定版本接口。

## 交付验证

测试检查小张量上的候选概率、固定公式、约束拒绝、终止/GAE、排列等变性、高层梯度、checkpoint 往返及独立解校验。`tests/test_opportunity_ranking.py` 检查关键选择翻转、微型实例的固定策略终局收益、共享随机数、排序梯度、温度不变性和全候选 KL；PPO 接线测试以 mock 替代优化器 step。

`tests/test_controller_attention.py` 新增 12 项测试：A/B 选后特征及资源归因、float64 容量边界、排除自身/无效候选、分块和排列一致性、相同节点 Value 下的关系区分、无效 token 的聚合隔离、带方向的差异池化、排序梯度到新增模块、终止/单候选 backward、独立开关组合与解码器一致性、checkpoint/CLI、N=100/250/500 与不同 M 的单状态前向。**未执行真实参数更新、完整训练或 Chu 评估。** 这些测试不证明学习到正确终局排序，也不证明 gap 下降；训练收敛性、实际 GPU 内存/速度和最终 gap 需在正式实验中验证。

`tests/test_combination_theta.py` 新增 12 项检查：二阶混合差分、互补搭配排序翻转、独立标量公式、零系数精确回退、系数界与排列等变性、资源/收益单位缩放及分块一致性、float64 容量边界、新 θ 梯度与 Critic 隔离、空候选/单资源、贡献诊断、配置/CLI、CUDA 确定性 backward。

`tests/test_performance.py` 检查缓存前后的分数/价值/参数梯度一致性、状态变化后的失效、切片/配置切换、内存预算与分配失败回退、消融开关、PPO 缓存接线、CUDA 分块及 checkpoint 参数兼容。日志/续训流程测试仍使用模拟 rollout 与优化器。

`tests/test_theta_ranking.py` 新增 14 项检查：合法域/零上界/当前 θ 保留、按动作去重和预算、真实后继状态、剔除历史前缀的 Q、ragged 分支共享噪声及重排、独立 RNG/冻结参数、改进与退化标签、无替代动作跳过、真实 θ 采样接线、快照完整重放、排序梯度与温度/Critic 隔离、PPO 接线（mock optimizer）、CLI、CUDA 与不同 N/M 的候选前向接口。

本次接口回归：`python -m unittest discover -s tests -v` 共 **88 项通过**，未执行真实优化器更新、完整训练或 Chu 评估。人工小样例中，多组 θ 诱导相同更优动作时只评估一次，能得到 4.95 的原始后续利润提升标签；该结果可从冻结快照重放，不代表训练后实际 gap 或泛化改善。


## 全流程如下：
1. 输入 MKP 实例
   输入利润 P[B,N]、物品资源消耗 W[B,N,M]、资源容量 C[B,M]。
2. 初始化环境状态
   构造：
   - 剩余容量 remaining = C
   - 已选择标记 selected[B,N]
   - 可行性掩码：物品未被选择且不超过所有剩余容量
3. 状态特征构造
   将当前状态转换为：
   - 物品节点特征：利润、资源消耗、选择状态、可行状态等
   - 资源节点特征：剩余容量、资源紧张度、可行物品比例等
   - 物品—资源边特征：初始消耗、当前消耗、容量超额程度
   - 全局特征：累计收益、可行比例、平均/最小/最大剩余容量
4. 二分图 Transformer 编码
   对物品节点和资源节点分别嵌入，然后重复执行：
   - 物品与资源之间的双向交叉注意力
   - Actor 物品节点间带选后兼容关系的注意力
   - 资源节点内部 self-attention
   - 残差连接、LayerNorm 和前馈网络
   Actor 将边特征用于 attention bias 与 value 消息，影响物品—资源之间的信息交互。
5. 全局状态聚合
   对可行物品节点和资源节点分别池化，与全局特征拼接，再通过可学习查询提取竞争摘要，并聚合相似候选的有向差异，得到 pooled。
6. 高层策略生成
   pooled 输入高层策略头，输出策略参数 θ：
   - 资源价格
   - 利润权重
   - 稀缺性权重
   - 瓶颈惩罚权重
   - 温度参数
   - 选后机会损失和资源余量损失权重
   - 对称资源交互系数与多尺度搭配基函数权重
7. 弱解码器评分
   固定弱解码器使用 θ 对当前可行物品评分。Transformer 不直接生成物品 logits，只调节评分公式中的策略参数。
8. 选择动作
   根据物品分数得到候选策略，采样或选择一个可行物品。
9. 环境状态更新
   执行动作后：
   - 更新剩余资源
   - 标记物品已选择
   - 返回即时奖励
   - 判断是否终止
10. 循环执行
    重复“状态构造 → Transformer 编码 → 生成 θ → 弱解码器评分 → 选择物品”，直到没有可行物品。

### 核心数据流：
MKP实例
→ 动态环境状态
→ 物品/资源/边/全局特征
→ 二分图Transformer
→ 全局状态表示
→ 高层策略θ
→ 固定弱解码器评分
→ 可行物品动作
→ 状态更新与奖励
→ 循环
训练时，PPO 根据实际物品动作的概率更新 Transformer 和高层策略，并增加同状态关键候选的终局收益排序监督；Critic 使用独立的二分图编码器估计状态价值。
