# Transformer 高层策略控制的强化学习 MKP

实现路径：**动态 MKP 状态 → 二分图 Transformer → 高层策略 θ → 固定弱解码器 → 可行物品 → 状态更新**。

目标是在 `MKP_dataset/Benchmarks_Chu.mat` 上研究接近或低于 **0.1% gap** 的解。该目标需要实验验证，代码交付本身不代表达到该精度。默认训练与测试均由使用者运行。

当前为 **controller v4**：针对 profit、资源消耗相近，但终局收益不同的关键候选，增强高层控制器的选后感知、关系注意力和差异聚合。**本轮保持 `StrategyTheta`、θ 范围、`WeakDecoder` 评分公式及原有终局排序监督不变**，解码器版本仍是 `opportunity_density_v3`。checkpoint schema 升为 4，控制器标识为 `candidate_relations_v4`；v1/v2/v3 checkpoint 不能续训或直接评估，需要重新训练，默认产物存入独立的 `checkpoints/controller_v4/`。

## 本轮核心修改：让 θ 看见关键候选的差别

动机来自用户提供的 oracle_ceiling 实验：曾找到能通过合法 θ 选择更优物品的轨迹。因此本轮优先改善控制器如何感知状态、如何把差异传到 θ；没有在本次交付中重新运行或独立验证 oracle 实验。原网络已有边 Value 和竞争查询，但没有显式读取每个候选的选后兼容关系，且已选、不可行物品仍参与注意力和均值池化。

| 改动 | 网络获得的信息 | 代码位置 |
| --- | --- | --- |
| 选后物品特征 | 选择 i 后阻断的可行收益质量、平滑兼容收益质量、其他物品仍可行的比例、平滑余量损失，共 4 维 | `mkp_attention.py:candidate_relations`，`mkp_Model.py:BipartiteEncoder` |
| 选后资源边特征 | 每个资源分别造成的阻断收益质量和余量损失，共 2 维；保留应调整哪个资源价格的信息 | `candidate_relations`，`BipartiteBlock` 的双向交叉注意力 |
| 物品关系注意力 | 利润差、负载差、共同装入余量、是否冲突等关系同时进入 attention bias 和 Value；即使节点 Value 接近，也能传递不同消息 | `mkp_attention.py:PairRelationAttention` |
| 相似候选差异聚合 | 将候选与相近物品比较，保留有方向的 embedding 差和选后特征差；通过差异强度及可学习查询汇聚关键候选 | `mkp_attention.py:CriticalContrastPool` |
| 可行候选聚合 | Actor 的物品消息、物品均值及竞争查询仅聚合当前可行物品；全局历史摘要仍保留 | `mkp_Model.py:EdgeCrossAttention/BipartiteEncoder` |

对当前可行且不同的 i、j，先用 float64 计算 `r_ijm = (remaining_m - w_im) - w_jm`。任一维 `r_ijm < 0` 表示选择 i 后 j 无法再装入。所有收益质量都除以选择前的可行总收益；排除自身、已选及当前不可行物品。资源边上的阻断归因逐资源计算，同一个物品可以在多个资源上冲突，因此各资源归因不应相加解释为总阻断收益。

物品间 7 维关系为：`(p_i-p_j)/max(p_i+p_j,eps)`、平均 `|w_i-w_j|/remaining`、最小相对共同余量、是否阻断、平滑兼容度、相对选择前的平滑余量损失、j 的可行收益质量。实现对零分母作下界保护，对极端相对负载作截断；容量是否超限始终使用原始 float64 余量比较。差异池化按利润差与负载差为其他候选分配相似性权重，不依赖物品序号或硬 top-k；再用候选自身与相似候选的差异构成非线性消息，避免正负差直接求平均后抵消。

例如 A=`(p=10.05,w=[5.2,5])`、B=`(p=10,w=[5,5.2])`、C=`(p=5,w=[5,1])`，剩余容量为 `[10,10]`：A、B 即时属性相近，A 会阻断 C，B 保留 C。新 Actor 可以直接看到这一区别及其资源归因，再通过 θ 改变既有公式的相对评分。**这些是当前状态的静态兼容信息，不是终局收益预测，也不能保证识别所有多步组合效应。** 终局方向仍由 PPO 和已有的冻结策略续行排序损失学习；排序标签仍是当前策略下的收益估计。

数据流：`当前状态 → 选后关系特征 → Actor 二分图与物品关系注意力 → 相似候选差异汇聚 → 原有 θ → 原有弱解码器`。新增注意力权重仅用于内部表示，不输出动作分数、不预测物品 Q 值、不在推理时试走轨迹。Critic 的独立编码器、PPO/GAE、奖励及 ranking 标签构造保持原样。原有 θ 输出头的零权重初始化也保留：策略/排序分支在初始化时首先更新 θ 输出头，之后梯度传入上游表示；可行性辅助分支仍可从首步向 Actor 编码器传递梯度。梯度测试使用非零 θ 头权重验证排序通路，没有执行优化器更新。

### 本轮消融与计算开销

四个开关默认开启，可分别关闭并组合做因子消融，使用相同种子、训练预算、独立 checkpoint 目录，最终比较配对收益/gap：

| 消融 | 命令参数 |
| --- | --- |
| 不向物品输入/资源边注入选后特征 | `--no-actor-postdecision-features` |
| 恢复普通物品自注意力 | `--no-actor-pair-attention` |
| 关闭相似候选差异池化 | `--no-actor-contrast-pooling` |
| 恢复普通注意力/均值/竞争查询中的全物品聚合 | `--no-actor-feasible-attention` |

前 3 个开关控制不同使用位置：关闭特征注入时，关系注意力和差异池化仍可使用关系特征；关系模块始终排除自身和无效候选。关闭第 4 项只恢复普通聚合中的全物品读取，不会把无效候选引入关系模块。四项全部关闭恢复 v3 的 Actor 结构，保留原有 v3 解码器与训练信号，但仍需以 v4 checkpoint 格式重新训练。

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
& 'D:\Zlb08\.envs\RL_MKPtest\Scripts\python.exe' train_ConstrainedMKP.py --resources 5 --resume checkpoints/controller_v4/mkp_M5_last.pt --updates 4000
# 可选：另报 128 条随机轨迹中的最佳解，不替代单轨迹 greedy 结果
& 'D:\Zlb08\.envs\RL_MKPtest\Scripts\python.exe' Benchmark_zeroShot.py --sample-rollouts 128
```

其余参数集中在 `mkp_config.py`，包括模型、生成器、PPO、日志、checkpoint、benchmark。`--help` 只显示参数，不运行实验。运行目录不会影响默认数据和产物路径。

## 模块与输出

| 文件                      | 职责                                                      |
| ------------------------- | --------------------------------------------------------- |
| `mkp_config.py`           | 所有可调整默认参数及校验                                  |
| `mkp_data.py`             | 调用现有 `read_mat.py`，统一数组方向；独立随机训练数据    |
| `mkp_env.py`              | float64 容量与收益记账，逐步选择，终止与可行候选集合      |
| `mkp_Model.py`            | 二分图多头 Transformer、高层策略、固定解码器、独立 Critic |
| `mkp_attention.py`        | Actor 选后兼容特征、物品关系注意力、相似候选差异池化       |
| `mkp_Trainer.py`          | 完整 episode 采样、GAE、PPO、合成验证、日志、checkpoint   |
| `mkp_ranking.py`          | 关键候选筛选、固定策略成对续行、收益标签与排序损失       |
| `train_ConstrainedMKP.py` | 简洁训练入口                                              |
| `Benchmark_zeroShot.py`   | 冻结推理、独立解校验、CSV 与物品序列                      |
| `mkp_utils.py`            | 复现设置、checkpoint 合约、共享构造过程                   |
| `tests/test_contracts.py` | 小张量接口与梯度校验，不执行训练或 benchmark              |

- `runs/<时间戳>/train.log`：每次更新的日志。
- `runs/<时间戳>/metrics.csv`：收益、归一化 return、利用率、PPO 各项损失、归一化 entropy、KL、梯度范数、Critic explained variance、合成验证表现。
- `runs/<时间戳>/config.json`：配置与运行环境。
- `checkpoints/controller_v4/mkp_M5_best.pt`、同目录 `mkp_M10_best.pt`：按固定合成验证集的 greedy 归一化收益选择。
- `checkpoints/controller_v4/mkp_M5_last.pt`、同目录 `mkp_M10_last.pt`：最近一次保存的模型，包含优化器和 RNG 状态；默认每 100 次更新、产生新 best 及训练结束时保存。
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

| TrainConfig 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `convergence_tolerance_pct` | `0.05` | 相对最佳验证收益的百分比容差，不是 Benchmark gap 百分点 |
| `convergence_patience_updates` | `1000` | 连续平台至少跨越的 update 数 |
| `convergence_min_evaluations` | `20` | 平台至少包含的有效验证次数 |
| `convergence_min_updates` | `1000` | 最早允许确认平台的累计 update |
| `plot_smoothing_updates` | `100` | RL 曲线平滑窗口，仅影响显示 |

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

高层每步只输出 M 个正资源价格及 6 个标量，共 M+6 个数，维度不随 N 增长：

- `resource_prices`：均值为 1，避免价格尺度冗余。
- `profit_exponent`：收益竞争强度 α。
- `scarcity_exponent`：资源稀缺性强度 β。
- `bottleneck_weight`：单一瓶颈占用惩罚 κ。
- `temperature`：随机策略温度 τ；只影响采样概率，在同一步 greedy 排序中不改变 argmax。
- `opportunity_weight`：选后永久失去的可行收益机会的惩罚强度 λ。
- `reserve_weight`：选后资源余量下降的平滑惩罚强度 ρ。

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
pi_i    = softmax(score_i / temperature)，只在可行候选集合中归一化
```

零总资源消耗的物品用 eps 成本处理；数值实现使用 log-sum-exp。默认固定对照为 price=1、α=1、β=1、κ=0.1、τ=0.2、λ=ρ=0，保留原始密度基线。网络初始化为同样的密度参数，并令 λ=ρ=0.1。

弱解码器没有可训练参数、物品指针网络、局部搜索、修复、beam search 或优化求解器。原始密度评分为 O(KM)，新增机会项为 O(KNM)，K 为可行物品数；候选按默认 128 个分块，额外峰值张量为 O(128NM)。`fixed` 对照会跳过未使用的机会项计算。机会项增加了实际推理成本；不能把无参数等同于没有计算开销。

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

每次 PPO 更新之前，从本轮合成 rollout 抽取最多 64 个状态。以实际 greedy 候选为锚，在最高分的 8 个可行候选中，按利润差、逐资源占用差和原始 score 差选择相近对手。最多保留 16 个状态：约 3/4 优先取难区分的状态，其余随机选取，限制只盯着最小分差带来的偏置。

对每个状态复制两个分支，分别先选 i、j；后续都使用**本次更新前、参数固定的策略**构造到终点。每对执行 1 组 greedy 续行和 4 组随机续行；随机分支在相同步骤对相同原始物品共享 Gumbel 噪声。标签为归一化终局收益差 Δ，默认各占 50% greedy 和随机均值。过滤 `|Δ| <= max(0.001, SE)` 的小差异/噪声样本，也过滤 greedy 与随机均值明确冲突的方向。该标签是当前策略下的续行估计，不是最优 Q 值或 oracle。

```text
target = sigmoid(delta / ranking_return_scale)
margin = (score_i - score_j) / ranking_score_scale
ranking_loss = BCEWithLogits(margin, target)
total_loss = 原 PPO/Value/可行性/Entropy 损失 + 0.1 * ranking_loss
```

排序使用温度缩放**之前**的 score，模型不能只调低采样温度来满足排序监督。标签及续行均无梯度；梯度由固定 score 经 θ 回到 Actor，不进入 Critic。反事实分支不进入 PPO buffer，不改变 GAE/Value 的目标。每个 PPO minibatch 另外计算整个标签状态集合的全候选 KL，超过 `target_kl` 即停止后续更新，避免只检查已采样动作而遗漏其他候选的概率漂移。标签使用独立 RNG，不消耗主 rollout 随机流。

`metrics.csv` / `train.log` 记录标签提议/接受数、平均终局收益差、标准误、标签耗时、排序损失、排序方向正确率和全候选 KL。`ranking_accuracy` 是当轮被接受训练对上的方向正确率；没有接受对时记 0，应结合 `ranking_pairs` 阅读，不能当作独立验证精度。Benchmark 记录 λ、ρ 均值，checkpoint/manifest 保存 v4 控制器标识、原 v3 解码器标识与完整模型配置。推理不执行反事实续行。

### 运行与独立消融

默认命令已开启全部改动。以下是原 v3 机制的开关，可与前述 v4 控制器开关组合。建议每组使用相同种子、训练预算及独立 checkpoint 目录，最终比较同一批实例的配对收益/gap：

| 关闭的部分 | 训练参数 |
| --- | --- |
| 长程排序监督 | `--ranking-coefficient 0` |
| 选后机会评分 | `--opportunity-weight-max 0 --reserve-weight-max 0` |
| Actor 边 Value | `--no-actor-edge-values` |
| 竞争摘要查询 | `--opportunity-tokens 0` |

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

本次接口回归：`python -m unittest discover -s tests -v` 共 **53 项通过**，包括现有 CUDA 确定性梯度检查。源码 AST 对比确认 `StrategyTheta`、`HighLevelController` 输出头、`CandidatePolicy` 和 `WeakDecoder` 实现未改变；Python 源码编译检查与 `git diff --check` 通过。


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
