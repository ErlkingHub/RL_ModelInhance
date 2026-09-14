# Transformer 高层策略控制的强化学习 MKP

实现路径：**动态 MKP 状态 → 二分图 Transformer → 高层策略 θ → 固定弱解码器 → 可行物品 → 状态更新**。

目标是在 `MKP_dataset/Benchmarks_Chu.mat` 上研究接近或低于 **0.1% gap** 的解。该目标需要实验验证，代码交付本身不代表达到该精度。默认训练与测试均由使用者运行。

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
& 'D:\Zlb08\.envs\RL_MKPtest\Scripts\python.exe' train_ConstrainedMKP.py --resources 5 --resume checkpoints/mkp_M5_last.pt --updates 4000
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
| `mkp_Trainer.py`          | 完整 episode 采样、GAE、PPO、合成验证、日志、checkpoint   |
| `train_ConstrainedMKP.py` | 简洁训练入口                                              |
| `Benchmark_zeroShot.py`   | 冻结推理、独立解校验、CSV 与物品序列                      |
| `mkp_utils.py`            | 复现设置、checkpoint 合约、共享构造过程                   |
| `tests/test_contracts.py` | 小张量接口与梯度校验，不执行训练或 benchmark              |

- `runs/<时间戳>/train.log`：每次更新的日志。
- `runs/<时间戳>/metrics.csv`：收益、归一化 return、利用率、PPO 各项损失、归一化 entropy、KL、梯度范数、Critic explained variance、合成验证表现。
- `runs/<时间戳>/training_curves_M5.png`、`training_curves_M10.png`：按 M 分开的 4×3 训练诊断图，首次更新、每 100 次更新、各模型结束及退出时刷新；浅色原始曲线，深色为最近 25 条有效观测的均值。
- `runs/<时间戳>/config.json`：配置与运行环境。
- `checkpoints/mkp_M5_best.pt`、`mkp_M10_best.pt`：按固定合成验证集的 greedy 归一化收益选择。
- `checkpoints/mkp_M5_last.pt`、`mkp_M10_last.pt`：最近一次保存的模型，包含优化器和 RNG 状态；默认每 100 次更新、产生新 best 及训练结束时保存。
- `results/<时间戳>/instances.csv`：逐实例、逐方法报告。
- `results/<时间戳>/summary.csv`：每种方法按 M/N 分组及总体报告。
- `results/<时间戳>/solutions.jsonl`：零基索引物品选择顺序，可独立复核。
- `results/<时间戳>/manifest.json`：数据及 checkpoint SHA256、预算、设备、配置、运行状态。

重新训练会更新同一 M 的标准 checkpoint 文件名；日志目录保留每次配置。继续训练时保留原模型、数据分布、训练 N 和固定验证设置。

## 当前代码中的二分图可以概括为：
- 左侧节点：N 个物品节点。每个节点包含利润、初始资源消耗、当前可行性、是否已选择等 8 维特征。
- 右侧节点：M 个资源节点。每个节点包含剩余容量比例、资源需求、已选择比例、可行物品比例等 6 维特征。
- 边：每个物品与每个资源相连，即形成完整的物品—资源关系。边特征为：
  1. 初始消耗；
  2. 当前消耗；
  3. 当前相对容量的超额程度。
代码中没有显式构造邻接矩阵，而是用 edge_features[B,N,M,3] 表示边，并在交叉注意力中作为 attention bias 使用。
Transformer 的作用是：
1. 分别将物品节点和资源节点映射到统一的 embedding 空间；
2. 通过物品→资源、资源→物品的交叉多头注意力，让两类节点交换信息，并利用边特征体现资源消耗关系；
3. 通过各自的 self-attention，让物品之间比较利润与负载，资源之间比较紧张程度和瓶颈；
4. 对所有节点进行池化，形成全局状态表示；
5. 高层策略头根据该表示输出低维策略参数 θ，包括：
   - 资源价格 resource_prices
   - 利润权重 profit_exponent
   - 稀缺性权重 scarcity_exponent
   - 瓶颈惩罚 bottleneck_weight
   - 温度 temperature
随后，固定的弱解码器使用 θ 对当前可行物品打分并选择物品。因此，Transformer 不直接输出每个物品的动作概率，而是根据全局二分图状态调整解码器的策略参数。

## 模型如何感知状态与约束

物品节点包含归一化收益、初始资源占用、已选状态、当前可行性、资源超额程度与当前容量下的占用比例。资源节点包含剩余容量比例、已用比例、未选物品需求和可行物品需求。二分图边包含每个物品对每个资源的初始占用、动态占用与超额量。

每个模块包含：物品→资源与资源→物品的带边偏置多头交叉注意力、物品间多头自注意力、资源间多头自注意力、前馈网络。默认 8 头、128 维、2 层。节点没有位置编码，支持不同 N/M 的输入；默认每个 M 单独训练，避免未经验证地宣称跨资源维度泛化。

物品自注意力可学习收益与资源占用的竞争关系，资源自注意力可学习瓶颈资源的竞争关系，但**不预先宣称每个注意力头已经学到了某种语义**。全局剩余容量、已选比例和累计收益也进入策略汇聚。已选和不可行物品仍可作为状态背景参与注意力。

另有当前可行性的辅助预测损失，帮助编码器学习约束表示；该预测头的输出不参与物品评分。Critic 使用独立编码器，不向弱解码器提供预测收益。

### 关于 mask

注意力没有 `attn_mask`/`key_padding_mask`，没有把不可行物品的策略 logits 设成 `-inf`。环境根据真实容量生成可行候选索引，解码器使用分段 softmax 构成不同候选数的 categorical 分布。终止行使用动作 `-1`。

这里仍有布尔型“可行集合”和“已选状态”；它们是 MKP 的物理约束检查。完全取消可行性检查无法保证输出合法解。约束状态显式进入模型，合法性由环境再次检查，两者同时保留。评分公式内部的 `-inf` 仅用于表示零资源消耗的对数加法项，不是屏蔽物品 logits。

## 高层策略与固定弱解码器

高层每步只输出 M 个正资源价格及 4 个标量，共 M+4 个数：

- `resource_prices`：均值为 1，避免价格尺度冗余。
- `profit_exponent`：收益竞争强度 α。
- `scarcity_exponent`：资源稀缺性强度 β。
- `bottleneck_weight`：单一瓶颈占用惩罚 κ。
- `temperature`：随机策略温度 τ；只影响采样概率，在同一步 greedy 排序中不改变 argmax。

设 p_i 为收益、w_ij 为占用、C_j 为初始容量、c_j 为当前剩余容量：

```text
q_ij    = N * w_ij / C_j
s_j     = C_j / max(c_j, eps)
cost_i  = mean_j(price_j * q_ij * s_j ** beta)
score_i = alpha * log(max(p_i / mean(p), eps))
          - log(cost_i)
          - kappa * max_j(w_ij / max(c_j, eps))
pi_i    = softmax(score_i / temperature)，只在可行候选集合中归一化
```

零总资源消耗的物品用 eps 成本处理；数值实现使用 log-sum-exp。默认固定对照为 price=1、α=1、β=1、κ=0.1、τ=0.2。默认网络的高层头初始化在该策略附近，避免把随机劣质起点误当作弱基线。

弱解码器没有可训练参数、物品指针网络、局部搜索、修复、beam search 或优化求解器。每步评分的开销为 O(KM)，K 为可行物品数；整体模型的物品自注意力仍有 O(N²) 开销，不能把弱解码器的速度等同于完整模型推理速度。

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

## 条件价格头与训练诊断更新

默认 `ModelConfig.price_head_type="conditional"`，资源价格 logits 改为：

```text
z_j = v^T tanh(W_r h_j + b_r + W_g g)
price_j = price_floor + (1 - price_floor) * M * softmax(z)_j
```

全局状态在 tanh 之前与各资源表示交互，能够改变相对资源价格。最终投影使用标准差 `price_head_init_std=0.01` 的小非零权重，使初始价格接近均匀，同时从第一步就保留上下文梯度。二分图编码器、独立 Critic、其他策略参数、弱解码器 score 以及 PPO 优化目标均保持原设计。

旧 checkpoint 缺失 `price_head_type` 时自动按旧线性价格头加载，不会隐式转换成新结构。新结构需要从头训练；直接用旧权重恢复默认新结构训练会明确报错。显式设置 `price_head_type="linear"` 可用于旧结构对照及兼容续训。已有 checkpoint 文件不会因加载或绘图而被修改。

图表包含训练归一化收益、固定合成验证收益/历史最好值、相对固定弱策略的验证提升、policy/total/value/auxiliary loss、归一化策略熵、KL、clip fraction、实际 PPO 更新比例、提前停止频率、裁剪前梯度范数、Critic explained variance 与 MC RMSE、利用率/选中比例及高层策略统计。

新增 CSV 字段的含义：

- `total_loss`：按现有系数加权后的 PPO 总损失，按实际更新的样本数平均。
- `entropy_raw` / `rollout_entropy_raw`：原始 categorical entropy，单位 nats；原有 `entropy` 和新增 `rollout_entropy` 都按候选数归一化。前者在 PPO 更新中记录，后者在采集轨迹时记录。
- `stop_kl`：触发提前停止的那个 minibatch 的 KL；未停止时留空。原有 `approx_kl` 仍只统计已接受的 minibatch。
- `ppo_update_fraction`：实际完成的 transition 更新次数 / 全部 PPO epochs 计划的 transition 更新次数。
- `critic_mc_rmse` / `critic_mc_ev`：采集时的 Critic 预测与完整折扣 Monte Carlo return 的误差及 explained variance，不使用价值 bootstrap，排除终止后的 padding。原有 `explained_variance` 继续使用 GAE 目标；两者并列显示，均不反馈进优化。
- `theta_alpha/beta/kappa/temperature/price_std`：对采集轨迹中所有有效决策等权平均；终止行不参与。
- `selected_fraction`：episode 结束时的平均选中物品比例。

绘图参数为 `TrainConfig.plot_every`、`plot_smoothing_window`、`plot_dpi`。验证点不做平滑；不同 N 的训练收益分别画线。图中固定验证收益与训练收益来自不同数据集，不直接相减；相对弱策略提升也不等于 Chu 最优性 gap。

可以随时只读取已有日志补画 PNG，不加载 checkpoint，也不执行环境或推理：

```powershell
& 'D:\Zlb08\.envs\RL_MKPtest\Scripts\python.exe' mkp_diagnostics.py --run-dir runs/20260910_154547_742166
```

旧日志可直接绘图，未记录的新指标显示为缺失，不用零值填补。旧日志生成的图片标为 `linear (legacy)`，代表旧实验。绘图依赖 matplotlib；训练期间绘图失败会写入日志，保留 CSV 和 checkpoint，可在修复绘图环境后离线补画。

## 交付验证

`tests/test_contracts.py` 只检查小张量上的候选概率、固定公式、约束拒绝、终止/GAE、排列等变性、高层梯度、checkpoint 往返及独立解校验；不含优化器更新、训练 episode 或 Chu 推理。训练收敛性、实际 GPU 内存/速度和最终 gap 需在正式实验中验证。


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
   - 物品节点内部 self-attention
   - 资源节点内部 self-attention
   - 残差连接、LayerNorm 和前馈网络
   边特征被转换为 attention bias，影响物品—资源之间的信息交互。
5. 全局状态聚合
   对物品节点和资源节点分别平均池化，并与全局特征拼接，得到全局状态表示 pooled。
6. 高层策略生成
   pooled 输入高层策略头，输出策略参数 θ：
   - 资源价格
   - 利润权重
   - 稀缺性权重
   - 瓶颈惩罚权重
   - 温度参数
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
训练时，PPO 根据实际物品动作的概率更新 Transformer 和高层策略；Critic 使用独立的二分图编码器估计状态价值。
