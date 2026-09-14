# BGNN 分支：Item–Constraint 二分图 + PPO

本分支默认运行 **BGNN + 动态高层策略 + 固定弱评分**，保留可选的 **BGNN + 直接 item logits + mask** 对照。训练和完整 benchmark 由使用者运行。

## 运行命令

在项目目录执行；默认配置保留你当前设置的 **800 次更新、M=5/10、训练 N=100**：

```powershell
# 主方法：训练 BGNN + weak head
& 'D:\Zlb08\.envs\RL_MKPtest\Scripts\python.exe' train_ConstrainedMKP.py
# 主方法：仅测试 learned_dynamic，180 个 Chu 实例
& 'D:\Zlb08\.envs\RL_MKPtest\Scripts\python.exe' Benchmark_zeroShot.py

# 可选：训练直接 item 策略头，再将两个已训练头配对比较
& 'D:\Zlb08\.envs\RL_MKPtest\Scripts\python.exe' train_ConstrainedMKP.py --policy-head direct_item
& 'D:\Zlb08\.envs\RL_MKPtest\Scripts\python.exe' Benchmark_zeroShot.py --compare-heads

# 或从头依次训练两种头（会更新两种实验自己的 checkpoint）
& 'D:\Zlb08\.envs\RL_MKPtest\Scripts\python.exe' train_ConstrainedMKP.py --compare-heads

# 只评估直接 item 策略
& 'D:\Zlb08\.envs\RL_MKPtest\Scripts\python.exe' Benchmark_zeroShot.py --policy-head direct_item

# 恢复同一种结构的训练，updates 是累计更新次数目标
& 'D:\Zlb08\.envs\RL_MKPtest\Scripts\python.exe' train_ConstrainedMKP.py --resources 5 --resume checkpoints/bgnn_weak/mkp_M5_last.pt --updates 1600
```

`--encoder transformer` 可显式构造原 Transformer 编码器，但本分支默认 BGNN。BGNN 不含注意力，`num_heads` 仅对 Transformer 生效，`num_layers` 对 BGNN 表示完整的双向消息传递轮数。旧 Transformer checkpoint 不会自动转为 BGNN；新结构必须从头训练。

## 图结构与 Constraint 节点

物品节点 8 维：收益、初始资源占用、已选/可行状态、动态占用及超额量等。Constraint 节点 7 维：

1. 剩余容量比例。
2. 已用容量比例。
3. 未选物品需求 / 当前剩余容量。
4. 可行物品需求 / 当前剩余容量。
5. 已选物品比例。
6. 可行物品比例。
7. 收益加权资源需求压力的相对代理量。

代理量为：

```text
u_j = sum_{未选 i} [(p_i / mean(p)) * w_ij] / max(c_j, eps)
proxy_j = u_j / max(mean_j(u_j), eps)
```

需求特征使用截断后的 log1p 变换，零容量/零需求保持有限。这个量作为“资源价格/对偶信息”的启发式输入，**不是 LP 求解器计算出的对偶变量，也不满足任何已证明的对偶最优性条件**。主方法还会通过条件价格头学习正资源价格，但同样不将其解释为精确对偶解。

边携带 3 维动态消耗信息。`w_ij>0` 表示真实物品—资源连接，零消耗边不传递消息；这是图邻接关系，不是动作可行性 mask。

每轮 BGNN 执行：

```text
Item 表示 + Constraint 表示 + 消耗边特征
  -> 边消息 MLP
  -> 按度数归一化聚合到 Constraint
  -> 残差更新 Constraint
  -> 使用新 Constraint 表示反向传递到 Item
  -> 残差更新 Item
```

消息内容显式包含边特征；不同物品通过 Constraint 节点交流。无 item-item 自注意力，也无 Q/K/V attention。中间消息张量为 `[B,N,M,D]`，对 N/M 的图连接数量是 O(NM)，消息 MLP 仍有隐藏维度计算开销。Actor 和 Critic 都使用 BGNN，但参数完全独立。

编码器保留已选/不可行物品作为状态背景，动作合法性由可行集合和环境检查保证。

## 两种策略头究竟比较什么

| 项目 | 主方法 `weak` | 可选 `direct_item` |
| --- | --- | --- |
| 状态编码 | 每步 BGNN | 每步 BGNN |
| 输出 | 资源价格 + α/β/κ/τ | 每个物品的可学习 logit |
| 评分 | 现有固定弱解码器公式 | 共享 item MLP + 全局上下文，tanh 限幅 |
| 候选处理 | ragged 可行候选 categorical | dense logits + 可行性 mask |
| PPO 训练动作 | 按评分分布采样一件 | 按 masked 分布采样一件 |
| 默认测试动作 | greedy 选择一件 | masked argmax 选择一件 |
| 更新 | 更新容量后重新编码和评分 | 更新容量后重新编码和评分 |

两者都是**自回归构造**，不做一次性静态排序后装包。若对相同 logits 仅切换 ragged 候选和 dense mask，其概率与 greedy 选择应相同，不能形成方法贡献。因此这里额外实现的对照是**固定弱评分与可学习物品评分**；不能把结果差异单独归因于 mask。

主方法保持原非线性条件价格头及固定 score，不增加学习型物品评分；直接 item head 是用户请求的独立比较分支，不混入主方法。直接头没有 θ，其日志/CSV 中的 θ 指标为缺失，图中显示不适用。

直接头分布对已选/不可行物品赋零概率；全不可行行使用安全占位分布避免 NaN，对外动作始终为 -1，log probability 和 entropy 为零。环境仍再次验证所选动作。

## PPO、数据与验证

PPO 继续使用实际 item action 的 log probability。gamma、GAE、clipping、entropy、value/auxiliary loss 和收益归一化均沿用当前配置。每步 reward 是所选物品收益 / 全部物品收益之和，默认 gamma=1。

训练只使用独立合成数据；验证也只跑当前动态策略，不再计算 fixed 或 learned_static 对照。checkpoint 仍按固定合成验证集的 greedy 归一化收益选择，训练日志中的 `validation_improvement_pct` 改为 `validation_objective_mean`。不使用 Chu 来训练或选择 checkpoint。

两种头分别训练。相同 seed、模型公共配置下，初始 BGNN Actor 编码器与独立 Critic 权重匹配，合成数据生成器也使用相同设置。采样器和策略不同，不保证动作轨迹相同。有效比较应保持训练预算、种子、数据分布一致；benchmark manifest 会保存训练配置和 checkpoint 哈希，最佳 checkpoint 的 update 可以不同。

## 输出与曲线

```text
runs/bgnn_weak/<时间戳>/
runs/bgnn_direct_item/<时间戳>/
    config.json
    train.log
    metrics.csv
    training_curves_M5.png
    training_curves_M10.png

checkpoints/bgnn_weak/mkp_M{5,10}_{best,last}.pt
checkpoints/bgnn_direct_item/mkp_M{5,10}_{best,last}.pt

results/<时间戳>/
    instances.csv
    summary.csv
    solutions.jsonl
    manifest.json
    head_comparison.csv  # 仅 --compare-heads
```

默认结果仅包含 `learned_dynamic`；直接头名称为 `direct_item_masked`。配对 CSV 的 `direct_minus_weak_gap_pp` 是 gap 百分点差，负值表示直接头更好；同时记录收益变化和耗时。fixed/static 的提升指标已移除。

默认采用单轨迹 greedy。`--sample-rollouts K` 仍可显式启用额外的 best-of-K 样本结果，以不同 method 行记录；`head_comparison.csv` 只使用两种头的 greedy 结果，不混入多轨迹预算。

PNG 继续展示 loss、policy entropy、KL（含停止时 KL）、clip fraction、实际更新比例、梯度范数、GAE/MC Critic 质量、验证收益和利用率。默认每 100 次更新刷新，结束时保存。离线补画只读取日志：

```powershell
& 'D:\Zlb08\.envs\RL_MKPtest\Scripts\python.exe' mkp_diagnostics.py --run-dir runs/bgnn_weak/你的运行时间戳
```

旧 Transformer 日志和 checkpoint 文件保留。旧 checkpoint 可由加载器按原结构读取，但本分支训练/评测入口按变体目录查找模型，不会偷偷使用旧权重替代未训练的 BGNN。

## 代码与参考来源

- `mkp_bgnn.py`：约束价格代理、双向边消息及 BGNN 编码器。
- `mkp_Model.py`：编码器选择、弱策略头、直接 item 策略头及 mask 分布。
- `mkp_Trainer.py`：统一 PPO 接口、分变体训练与动态验证。
- `Benchmark_zeroShot.py`：动态评测与可选头部对照报告。
- `mkp_config.py`：默认参数和 checkpoint 变体约定。

二分图神经网络思路参考 [Gasse 等，2019](https://arxiv.org/abs/1906.01629) 的变量—约束表示；该论文学习分支定界变量选择，本实现是动态 MKP 构造 + PPO，不是原论文复现。资源压力代理和策略头适配是本项目的设计，尚未通过正式实验验证能降低 gap。

接口检查涵盖图消息、无注意力结构、零消耗边、资源压力状态响应、排列等变性、弱/直接头梯度、mask 与 ragged 概率一致性、终止状态、旧/新 checkpoint 和模拟报告。没有运行完整训练、Chu 推理或性能 benchmark。
