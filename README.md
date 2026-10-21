# 资源路由 Transformer 高层策略控制 MKP

当前默认 Actor 为 `resource_routing_v1`：

```text
动态 MKP 状态
  → 可行候选与资源几何
  → [候选汇聚到资源 → 资源竞争注意力 → 更新后的资源回传候选] × 2
  → 分别归一化的物品/资源策略读出
  → 状态级 θ → 原 v5 固定弱解码器 → 单个可行物品 → 状态更新
```

研究目标是降低 `learned_dynamic` 单轨迹 greedy 的 gap。历史 `results/20260922_083444_537744` 中，180 例平均 gap 为 **0.3997080664%**，M=10/N=100 为 **0.7701558973%**。本轮只完成代码和接口验证，**没有正式训练或新 Chu 评估，不能宣称已经或保证达到 <0.1%**。正式实验由使用者运行。

## 本轮结构变化

原模型已含选后物品关系和两物品搭配基函数。本轮关注资源之间的竞争表示和信息传递顺序，具体实现见 `mkp_routing.py`。

| 部分 | 当前设计 | 对资源分配的作用 |
|---|---|---|
| 资源输入 `[B,M,12]` | 可行负载均值/最大值/标准差、收益加权负载、需求压力、瓶颈归因、容量比例等 | 区分相同总需求下的收益竞争与瓶颈责任 |
| 资源对 `[B,M,M,5]` | 负载余弦、相关系数、收益加权余弦、压力差和负载差异 | 区分共同紧张、资源替代与消耗互补的状态 |
| 顺序消息传递 | gather → resource attention → broadcast | 每轮回传物品时，已纳入本轮候选需求和资源竞争信息 |
| 资源汇总 | 压力偏置汇总、逐维最大值及标准差 | 减少少数瓶颈在全资源均值中被稀释 |
| 九个策略角色 | 六个标量、价格、交互、搭配分别读取；两类节点各自归一化后融合 | 减少所有控制量共用一个摘要的压缩，以及候选数量对资源注意力占比的影响 |

默认仍是 D=128、8 个头、2 轮。注意力头没有预设的经济语义；上述特征也不是最优影子价格或未来价值。

新 Actor 不构造 `[B,N,N,*]` 物品对关系或物品自注意力，通过资源节点中转候选竞争信息。固定 M 时，Actor 的注意力存储随 N 线性增长，几何计算约 O(BNM²)。**固定解码器的机会项/搭配项仍有物品对计算，Critic 仍含物品自注意力；整个模型与完整训练并未变成线性复杂度。** 资源中转也可能损失直接物品对建模能力，效果需由消融验证。

## 研究边界与兼容性

- θ 仍是 `M+6+M(M+1)/2+K` 个非重复条目，默认 K=3，M5/M10 分别为 29/74。字段、范围及弱解码器评分公式均未改变。
- Actor 只输出状态级策略控制量。可行性辅助头不参与解码评分；没有 learned item scorer、搜索、修复或推理求解器。
- Critic 仍使用独立的原编码器。PPO、奖励、GAE 与原成对续行监督保持原逻辑。
- v5 θ/decoder 标识和 schema=5 保留。**网络结构由 checkpoint 的 `model_config.actor_encoder` 和 `actor_strategy_readout` 明确区分**。
- 旧 v5 checkpoint 缺少新字段时，`ModelConfig.from_checkpoint()` 恢复 `legacy` / `False` 并严格加载原 state dict。不会把旧权重解释为新网络。其他历史 schema 仍拒绝加载。
- 新结构需要重新训练，默认独立目录为 `checkpoints/resource_routing_v1`。原 `combination_v5` 权重和历史结果未修改。

当前磁盘上原目录的 M5/M10 best 为 update=1550/1500，SHA 与历史 0.399708% manifest 所用的 update=1100/1875 **均不匹配**。因此，当前同名 checkpoint 只能作为现存旧模型使用，不能声称它复现该历史数字。核验记录在 `outputs/resource_routing_v1/source_and_provenance.json`。

## 训练与评估

在项目目录、已安装 torch/numpy/scipy/tqdm/matplotlib 的 Python 环境中运行：

```powershell
python train_ConstrainedMKP.py --checkpoint-dir checkpoints/resource_routing_v1
python Benchmark_zeroShot.py --checkpoint-dir checkpoints/resource_routing_v1
```

默认分别训练 M=5、10，合成数据 N=100，每个 M 为 2000 updates；batch=64、PPO minibatch=128、3 epochs、learning rate=1e-4。没有通过扩大默认训练规模或预算改变对照。Benchmark 默认测试 M=5/10、N=100/250/500 共 180 个 Chu 实例，使用合成验证选出的 best checkpoint。结果查看 `summary.csv` 中 `method=learned_dynamic, M=all, N=all` 的 `mean_gap_pct`；同时查看分组、最差值、命中率和可行率。

```powershell
# 单独训练 M=10
python train_ConstrainedMKP.py --resources 10 --checkpoint-dir checkpoints/resource_routing_v1

# 续训新结构；updates 为累计目标，其他模型与训练数据配置应一致
python train_ConstrainedMKP.py --resources 5 --resume checkpoints/resource_routing_v1/mkp_M5_last.pt --updates 4000

# 可单独调整结构大小；Benchmark 从 checkpoint 读取实际配置
python train_ConstrainedMKP.py --embed-dim 128 --num-heads 8 --num-layers 3 --checkpoint-dir checkpoints/resource_routing_3layers
```

可先以相同 seed/预算做以下四种结构的对照。各行使用独立目录；训练后对对应目录运行同一 Benchmark 命令。建议之后再做多种子重复。

```powershell
# 原 v5 网络，从头训练的匹配对照
python train_ConstrainedMKP.py --actor-encoder legacy --no-actor-strategy-readout --checkpoint-dir checkpoints/routing_ablation_legacy

# 只更换资源路由编码器
python train_ConstrainedMKP.py --no-actor-strategy-readout --checkpoint-dir checkpoints/routing_ablation_encoder

# 只更换策略读出
python train_ConstrainedMKP.py --actor-encoder legacy --checkpoint-dir checkpoints/routing_ablation_readout

# 默认组合
python train_ConstrainedMKP.py --checkpoint-dir checkpoints/resource_routing_v1
```

`--actor-edge-values`、`--actor-postdecision-features`、`--actor-pair-attention`、`--actor-contrast-pooling`、`--actor-feasible-attention`、`--opportunity-tokens` 及 actor relation 参数用于 **legacy 编码器**。资源路由编码器使用固定可行性掩码及本文件所述几何，不读取这些旧架构开关。解码器参数和执行缓存参数仍对两种架构有效。

## 缓存与日志

`--performance-profile auto` 在至少 20GiB 的 CUDA 设备上启用 `memory24`，其他设备用 `reference`。可用 `--performance-profile reference` 关闭跨 epoch 批量缓存；缓存只保存无参数几何，模型参数与 θ 每次重新计算。缓存签名包含编码器类型，状态原地更新或替换后失效。

默认 `--feature-cache-gib 6 --cache-headroom-gib 6 --feature-cache-batch-size 64`。它们限制缓存预算和构建方式，不是整个训练的显存上限。新结构的 GPU 吞吐量和完整训练速度尚未测量。

`runs/<时间戳>/` 保存配置、训练日志、CSV、合成验证曲线和性能平台诊断；平台判定不等于 Chu gap 收敛，也不提前停止默认预算。日志中的 `RANKING accepted/proposed` 可检查监督覆盖；当前排序标签仍可能稀疏，结构改造不自动解决该问题。

`results/<时间戳>/` 保存逐实例结果、summary、解序列、数据/权重 SHA 和训练诊断关联。请按 checkpoint 身份及同一实例/参考值比较，不能把高利用率直接当成高收益。

## 验证

```powershell
python tests/run_routing_cpu_checks.py
# 只做源码边界、现存旧权重身份和人工状态前向核验
python tests/run_routing_cpu_checks.py --audit-only
```

当前 CPU 检查：**85 项，81 通过、4 个 CUDA 检查跳过、0 失败/错误**。覆盖梯度连通、Actor/Critic 隔离、置换/单位不变性、资源竞争反例、空/单候选、N=100/250/500 接口、缓存分数/梯度一致、旧新 checkpoint 以及模拟训练保存/续训。没有真实优化器更新或正式训练。记录位于 `outputs/resource_routing_v1/`。

初始目录已有六个未跟踪的旧 v8/v9 测试，依赖当前 checkout 不包含的 resource/teacher/imitation 模块；本轮保留这些文件。上述入口明确运行本分支六个既有测试模块和新路由测试，不导入这些遗留测试。

默认网络参数从 2,550,093 降到 2,402,653；Actor 从 1,452,076 降到 1,304,636。B=1/N=500/M=10 时，新 Actor 几何输出共 98,504 bytes，旧候选关系输出为 7,298,000 bytes。**这只是对应几何张量的大小，排除了注意力激活、梯度、解码器和 Critic，不能当作总显存或速度测量。**

原弱解码器公式、历史计时与旧结构说明保存在 [docs/legacy_v5.md](docs/legacy_v5.md)，其中路径及默认值属于改造前版本。
