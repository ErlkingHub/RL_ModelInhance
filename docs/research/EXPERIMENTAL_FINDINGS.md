# 实验发现与证据边界

本文解释当前实验支持哪些研究判断，以及哪些判断仍缺少证据。审计基线为 `40_lp-dual-bipartite-bound-shaping@a933ff7`，截至 **2026-10-23**。本次整理只读取已有产物并复算统计、核对保存的解，未训练模型或运行新的 Benchmark。

统一标记：**A 已确认事实**；**B 已实现但未充分验证的机制**；**C 研究假设**；**D 历史探索方向**；**E 后续研究计划**。其中，“读取到某份报告”是 A；报告转述的历史效果是否可复现，需要另行判断。总入口见 [研究概览](RESEARCH_OVERVIEW.md)，证据范围见 [证据索引](EVIDENCE_INDEX.md)。

## 1. 当前能够给出的结论

**A：当前提交保存了一组 LP-dual θ v6 的 Chu 冻结推理结果。** 对 180 个实例，每例执行一条 `learned_dynamic` greedy 轨迹，平均 best-known gap 为 **0.309926%**，保存的 180 个解均可行且已构造至无法再加入物品。该结论可以由 tracked 数据集、逐实例结果和解序列独立核对。

**A：现有证据不支持“已经稳定达到 0.1% gap”。** 总体仅 44/180 个实例满足 `gap_pct <= 0.1`；六组中只有 M=5、N=500 的组均值低于 0.1%。两个 M 的来源收敛摘要均为 `not_converged`。

**B/C：结果尚不能识别高层控制器、LP 特征、LP 评分项或奖励整形各自的增量贡献。** 当前结果没有相同底层评分规则下的固定 θ、静态 θ、未训练 Actor、移除 LP 项或关闭整形的配对对照；也没有多训练种子结果。它是一组系统输出的描述性记录，不能单独证明某个模块有效或优于其他 Solver。

## 2. 可长期追溯的实验产物

当前提交中唯一纳入 Git 的实验结果目录为 [`results/20261022_195937_430788/`](../../results/20261022_195937_430788/)。其中包含：

| 文件 | 作用与可验证范围 |
| --- | --- |
| [manifest.json](../../results/20261022_195937_430788/manifest.json) | 数据与 checkpoint SHA256、推理预算、模型配置、设备和完成状态 |
| [instances.csv](../../results/20261022_195937_430788/instances.csv) | 180 条逐实例 objective、reference、gap、LP bound、利用率和控制量摘要 |
| [solutions.jsonl](../../results/20261022_195937_430788/solutions.jsonl) | 180 个解的零基物品索引及选择顺序，可重新计算目标值与资源使用量 |
| [summary.csv](../../results/20261022_195937_430788/summary.csv) | 六个 M/N 分组和总体统计；能够从 instances 重建 |
| [training_convergence.json](../../results/20261022_195937_430788/training_convergence.json) | 被测 checkpoint 与来源训练收敛摘要的关联；不含完整训练曲线 |

数据集 [`MKP_dataset/Benchmarks_Chu.mat`](../../MKP_dataset/Benchmarks_Chu.mat) 也在 Git 中。文件 SHA256 为：

```text
61da123cbf4ce3ff7178a62065f2f97a31ab60652683544b888d82b00c5fcd6c
```

**A：** 本次重新计算的数据文件摘要与 manifest 完全一致。原 `.mat` 含 270 个实例；当前 Benchmark 按 M∈{5,10}、N∈{100,250,500} 过滤得到 180 个实例，未覆盖该文件全部实例。每个 M/N 组 30 例，原始 `a` 字段分别为容量比例 0.25、0.5、0.75，各 10 例。数据读取口径见 [mkp_data.py](../../mkp_data.py) 和 [read_mat.py](../../read_mat.py)。

## 3. 评价指标：分母、单位与统计口径

### 3.1 best-known gap

令保存解的原始利润为 $z$，数据集给定参考利润为 $z_{ref}$，当前 Benchmark 定义：

$$
\mathrm{gap}_{ref}(\%)=100\frac{z_{ref}-z}{z_{ref}}.
$$

**A：** 这是有符号、未截断的百分数；`0.309926` 表示 **0.309926%**，对应比例约 0.00309926。两组 gap 相减的单位是“百分点”。若将来超过参考值，应保留负 gap，而不是截为零。当前 180 例没有超过参考值，1 例等于参考值。

manifest 将参考类型明确写为 `dataset_best_known_not_verified_optimal`。所以本文使用 **best-known gap**，不将其称为“已证最优 gap”。达到参考值也不能替代最优性证书。实现见 [Benchmark_zeroShot.py](../../Benchmark_zeroShot.py) 中 `benchmark`。

**A：** 分组均值是逐实例 gap 的算术平均，不是先汇总利润再计算比值；总体也是 180 例等权平均。六组各含 30 例，因此本批总体恰好等于六个组均值的等权平均。`target_hit` 的规则是 `gap_pct <= 0.1`，不是严格小于。

### 3.2 LP bound gap

结果另报根状态的上界 $U$ 与：

$$
\mathrm{gap}_{LP}(\%)=100\frac{U-z}{U}.
$$

**A/B：** [mkp_lp.py](../../mkp_lp.py) 以有限步平滑对偶优化产生非负资源价格，再计算弱对偶上界；不能把有限步计算直接写成“精确求得 LP 最优值”。上界的理论有效性来自非负对偶价格对应的对偶目标，而不是 Newton 迭代已收敛。该模块不输出整数最优性证书。

本批 `lp_gap_pct` 平均为 **0.656081%**，与 0.309926% best-known gap 使用不同参考和分母。两列不能互换；也不能将 LP bound 与整数目标之差全归因于策略失误，其中包含松弛界的差距。本次重算核对了保存的 LP gap 公式，未重新运行 LP 优化。

### 3.3 Reward 与资源利用率

**A/B：** 当前代码的利润奖励按实例全部物品总利润归一化；LP potential shaping 的具体形式与成立条件见 [算法设计](ALGORITHM_DESIGN.md)、[mkp_env.py](../../mkp_env.py)、[mkp_Trainer.py](../../mkp_Trainer.py)。训练 return、验证 return、best-known gap 的分母不同，不能直接换算为同一指标。

利用率先在实例内对 M 个资源的 `usage/capacity` 取均值，再跨实例平均。总体 **99.293190%** 说明本批资源使用较满；它既不证明物品组合最优，也不能独立证明模型学习到了长期资源规划。可行性约束和使用量计算属于环境/校验机制，不应当作学习到的约束感知证据。

## 4. 当前结果的实验设置与缺失信息

以下“保存值”来自 manifest 及收敛摘要；配置文件的默认值单列，不能补写成该历史运行的实际参数。

| 项目 | 保存值 / 可确认范围 |
| --- | --- |
| 方法与版本 | `learned_dynamic`；checkpoint schema 6；`lp_dual_theta_v6` / `lp_dual_density_v6` |
| 训练数据来源 | manifest 声明 `synthetic_only`；训练 N=[100]；M=5/10 分别加载模型 |
| 推理预算 | 每实例 1 条 greedy 轨迹；`sample_rollouts=0`，没有 best-of-K 搜索结果 |
| 推理随机种子 | Benchmark config 的 `seed=20260910`；此字段不是训练 seed 的证明 |
| 训练总更新数 | 嵌入的收敛摘要对两个 M 均报告 `final_update=2000` |
| 被测 checkpoint | M5：update 625；M10：update 1175；均使用 `best` 标签 |
| 模型结构 | manifest 保留完整 `model_config`，含 embed_dim=128、8 heads、2 layers，以及 LP 和关系特征配置 |
| checkpoint 选择 | 当前训练代码按固定合成验证集 greedy 收益选 best；保存的 best update 与嵌入摘要一致 |
| 推理环境 | PyTorch 2.11.0+cu126，CUDA，NVIDIA GeForce RTX 4090 D；实际配置 `device=auto` |
| 可行性容差 | 原始资源单位的绝对容差 `1e-7` |
| 完成状态 | manifest `status=complete`，`completed_rows=180` |
| 原训练 run | 记录为 Linux 绝对路径 `/home/zhu/AutoML/librian/RL_165_v1/runs/20261021_120620_435177`；本机与 Git 中均无该目录产物 |

**信息不足：** 当前结果没有完整 `train_config`、`generator_config`、训练配置文件、原训练 metrics/log、完整验证历史，也未记录产生模型的源码 commit。无法从当前保存产物核实实际训练 seed、validation seed、batch、learning rate、PPO 实际 minibatch 数、完整训练时长，以及 LP shaping 或 ranking 的实际运行开关和系数。源码版本与模型配置字段提供部分一致性线索，不能补齐这些缺失项。

例如当前 [mkp_config.py](../../mkp_config.py) 默认 `seed=20260910`、batch=64、learning_rate=1e-4、PPO epochs=3、`lp_reward_shaping=True`；这些是 **A：当前默认配置**，并非该模型已经核实的训练设置。未来应保存完整原始配置，避免依赖默认值回填。

两个 checkpoint 仅有摘要记录，原权重不在当前 Git，也不在本机 `checkpoints/lp_dual_v6/`：

| M | 被测 update | SHA256（manifest 记录，当前无法对权重复验） |
| --- | ---: | --- |
| 5 | 625 | `4e72c4cd9867e38883f1326c4cadc24103b77320b4705f5f75f3bd35e0fcfa95` |
| 10 | 1175 | `44c3cecb969a5c6a28be0024cc9e63e452191d6053680aed2d5ec7c0fa5b59b5` |

因此，目前可复核保存解和统计，无法仅从仓库重新执行同一模型的推理。不得将本机其他目录中的 v5/v8/v9 权重替换后称为复现当前 v6。

## 5. 六组结果与总体结果

**A：** 以下数值重算自 tracked `instances.csv`，与 `summary.csv` 一致。gap 单位均为百分数；利用率列已从比例转为百分数。

| M | N | 实例数 | 平均 gap % | 中位 gap % | 最差 gap % | gap≤0.1% | 平均利用率 % | 平均构造秒数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 100 | 30 | 0.452063 | 0.422969 | 1.593544 | 2/30 | 98.963509 | 4.869 |
| 5 | 250 | 30 | 0.219406 | 0.201542 | 0.590241 | 6/30 | 99.683526 | 13.887 |
| 5 | 500 | 30 | 0.091935 | 0.087047 | 0.257892 | 22/30 | 99.878851 | 34.715 |
| 10 | 100 | 30 | 0.608209 | 0.513801 | 1.454973 | 2/30 | 98.119656 | 4.542 |
| 10 | 250 | 30 | 0.325096 | 0.260134 | 0.830148 | 3/30 | 99.363288 | 13.051 |
| 10 | 500 | 30 | 0.162849 | 0.143895 | 0.383232 | 9/30 | 99.750313 | 32.678 |
| 全部 | 全部 | 180 | **0.309926** | 0.215324 | 1.593544 | **44/180** | **99.293190** | **17.290** |

按 M 汇总，M5 为 0.254468%，M10 为 0.365385%。这些是同一批实例的嵌套汇总，不能作为额外独立实验次数。

**A：Zero-shot 的可证范围。** manifest 的训练 N 只有 100；N=250/500 共 120 行标记 `size_zero_shot=1`。这些记录支持“保存模型在未列为训练规模的较大 N 上产生了可行解”。由于缺少多种子、不同数据族和匹配 Baseline，尚不能证明其优于其他方法，或对任意规模/分布具备稳定泛化能力。M5/M10 各有独立模型，未验证跨 M 迁移。

**A/C：** 本批较大 N 的平均 gap 较小，但不同 N 对应不同实例和可能不同的相对难度。该趋势不能直接解释为模型随着规模增大而更善于长期规划，也不是同一实例的尺度因果实验。

计时记录包含数据传输和完整构造、CUDA 同步、无 warmup，排除 checkpoint 加载。根状态诊断 LP bound 在计时函数之外，而构造各步骤所需的 LP 计算包含在构造中。180 例累计约 3112.219 秒。没有同硬件、同预算 Baseline，不能据此给出速度优势；详见 [Benchmark_zeroShot.py](../../Benchmark_zeroShot.py) 的 `solve_instance` 与 `benchmark`。

## 6. 不应被总体均值掩盖的负面发现

### 6.1 资源紧张层仍较难

**A：** 按数据集原始 `a` 字段分层，结果如下；这是现有测试集的描述性分层，不是新训练实验。

| 容量比例 a | 实例数 | 平均 gap % |
| --- | ---: | ---: |
| 0.25 | 60 | 0.510413 |
| 0.50 | 60 | 0.259403 |
| 0.75 | 60 | 0.159963 |

其中 M10/N100/a=0.25 的 10 例平均 gap 为 **1.009912%**，M5/N100/a=0.25 为 **0.727581%**。这些分层应优先检查早中期关键决策、可行候选之间的资源竞争及控制器的贡献。不能只报告 M5/N500 组均值低于 0.1%，再推广到全部场景。

### 6.2 两个训练摘要均未确认收敛

**A：** [training_convergence.json](../../results/20261022_195937_430788/training_convergence.json) 的诊断仅覆盖 `fixed_synthetic_greedy_validation_only`：

| M | 最佳验证 update | 最佳验证归一化 return | 最终 update | 尾段跨度 / 验证次数 | 最终状态 |
| --- | ---: | ---: | ---: | --- | --- |
| 5 | 625 | 0.571836448 | 2000 | 350 updates / 15 次 | `not_converged` |
| 10 | 1175 | 0.599457150 | 2000 | 100 updates / 5 次 | `not_converged` |

保存的判据要求尾段全部验证收益不低于历史最佳的 99.95%，跨度至少 1000 updates、至少 20 次验证、累计至少 1000 updates。本批尾段不足以满足判据，`plateau_start_update` 和 `confirmation_update` 均为空。

这不是“训练失败”或“继续训练必有改善”的证明；它明确限制了“已经收敛”的表述。该 0.05% 是相对验证收益的容差，不是 0.05 个 best-known gap 百分点；合成验证收敛也不等于 Chu gap 收敛。

### 6.3 控制量有记录，干预贡献没有被识别

**A：** 逐实例表保存了 θ 和评分项的时间均值。再按实例等权汇总，M5/M10 的 LP primal 权重均值分别约 0.753864/1.422398，LP dual 权重约 0.016522/0.005695。这些值可用于提出模块消融问题，但系数大小受到特征尺度影响，不能直接解释为相应模块的性能贡献。

当前表没有逐状态的固定控制器对照排序、干预触发率、干预前后 argmax、配对后缀收益或分阶段动作保持记录。小的某个权重也不能证明“该机制无用”。关于 **Gated / Selective Intervention**，当前结果没有可辨认的开/关干预实验；动态系数、温度或可行性 mask 本身均不能替代该证据。

## 7. Baseline 与 Ablation 的证据缺口

当前 [Benchmark_zeroShot.py](../../Benchmark_zeroShot.py) 仅报告 `learned_dynamic`，可选 `learned_dynamic_sampled` 也是同一动态策略的多轨迹预算，并非另一 Baseline。README 说明 v6 删除了 `fixed` 和 `learned_static` 报告方法；因此下面的关键研究判断尚未由当前 tracked 结果回答。

| 待判断问题 | 当前证据缺口 | E：建议的匹配验证 |
| --- | --- | --- |
| 高层学习带来多少收益？ | 无固定/未训练高层对照 | 固定同一评分规则、LP 计算和推理预算，比较固定 θ、未训练 Actor、训练 Actor |
| 动态控制是否优于静态控制？ | 无同参数域静态对照 | 控制初始化、训练数据、参数域与预算，分别评估静态和状态依赖 θ |
| LP 特征、评分项、shaping 哪个有效？ | v6 多个组件同时改变 | 分别去除节点/边 LP 特征、两个 LP 评分项、potential shaping，并报告交互效应 |
| 关系编码是否改善约束理解？ | 无独立结构消融 | 保持 Decoder 与训练不变，消融关系特征/attention，检查硬约束层和状态行为 |
| ranking 是否促进 greedy 质量？ | 当前训练原日志与标签统计缺失 | PPO-only 与 PPO+ranking 配对，记录标签接受率、方向、实际 minibatch、动作与收益保持 |
| 结果是否稳定？ | 只有一组 M5/M10 模型；训练 seed 未保存 | 独立训练种子，预先定义预算、验证选择规则、best/last 报告和实例层统计 |
| 性能是否主要来自底层规则？ | 无相同固定底层下的增量对照 | 固定底层评分函数；量化学习前后排序变化及真实终局目标，单列 LP 特征计算成本 |

固定底层指固定评分函数及其参数域，不能把本来每步计算的状态依赖特征全部冻结为常数，否则改变了待比较问题。评价应区分“评分规则固定”与“控制信号随状态变化”。研究方案细节见 [开放研究问题](OPEN_RESEARCH_QUESTIONS.md)。

## 8. 历史材料：保留负面线索，限制归属

**D：** Git 历史和各分支源码可追溯架构探索，但 `git log --all -- results/` 仅找到当前 v6 五个文件的提交。旧 README、提交标题和本机报告提及的历史 gap 不是一组可复算的 tracked 基准序列。不能把它们拼成严格的“版本逐代提升曲线”，也不能用默认配置补齐公平性条件。技术阶段和分叉关系见 [研究演进](RESEARCH_EVOLUTION.md)。

本机 `runs/`、`checkpoints/`、`outputs/` 受 `.gitignore` 排除。调查时确实读到下列材料；它们不随仓库克隆交付，以下路径作为取证线索以代码形式保留，不伪装为可访问的仓库链接：

| 本机线索 | 可保留的研究信息 | 限制 |
| --- | --- | --- |
| `outputs/routing_result_audit_20261021/analysis.json`、`analysis.md` | 报告记录 resource routing 比较并非全面改善，紧容量/M10-N100 层值得关注；指出编码、readout、初始化等同时变化 | 报告引用的旧 `results/20260922_083444_537744/` 与 `results/20261021_064824_670682/` 原始文件在当前工作区缺失，且不在可见 Git 历史中，本次不能重算该比较 |
| `outputs/resource_controller_20261019/saved_run_comparison.json` | 报告记录 readout/controller 方向的负面比较，说明“更复杂表示必然更好”没有得到支持 | 对应历史原始结果缺失；不构成当前 v6 的配对消融 |
| `outputs/learning_signal_audit_20261017/review.md`、`audit_results.json` | 历史审计关注 ranking 标签稀疏、混合续行目标与 greedy 评价不一致、小实例探针覆盖不足 | 对应另一分支/阶段；需要先核对当时源文件与 checkpoint，再判断哪些机制仍适用当前 v6 |
| `outputs/decision_audit_v2_smallstep_retention50_analysis/summary.md`，以及 `outputs/decision_audit_v2_smallstep/retention_50/retention.json` | `nine_actor` 相关本机材料显示短期保持收益不能自动推广到更长过程，后段退化、候选标签滞后和 KL 实际预算值得研究 | 部分原始 JSON 仍在本机，但未纳入 Git；与当前 v6 不同分支。一个来源实例的多个前缀具有相关性，不能当作独立种子或 Chu 完整实例结果 |

这些历史材料有助于制定 **C/E** 假设和验证方案，不足以确认当前 v6 已复现同样问题或已解决它们。若需要把历史量化结论纳入论文或统一基准表，应先归档原始结果、匹配源码与完整配置；不能只归档分析报告。

## 9. 本次复核执行范围

**A：** 本次做了以下不运行模型的离线校验：

1. 使用 Git 文件清单区分 tracked 实验材料与本机补充材料；对 tracked `.mat` 重新计算 SHA256。
2. 从 180 条实例行重算 best-known gap 和 LP gap，公式误差均为 0；重建分组均值，与 summary 的最大浮点差约为 `2.22e-16`。
3. 核对 180 条 CSV 与 180 个 JSONL 解的一一对应关系、索引范围、无重复物品、选择数、解文件 objective 及 checkpoint 摘要字段。
4. 使用 Python 标准库解压并读取 MATLAB v5 结构中的原始 `P/R/B/best` 数值，对每个保存解独立求和：目标利润与 reference 均匹配，最大容量违反为 0，逐资源利用率与结果一致，180 个终局均不存在可再加入的未选物品。
5. 从原始 `a` 字段核对容量层取值及每组数量。标准库解析仅用于这份文件的取证，未替换项目数据加载器。

上述核对确认保存产物内部一致、解可行且 maximal；**maximal 不等于 maximum**，不能证明全局最优。未重新运行 checkpoint、LP 求解、训练、泛化评估或历史版本比较；没有用原始权重验证 manifest 摘要，也没有从缺失训练日志重建曲线。

## 10. 下一批证据的最低维护要求

**E：** 优先补齐原 v6 checkpoint、完整训练配置、原始日志与源码 commit，先使当前锚点能够重放。随后围绕“高层学习相对同一固定底层的增量贡献”建立匹配 Baseline 和 Ablation，再讨论结构扩展。

每批正式实验至少归档：数据 SHA/实例 ID 与参考类型、源码 commit 与未提交差异、训练和验证种子、完整训练/生成器配置、实际更新与 minibatch 预算、checkpoint SHA、best/last 选择规则、推理预算/硬件/计时边界、逐实例结果、解序列、训练诊断及失败状态。较大的权重可存于受维护的外部制品位置，但仓库应保存校验摘要与获取方式。

研究结论以独立实例和独立训练种子为统计单位。冻结旧策略的有限候选收益、可达 θ witness、接口测试、单个最佳 checkpoint 或相关前缀上的改善各有用途，应分别标记，不能升级为多种子训练有效性、精确 regret 或全局最优证明。
