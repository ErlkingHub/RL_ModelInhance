# 证据索引与维护约定

> 核查日期 2026-10-23；代码与结果固定基线 `a933ff7a15aa1be84ec7998a15c1daa0bd21e797`，分支 `40_lp-dual-bipartite-bound-shaping`。文档来自仓库核查，可脱离 AI 对话独立使用。

## 分类与证据优先级

| 标记 | 定义 | 使用要求 |
| --- | --- | --- |
| A | 已确认事实 | 给出源码、版本或可复核原始结果；说明核实范围 |
| B | 已实现但未充分验证的机制 | 实现可确认，效果仍需消融和重复实验 |
| C | 研究假设 | 明确预测和可区分的验证方法，不写成结论 |
| D | 历史探索方向 | 明确所在提交/分支，不能默认为当前实现或已放弃 |
| E | 后续研究计划 | 不描述为已经运行的实验或创新 |

优先使用固定提交的源码与原始结果，其次使用有来源的实验摘要。README 和提交消息可证明曾经记录了某项设计或数值，不能独自证明性能。被忽略的本机报告不构成远端读者可复算的原始证据。A 类实现事实与 B 类效果状态可以同时适用于同一机制。

## 当前证据目录

| ID | 证据与入口 | 可支持的陈述 | 主要局限 |
| --- | --- | --- | --- |
| S01 | [mkp_config.py](../../mkp_config.py)、[mkp_Model.py](../../mkp_Model.py) | schema/版本、默认配置、θ 与 Actor/Critic 接口 | 默认不等于存档训练实际配置 |
| S02 | [mkp_env.py](../../mkp_env.py)、[mkp_data.py](../../mkp_data.py)、[read_mat.py](../../read_mat.py) | 硬可行性、状态、合成生成器和 MAT 适配 | 生成器是 Chu-inspired，并非原始分布复刻证明 |
| S03 | [mkp_attention.py](../../mkp_attention.py)、[mkp_basis.py](../../mkp_basis.py) | 候选关系、选后特征、组合基函数 | 人工样例只证明局部接口/表达能力 |
| S04 | [mkp_lp.py](../../mkp_lp.py) | LP 对偶近似、上界与缓存 | primal 近似未保证可行；不是整数最优证书 |
| S05 | [mkp_Trainer.py](../../mkp_Trainer.py)、[mkp_ranking.py](../../mkp_ranking.py) | PPO、整形、排序监督、checkpoint 选择 | 当前存档没有完整来源训练配置/日志 |
| S06 | [Benchmark_zeroShot.py](../../Benchmark_zeroShot.py)、[mkp_utils.py](../../mkp_utils.py) | greedy/sample 区别、signed gap、解校验、冻结模型加载 | 不能重建缺失权重或其训练过程 |
| S07 | [mkp_diagnostics.py](../../mkp_diagnostics.py) | 合成验证平台诊断的操作性定义 | `not_converged`/平台均不是 Chu 收敛证明 |
| S08 | [tests](../../tests/)、[mkp_performance.py](../../mkp_performance.py) | 检查范围、缓存和分块执行设计 | 测试存在不代表本次执行，局部计时不等于训练提速 |
| R01 | [manifest.json](../../results/20261022_195937_430788/manifest.json) | 数据 SHA、评估参数、模型标识/配置、被测 checkpoint SHA/update | 路径为来源 Linux 环境；没有归档权重及完整训练配置 |
| R02 | [instances.csv](../../results/20261022_195937_430788/instances.csv)、[summary.csv](../../results/20261022_195937_430788/summary.csv) | 180 实例及分组指标、时间和控制量摘要 | 单一结果包，无 fixed/static 或因子消融 |
| R03 | [solutions.jsonl](../../results/20261022_195937_430788/solutions.jsonl)、[Benchmarks_Chu.mat](../../MKP_dataset/Benchmarks_Chu.mat) | 保存解的利润、容量、序列与终局 maximal 性可独立核算 | 解可校验不等于模型可重跑，也不证明最优 |
| R04 | [training_convergence.json](../../results/20261022_195937_430788/training_convergence.json) | 两个训练来源的诊断摘要与被测 best 关联 | 缺每步曲线，不能定位全部训练动态 |
| H01 | [Git 固定提交与分叉索引](RESEARCH_EVOLUTION.md) | 当前祖先和其他探索实现、历史报告 | 无原始历史结果的 gap 不可复算、不可作因果比较 |

固定源快照：[a933ff7](https://github.com/ErlkingHub/RL_ModelInhance/tree/a933ff7a15aa1be84ec7998a15c1daa0bd21e797)。相对链接服务于本分支阅读；需要跨分支引用时使用演进文档中的固定 commit 链接，不依赖会移动的分支头。

## 核查覆盖与缺失信息

本次检查了根 README、全部受跟踪 Python 模块与测试、当前受跟踪实验包、数据目录、忽略规则、本机残留研究输出以及所有可见 Git 分支/历史。初始工作区无修改，没有已有 `AGENTS.md` 或 `CLAUDE.md`。本机历史结果目录多为空，已有输出摘要不能替代缺失原始结果；nine_actor 的部分 learning/retention 原始 JSON 仍在本机，但受忽略且未入 Git。不将其中其他分支的性能记入当前 v6。

通过 GitHub 连接器在线核查远端 `ErlkingHub/RL_ModelInhance`、默认分支和分支头；当前基线与远端一致。2026-10-23 的 [Issues 查询](https://github.com/ErlkingHub/RL_ModelInhance/issues?q=is%3Aissue) 与 [全部 PR 查询](https://github.com/ErlkingHub/RL_ModelInhance/pulls?q=is%3Apr) 返回空，未发现可补足历史设计决策的条目。此查询是本次时点状态，不保证未来不变。

本次仅进行了资料核查、保存结果的重算和文档验证，没有训练模型、生成新模型推理结果、运行正式基准或修改科学实现。当前原始结果的复核范围、误差和历史信息缺口以 [实验发现](EXPERIMENTAL_FINDINGS.md) 为准。README 中历史测试数、LP 精度和速度数字保留其原阶段来源属性，不作为本次新复测。

2026-10-23 文档审查覆盖 9 份新增/更新 Markdown：155 个本地链接及源文件行号边界、45 个固定 Git 提交/文件引用均通过检查；2 个 Mermaid flowchart 通过 Mermaid 11 实际解析器；展示数学公式采用 GitHub 支持的 `$$` 定界，代码围栏配对检查通过。交叉审查进一步核对了分支归属、百分数/百分点、训练与推理边界以及历史原件是否仍在本机。算法回归测试未重跑，因为本次没有修改实现；结果复核只读取既有产物。

## 新增结论时的最小记录

每个实验记录应包含研究问题与证据分类、源 commit、controller/decoder/schema、完整实际配置、训练/验证/评估种子、数据 SHA 与实例 ID、源/结果 checkpoint SHA、训练与实际优化预算、模型选择规则、硬件和时间口径、逐实例结果及解序列。对照额外记录共同不变量和无法匹配的差异。

维护步骤：

1. 将原始小型报告和 manifest 纳入版本管理；大权重放在可访问的归档位置，记录不可变地址和 SHA。不得将来源绝对路径当作读者可用下载地址。
2. 在 [实验发现](EXPERIMENTAL_FINDINGS.md) 新增实验行，标明重复数、失败与局限，不覆盖旧数据。
3. 若改变科学实现，更新 [算法设计](ALGORITHM_DESIGN.md) 和 [研究演进](RESEARCH_EVOLUTION.md)，明确对基线的影响。
4. 根据证据调整 [开放问题](OPEN_RESEARCH_QUESTIONS.md) 的状态，再更新 [总览](RESEARCH_OVERVIEW.md)。
5. 检查相对链接、Mermaid、术语、百分数与百分点、独立实例数及训练/评估边界；在文档提交或 PR 中记录实际验证结果。

推荐实验条目模板：

```text
实验 ID / 日期 / 问题 / 分类：
代码 commit / controller / decoder / schema：
数据 SHA / 实例 ID / 训练与测试边界：
完整配置 / seeds / 预算 / checkpoint SHA：
对照与共同不变量 / 模型选择规则：
逐实例结果 / 解序列 / manifest 路径：
结论 / 负面结果 / 不能推出什么：
```

仓库入口为 [研究总览](RESEARCH_OVERVIEW.md)。模型无关的维护原则同时供研究人员、Codex、Claude 和其他 Agent 使用。
