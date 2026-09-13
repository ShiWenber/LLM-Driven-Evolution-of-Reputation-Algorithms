# 为什么当前框架难以演化出胜过 Leading Eight 的策略

## 摘要

本报告只讨论一个目标：通过演化获得在预先固定条件下，比 canonical Leading Eight 更能合作且更能抵御入侵的策略。对当前代码和存档结果的审计显示，最主要的障碍不是模型不会写合作代码，而是**训练群体几乎总在全合作状态，缺少识别和排除缺陷者的选择压力；采纳时又用 LLM 重写代替忠实遗传；最终测试则测另一个、长时间尺度的博弈**。2026-09-11 的两组各五个种子的 `agent-type1` 存档，初代合作率均为 0.990—1.000，但这不能预言稀有入侵成功。建议先修复实验协议，再采用“可继承复制 + 稀有攻击者训练 + 多保真稳态筛选”的路线；不建议仅换更强的 LLM 或再提高平均合作率。以下区分代码已确认、存档已观察和仍须实验检验的因果解释。

## 1. 研究问题与方法

RQ1：当前演化过程实际对哪些行为施加选择，是否充分测试了经典规范赖以稳定的条件分支？RQ2：代码、训练/评测分布和统计推断中，哪些环节会阻碍可靠地发现更强策略？RQ3：文献中哪些机制提示可实施的改进，什么结果才足以称为“更好”？

检索从经典规范、私有信息与噪声、全局策略空间演化、LLM 文化演化四个角度展开；参考工作均在作者/出版方/arXiv 原页面核对。仓库审计覆盖 `prompts.py`、`evolution_architecture.py`、`population.py`、`game.py`、canonical baselines、入侵/fixation runner 和已存实验 JSON。没有重跑需要 API 或高计算量的实验。静态代码说明**现在**的实现，存档 JSON 说明**当时**的运行；两者未附完整代码提交哈希时不能默认完全一致。

## 2. “高合作”与“更强规范”的区别

Ohtsuki 与 Iwasa 的 Leading Eight 在二值公开评价的经典模型里满足维持合作、识别背叛、认可正当拒绝、犯错后恢复等约束；[Fujimoto 与 Ohtsuki（2024）](https://journals.aps.org/prxlife/abstract/10.1103/PRXLife.2.023009)进一步在私有评价下区分哪些规范支点保障高合作、哪些保障抵抗 ALLC/ALLD 的入侵。[Schmid 等（2023）](https://www.nature.com/articles/s41467-023-37817-x)说明量化声誉可以纠错，却只令一部分规范在其噪声设置下尤其稳健。换言之，连续声誉是表示手段，不会自动提供“拒绝坏人仍被认为是好人”的逻辑。[Ohtsuki 与 Iwasa（2006）](https://www.sciencedirect.com/science/article/pii/S0022519305003474)。

文献也提醒不要假设只需搜索更大就能找到处处压倒性的策略。[Murase 与 Hilbe（2024）](https://www.pnas.org/doi/10.1073/pnas.2406885121)在数千种二、三阶规范的全局演化计算中发现，well-mixed 群体里合作比只对照少数规范时脆弱得多，而群体结构改变了结论；[Hilbe 等（2018）](https://www.pnas.org/doi/10.1073/pnas.1810565115)和[Michel-Mata 等（2024）](https://www.nature.com/articles/s41586-024-07977-x)分别显示私有噪声会使声誉分歧累积，而聚合多次观察与适度宽容可以缓解它。因此“更好”必须明确**信息条件、种群结构、突变对手、成本与收益比**，不能只看纯种群的合作率。

LLM 近邻工作呈现相同张力：[Vallinder 与 Hughes（2024/2025）](https://arxiv.org/abs/2412.10270)发现不同基座模型的文化演化合作结果显著不同；[Horibe 等（2026，预印本）](https://arxiv.org/abs/2608.04507)报告，LLM 代理的抗搭便车能力更取决于排除缺陷者的强度，合作可以涌现却不一定形成经典高阶规范。这支持调查“模型的合作先验有没有掩盖策略判别”，但并不直接证明本仓库的某个特定失败原因。2025 年的[Glynatsi、Hilbe 与 Murase](https://arxiv.org/abs/2509.08006)则在**公开评价**条件下给出更广的稳定性刻画；它是规范性质的对照，不应不加条件直接套到本仓库的私有声誉和同时行动 PD。

## 3. 代码与存档的六个瓶颈

### 3.1 合作地板太高，关键状态几乎不进入训练

在两组 `N=16、1000 interactions/generation、updates=4` 的最近 `agent-type1` 存档中，`b/c=3/1` 与 `2/1` 各五个种子的初代合作率均在 0.990—1.000；例如 `b=3, seed=0` 的第 0 代 16 人的记录收益全是 50，第 99 代合作率 0.997、收益仍全是 50，且两种 LLM fallback 计数为 0。[存档示例](../results/quantitative_baseline/LLM_agent-type1_fermi_z_v3_g100_1000inter_N16_genreset_upd4_5seed_seed0/evolutionary.json)。这证明这些运行的初始化没有“不会合作”的困难；**它们不证明模型具有抵抗稀有 ALLD 或其他规范的能力**。

代码中的初始 prompt 要求设计“reputation-based strategy”，目标是个人收益；同质群体近乎全 C、演化端没有动作/观察翻转时，向坏对象合作、处罚坏人、误罚纠正等分支缺乏出现机会。`N=16` 的 1000 次同代随机配对使每个无序对平均相遇约 `1000 / C(16,2)=8.3` 次；`N=100`、10000 次入侵评测时约为 `10000 / C(100,2)=2.0` 次。若策略借助身份与重复对局记忆而不是第三方信誉，规模变大也会改变它的有效机制。[prompt](../experiments/v2_quantitative/prompts.py)；[交互实现](../experiments/v2_quantitative/game.py)。

**待检验的因果预测：**向训练群体注入预先固定比例的 ALLD、ALLC 和少量条件背叛者，若“缺陷者排除率”明显提高、未见种子的 `ρ(candidate→ALLD)` 也提高，则当前确有关键状态暴露不足；若没有，则应继续查策略表示或奖励定义。

### 3.2 模仿并不保真：策略难以靠成功复制积累

`FermiEvolutionRule.plan` 在一次采纳后，以 `μ=0.1` 调独立 LLM 初始化；其余 `0.9` 调亲代条件化 LLM 改写，**不存在原样复制的常规路径**。即使把 `μ` 设为 0，也仍是每次采纳重写。若所有人窗口收益相同，`β=5` 的 Fermi 接受概率为 0.5；在 `updates=4` 的最近实验中，这意味着中性条件下每代约两次被接受的重生成机会，而不是有优势基因型的忠实扩张。这是从代码推导的期望，不是实测子代质量。[计划代码](../experiments/v2_quantitative/evolution_architecture.py)；[人口管理器](../experiments/v2_quantitative/population.py)。

该设计既有搜索价值又有演化代价：新代码不断出现，但因“复制”和“变异”绑定，不能将适应度优势可靠归因于亲代策略，也无法用 `μ=0` 复现实验理论里的无突变选择。最先做的消融应是 **原样复制概率 `1−μ`、明确的低概率突变 `μ`**，并与当前“每次采纳都重写”在同等 LLM 调用预算下比较：亲子行为距离、成功谱系寿命、跨种子最佳候选的保留率，以及最终入侵表现。若忠实复制明显改进，瓶颈就在遗传保真；若没有，再考虑搜索方向而非继续加大调用量。

### 3.3 当代短窗口收益与 fixation 目标不是同一个量

当前演化以每代最后一部分交互的**个人累计 payoff**选 role model，默认 `fitness_window_interactions=200/1000`；而 fixation 对每个组成 `k=1…N−1` 先让信誉系统 burn-in，再估计两策略的稳态**每次参与的平均 payoff**，并将整条 `d(k)=π_M(k)−π_R(k)` 曲线用于 `ρ`。前者强烈依赖当代对手组成及随机互动，后者关心稀有出现、频率变化与长期声誉稳态。[演化评价](../experiments/v2_quantitative/evolution_architecture.py)；[fixation 定义](fixation_benchmark.md)。

仓库已经给出具体反例：`readme_best` 在有限代模仿扫描中从 10% 对 L1 增长到约 82%，而旧 fixation 估计对 L1 的 `ρ` 近零；`n50_seed2` 对 L1—L8 的旧 `ρ` 约 0.023、与 `1/50=0.020` 接近，却对 ALLD 仅约 0.0054。[已有结果](fixation_benchmark.md)。这些数是**旧预算下的诊断性结果**：文档同页有若干 `k` 的非平稳警告，不应把小数位当成确认结论。真正要看的是在哪段 `k` 上 `d(k)` 翻负，以及原因是识别慢、惩罚不正当、声誉分裂，还是 ALLC 抢占。

### 3.4 训练和测试换了博弈条件

已核查的存档包含 `b/c=3/1` 和 `2/1` 两类；fixation runner 的支付写死为 `b/c=2/1`。训练默认全观察、无动作/观察错误、每代重置代理和声誉；旧有限代入侵可设 1% 双错误，也每代重置；fixation 则在固定组成中持续演化声誉，按稳态取样，且采用与当前演化代码不完全相同的配对调度。`N=16` 训练与 `N=50` fixation、`N=100` 入侵也是明显的生态条件转移。这些差异**并非 bug 本身**，但若目标是“在 fixation benchmark 胜出”，训练中必须有与其相近的选择压力，并且对照实验要一项项改变条件，不能混称同分布。[运行入口](../experiments/run_fermi_v3.py)；[入侵 runner](../experiments/analysis/invasion/run_invasion.py)；[fixation runner](../experiments/analysis/invasion/run_fixation_benchmark.py)。

此外，当前 canonical 表将原文献的**顺序捐赠博弈**规范移植到双人**同时行动 PD**；[基线源码](../experiments/v2_quantitative/baselines.py)自己也注明不是精确复制。这意味着比较对象应称为“本框架移植版 L1—L8”，不能说与原论文完全相同。评估必须使用当前 `L1…L8`，而非存档图或旧文件里出现的 `IS/SS/SJ/...+` 组合。

### 3.5 小收益差也可能被高选择强度当成确定优势

最近记录使用 `β=5`，故一次 payoff 差为 1 时模仿概率 `σ(5)≈0.993`；当收益来自相对短的随机窗口，这可能放大采样噪声。若群体全合作且窗口收益完全相同，反而退化为 `σ(0)=0.5` 的中性抽样。两个区间都不利于发现罕见但真正稳健的规范：一边是随机漂移，一边是偶然表现被近乎确定地固定。需记录每代 payoff 方差、同一候选重评收益的方差、亲子代码行为差距，并用相同随机日程或多次匹配估计**期望 payoff**；若改后胜出策略更可重复，才能说过去主要受选择噪声影响。上述是机制推断，尚待配对消融。

### 3.6 确认实验存在缓存与收敛风险

`run_fixation_benchmark.py::cache_matches` 比较 N、burn-in、measure、β、误差率、候选哈希等，**不比较 `--seed`**；同输出目录只换 seed 而不加 `--force` 时会复用旧 `fixation_benchmark.json`。这会令“换新确认种子”表面执行、实际未重算。[缓存实现](../experiments/analysis/invasion/run_fixation_benchmark.py)。此外，已有结果的极端组成出现半窗口 payoff 差漂移，五复现的 ±1 标准差敏感性范围不是置信区间；固定的绝对 `0.01` 判词容差在 `N=50` 的中性值 `0.02` 附近也太粗，不能用“neutral”输出证明等效。[诊断与限制](fixation_benchmark.md)。这是**测量可靠性风险**，不是“策略演化失败”的直接原因，但会误导你判断什么已经成功。

## 4. 怎样改，才能给“更强策略”真实出现的机会

第一步是**修协议而不修论文结果**。冻结 `N、b/c、观察概率、两种错误率、声誉重置、每代交互数、配对机制、fitness 归一化、选择 β、候选代码和模型服务版本`；结果记录实际代码 commit 哈希与响应模型版本。把 `--seed` 纳入 fixation 缓存键，旧缓存重新核查；canonical L1—L8 与 ALLC/ALLD 同环境复测，剔除不平稳的 `ρ`。这一步的任务是让胜负可信，而不是预设胜出。

第二步是让演化**遇见入侵者**，但仍保持演化实验的因果解释。每代训练群体使用可复现的 mixed ecology：同质群体保持合作；另一些 episode 放 1%、5%、10%、50% 的固定对手（ALLD、ALLC、若干 L1—L8、易伪装的条件背叛者）；记录每个候选的自己收益、合作率、拒绝真正缺陷者率、错误惩罚率及正当拒绝后的声誉回报。种群选择仍可由实际 payoff 驱动；额外的对抗评分只用于**选择训练环境或精英候选**，否则研究性质会从“演化”变成“人工多目标优化”，应诚实命名。训练对手与最终测试对手、种子、误差条件必须拆开，避免把 benchmark 变成训练集。

第三步是分离复制与变异，建立多保真筛选。先以便宜的短程压力测试淘汰明显 ALLC 式/ALLD 式/伪信誉策略；对存活者绘少量组成的 `d(k)` 近似曲线；只对前几名运行完整 N=50、多复现、收敛检查的 fixation。保留 Pareto 集而非单一高合作冠军：合作、抵抗缺陷者、对 ALLC 的脆弱性与信息成本可能互相冲突。[Murase 与 Hilbe（2024）](https://www.pnas.org/doi/10.1073/pnas.2406885121)提示仅对少数突变类型获胜不等于全局占优；[Michel-Mata 等（2024）](https://www.nature.com/articles/s41586-024-07977-x)提示多观察与适度容忍值得作为可解释变异算子，但并不保证在本模型中胜出。

第四步才是决定论文判据。预注册主要测试为：在同一固定 `(N,b/c,p,error)` 下，一个冻结候选对每个 L1—L8 的**正向** `ρ(candidate→L_i)` 是否稳定超过 `1/N`，以及**反向** `ρ(L_i→candidate)` 是否低于 `1/N`；另报 ALLC/ALLD、纯种群合作率、信誉一致率和计算开销。没有把所有条件都胜过经典规范的理论保证；若只在私有且噪声信息下领先，应明确写“该条件下更好”。如果当前纯群体已接近 100% 合作，合作率本身没有可提升空间，真正可声称的进步只能来自抗入侵、纠错、成本或外推。

## 5. 最小决定性实验（优先做这三个）

| 实验 | 对照与控制 | 若支持瓶颈假说 | 若不支持，下一步 |
| --- | --- | --- | --- |
| **亲代保真消融** | 现有每次采纳重写 vs 原样复制 + 稀少变异；相同种子、收益、总 LLM 预算 | 胜出策略存活更久，未见种子的稳态表现更好：遗传保真是瓶颈 | 转查状态暴露与适应度定义 |
| **混合群体课程** | 全合作自演化 vs 预先冻结稀有 ALLD/ALLC/条件背叛者课程；相同生成预算 | 对缺陷者的识别、正当拒绝和抗反入侵改善：训练生态缺关键状态 | 分析类型1表示或理论上不可占优的条件 |
| **同环境桥接** | 同一冻结策略，在同 N、b/c、噪声和信息下同时跑有限代入侵与稳态 `d(k),ρ` | 差距明显缩小：过去主要是协议/时间尺度失配 | 若差距仍大，属内在频率依赖或多稳态，需改搜索目标 |

最后一个重要的负面对照是随机标签/随机评判、ALLC、ALLD 和“直接记住对手 ID 但不用第三方评价”。若 `agent-type2` 的获益被仅用对手 ID 的策略复现，就不能把它命名为新的**社会规范**；它可能只是小群体中的直接互惠。[Hilbe 等（2018）](https://www.pnas.org/doi/10.1073/pnas.1810565115)与[Schmid 等（2023）](https://www.nature.com/articles/s41467-023-37817-x)所研究的核心是他人行为如何经社会评价影响未来第三方合作，必须在实验中把这一机制单独识别。

## 6. 结论

RQ1：当前训练几乎从全合作出发，且没有显式噪声与少量固定缺陷者，选择压力很难看到 Leading Eight 的关键条件分支；最近存档的初代 0.990—1.000 和全同收益是强烈但非单独充分的证据。RQ2：每次采纳都 LLM 重写、短期窗口收益与长期 fixation 不一致、生态条件不匹配，是前三个应解决的框架问题；fixation 缓存漏 seed 与非平稳结果则影响验证。RQ3：文献提示稳健规范需要情境化正当惩罚、错误纠正以及在目标信息结构下验证；“更强策略”应在固定条件与未见对手上双向检验，而不是用更高初始合作率替代。

## 参考文献

1. Ohtsuki, H., Iwasa, Y., “The leading eight: Social norms that can maintain cooperation by indirect reciprocity,” *Journal of Theoretical Biology*, 2006.
2. Hilbe, C., Schmid, L., Tkadlec, J., et al., “Indirect reciprocity with private, noisy, and incomplete information,” *PNAS*, 2018.
3. Schmid, L., Ekbatani, F., Hilbe, C., Chatterjee, K., “Quantitative assessment can stabilize indirect reciprocity under imperfect information,” *Nature Communications*, 2023.
4. Fujimoto, Y., Ohtsuki, H., “Who is a Leader in the Leading Eight? Indirect Reciprocity under Private Assessment,” *PRX Life*, 2024.
5. Murase, Y., Hilbe, C., “Computational evolution of social norms in well-mixed and group-structured populations,” *PNAS*, 2024.
6. Michel-Mata, S., Kawakatsu, M., Sartini, J., et al., “The evolution of private reputations in information-abundant landscapes,” *Nature*, 2024.
7. Vallinder, A., Hughes, E., “Cultural Evolution of Cooperation among LLM Agents,” arXiv:2412.10270, 2024; AAMAS 2025 extended abstract.
8. Glynatsi, N. E., Hilbe, C., Murase, Y., “Exact conditions for evolutionary stability in indirect reciprocity under noise,” arXiv:2509.08006, 2025.
9. Horibe, K., Itao, K., Toyokawa, W., “Emergence of Reputation-Based Cooperation in LLM Agents,” arXiv:2608.04507, 2026（预印本）.

**局限：**文献比较涉及不同捐赠/PD 博弈和信息制度，无法移植其性能数字；目前没有新训练或复验，以上改造是待验证的预测。存档与当前代码未核对完整 commit 来源，尤其配对调度的旧记录不应仅依照当前实现重解释。
