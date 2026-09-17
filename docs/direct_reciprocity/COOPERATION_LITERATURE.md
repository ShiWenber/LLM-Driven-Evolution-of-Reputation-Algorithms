# Cooperation / IPD 原文核实笔记

检索日期：2026-09-16。问题：提示词能否改变合作表现；演化选择的对象是什么；哪些测量不足以支持普遍合作优势。使用 deep-research 技能，以论文原文及作者代码仓库交叉核实；未使用 Zotero，未修改实验代码。

## 1. Willis 等：需要区分摘要与完整论文

- [AAMAS 2025 Extended Abstract](https://ifaamas.csc.liv.ac.uk/Proceedings/aamas2025/pdfs/p2786.pdf)：题名 **Will Systems of LLM Agents Lead to Cooperation: An Investigation into a Social Dilemma**；Richard Willis、Yali Du、Joel Z. Leibo；pp. 2786–2788。
- [完整论文 arXiv:2501.16173v1](https://arxiv.org/html/2501.16173v1)：题名 **Will Systems of LLM Agents Cooperate: An Investigation into a Social Dilemma**；增加作者 Michael Luck；2025-01-27。完整论文确实包含 Moran 演化，不能据三页摘要缺少演化结果判定该工作没有演化。

摘要 §2–3：GPT-4o 先写自然语言策略，再转 Python；每个 attitude（Aggressive、Cooperative、Neutral）每种提示生成25个策略；1000轮 IPD，(T,R,P,S)=(5,3,1,0)，75策略循环赛重复20次。Default 直接描述游戏；Refine 增加自我批评和重写；Prose 从四种同构情景抽样，先写情景策略再转 IPD。Table 3 中，Default 的 A–A 收益1.81，而 C–A为1.55；Refine 分别2.20、2.53，说明改变提示可以改变对攻击型对手的最佳响应。11个人写算法对照赛重复200个种子。

完整论文 §3.1–3.4、§4.3：比较 GPT-4o 与 Claude 3.5 Sonnet；所有策略统一由 GPT-4o 编码。可选10%独立动作翻转噪声。选择单位是 attitude-strategy set：每场从所属25策略均匀重采样；适应度为全对全累计收益；按适应度比例繁殖、均匀随机死亡；无突变，直到 attitude 同质。N=12；初始4:4:4或8:2:2；每条件100次。

Table 6：GPT-4o、无噪声、平衡初始的攻击型吸收率为 Default 14%、Refine 19%、Prose 13%；攻击多数初始下 Refine 49%、Prose 35%。原表 Default 多数初始一行66%+19%+17%=102%，存在内部不一致，勿照抄全行。§5 将对齐导致合作偏置作为假说，非因果检验。

**评价判断**：这是固定策略库的 attitude 层选择，不是在线学习、具体算法的持续优化，也不是一般策略空间的入侵稳健性证明。多类型既有群体的吸收率也不等于单突变体 fixation probability。作者对特定 attitude 收益矩阵的 ESS 判断应保留其策略集合范围；不能推广为面对任意未见策略的 ESS。

## 2. 作者仓库揭示的复现限制

[evollm README](https://github.com/willis-richard/evollm)，Prompts / Strategies：仓库保存自然语言及算法，Prose 还保留情景策略和游戏策略。但明确声明 **GPT-4o + Refine + noise 策略已意外遗失**，所以相关条件无法仅凭现存原始策略完整重现。README 的温度0.7是运行示例，不能仅据示例确认所有发表实验的实际温度。论文 §3.2 还报告人工修复可解释错误，否则删除并重生成；这使结果包含生成后筛选环节。

## 3. 直接互惠中最有价值的反证：测量本身能改变结果

[Nicer than Humans: How Do Large Language Models Behave in the Prisoner’s Dilemma? — 原文](https://ojs.aaai.org/index.php/ICWSM/article/download/35829/37983/39897)，§3、§4.2、§5：Llama2、Llama3、GPT-3.5；每局100轮、每条件100局，面对不同合作概率的随机对手。Llama2 对 Always Defect，完整历史条件在约50轮后重新趋向全合作，窗口10却持续背叛；因此高合作率可能反映长上下文处理失效。加入累计得分显著改善理解测试（§4.1、Fig. A7）；CoT 未改善。局限是固定收益、随机对手、单个 LLM，且提示修订依赖研究者经验。该工作没有群体演化，不能把轨迹稳定状态称为演化稳定性。

## 4. Persona 确实影响合作，但不能直接等同内在偏好

[Phelps & Russell, The Machine Psychology of Cooperation, arXiv:2305.07970v2](https://arxiv.org/html/2305.07970v2)，§2、Table 5、§4：三版 GPT-3.5（0301、0613、1106），温度0.1/0.6；6轮 IPD，(T,R,P,S)=(7,5,3,0)；450种提示组合，五种 persona、每类三种措辞，另随机变化标签、大小写、顺序、代词和 CoT。对手是 AllC、AllD、合作开局及背叛开局的 TFT。汇总平均合作率：Cooperative .62、Competitive .34、Altruistic .67、Selfish .41。无演化选择。结论支持提示可操作化部分行为差异，不支持稳定内在偏好。正文复现次数有笔误：§2.3 写R=4却列t=0,1,2，§2.1及§2.4支持R=3，引用时应注明。

## 5. 邻近 framing 证据：明确不是 IPD

[Lorè & Heydari, Strategic Behavior of Large Language Models: Game Structure vs. Contextual Framing](https://arxiv.org/pdf/2309.05898)，§2–3、§5：GPT-3.5、GPT-4、LLaMA-2；四种2×2游戏、五种情境，每模型每组合300次。友谊、商业、外交等背景改变合作选择，GPT-3.5尤其受情境支配。它研究一次性选择，作者将 repeated games 列为未来方向；可用于说明 framing 混淆，不能用作直接互惠或演化结果。

## 6. 边界与补充线索

[Pal 等, Large language models instantiate evolutionarily robust strategies of cooperation, PNAS Nexus 5(6), pgag210](https://academic.oup.com/pnasnexus/article/5/6/pgag210/8705778)，2026-06-11。Box 1 / Fig. 1：将动作匿名为L/R，向五个模型询问初始状态及四种上一轮结果，每状态50次，推断 memory-one 策略；再用博弈论性质分析，而非只看已实现轨迹。需区分“对推断出的有限记忆策略成立的性质”与“LLM在任意长历史和提示下的行为保证”。此篇本次仅核实这一方法定位，未全面复核全部稳健性条件和补充材料，勿据此转述全部定理。

综合判断：已有工作同时提供合作倾向及其条件性证据。适当的主张应绑定模型、提示、策略表示、对手分布和选择过程；优先报告收益、合作率与入侵/吸收指标的区别，并将措辞变体、历史窗口及生成后筛选纳入测量审计。
