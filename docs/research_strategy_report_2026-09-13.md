# 从高合作率到抗入侵：LLM 演化社会规范的瓶颈与两条研究路线

## 摘要

截至 2026-09-13，本项目遇到的核心矛盾不是“LLM 不会合作”，而是**高合作率与演化稳定性之间的目标错位**。DeepSeek 官方确认，兼容模型名 `deepseek-v4-flash` 已临时指向 V4.1-Flash；但用户观察到的初代约 80%（旧版约 40%）尚需在同环境、同提示和可核查服务版本下复验。现有演化以当代自身收益筛选，最终检验却用对整个频率轴的稳态收益和 fixation probability；更强的预训练合作倾向并不自然形成有效识别、正当惩罚和宽恕机制。检视仓库发现，演化与部分 benchmark 的参数和语义尚需对齐，且已有 fixation 结果包含非平稳警告。建议先做协议审计与失效机制定位，再选一条主线：将攻击者纳入演化搜索以探索特定信息条件下的稳定规范；或转向**带可验证交互证据的内生通信协议**，明确将其定位为新的设置，而非未经检验的“普遍超过 leading eight”。

## 1. 问题与调查范围

- **RQ1：**为何模型版本升级和初始合作率上升，不带来对 leading eight 的稳态入侵优势？
- **RQ2：**如何修改搜索目标和验证协议，才能可靠地区分策略机制改善与评估错位？
- **RQ3：**由 `agent-type2` 自行形成通信标签是否有独立的研究价值，需要哪些最小机制与反事实？

方法：核对仓库截至本日的 README、规范实现、演化 prompt/选择与入侵/fixation 代码；检索个人 Zotero 文献（Ohtsuki–Iwasa、Hilbe、Schmid、Michel-Mata、Pires、Song 等），并以作者/期刊/arXiv 原页面核实 2024—2026 近邻研究。按**规范能否条件合作**、**信息与评估协议**、**表示/通信是否内生**三个互补层次组织，而不按论文逐篇罗列。此为设计与证据审计，**没有重新运行昂贵的演化或 fixation 实验**；未见原始结果的地方不报新数值。

## 2. 文献给出的边界：不是所有“合作”都能挡住搭便车者

经典 leading eight 不是单纯偏爱合作，而是把“合作、惩罚坏人、惩罚是否正当、错误后修复”组成联合规则。Ohtsuki 与 Iwasa 指出其共同性质是友善、报复、道歉与宽恕；Hilbe 等则发现私有、噪声和不完全信息会使多数规则失去稳定性。两者不矛盾：前者的稳定结论有其观察与更新条件，后者改变了这些条件。[Ohtsuki & Iwasa, 2006](https://www.sciencedirect.com/science/article/pii/S0022519305003474)；[Hilbe et al., 2018](https://www.pnas.org/doi/10.1073/pnas.1810565115)。

Schmid 等显示量化声誉更新在私有、错误信息下可起纠错作用，但并不是任意连续分值都稳定；他们识别出条件下更稳健的部分规则。Michel-Mata 等的多次观察、一定容忍和私有共识则提供另一条纠错路径：抗噪性不只由一次动作判定决定。[Schmid et al., 2023](https://www.nature.com/articles/s41467-023-37817-x)；[Michel-Mata et al., 2024](https://www.nature.com/articles/s41586-024-07977-x)。

最近的 LLM 证据提醒不要把“更会合作”等同于“更懂规范”：Pires 等发现模型对与低声誉对象合作的评判分歧明显；Horibe 等 2026 年预印本在不同 LLM 后端的演化实验中，把抗搭便车能力与对坏对象的区分/排除联系起来，而非与 leading-eight L1 的表面吻合程度简单对应；Zhu 等的 ALIGN 则提出开放语言 gossip 可辅助去中心化代理识别恶意进入者。这三项研究任务、收益和攻击模型不相同，不能直接横比百分比，却共同指向**判别与信息机制**而不是总体合作率。[Pires et al., 2025](https://arxiv.org/abs/2507.00088)；[Horibe et al., 2026，预印本](https://arxiv.org/abs/2608.04507)；[Zhu et al., 2026](https://arxiv.org/abs/2602.07777)。

| 工作类别 | 规范/信息自由度 | 已有证据所回答的事 | 对本项目的限制 |
| --- | --- | --- | --- |
| Ohtsuki–Iwasa / Hilbe | 离散评判，公开或私有噪声 | 什么规则在给定信息条件下稳住合作 | 游戏/误差/公开性变化时不保证原排序 |
| Schmid / Michel-Mata | 累积量化声誉或多观察聚合 | 信息纠错和有条件的稳定性 | 不能把连续标签本身当作新增机制 |
| Pires / Horibe | LLM 评判或提示词文化演化 | 模型偏好与排除搭便车者的差异 | 近邻工作已覆盖“LLM 合作但不稳定”叙事 |
| Song / Zhu | 联合学习声誉规则与策略，或语言 gossip | 内生评判/通信可能改变合作 | “自发声誉”或“有 gossip”本身不是足够新意 |

## 3. RQ1：仓库中的具体失配与诊断

1. **优化的是短期私利，不是跨频率稳定性。** `experiments/v2_quantitative/prompts.py` 明确要求最大化“自身累计收益”；`population.py` 每代基于窗口收益做 Fermi 模仿，默认每次采纳还会发起一次 LLM 生成（“微变异”并非忠实复制）。而 `run_fixation_benchmark.py` 测的是每个组成 `k=1…N−1` 的稳态平均收益差 `d(k)`，再通过 `ρ=1/[1+Σ_i exp(−βΣ_{k≤i}d(k))]` 综合整个频率轴。一个在同类人群中收益高的规则，混入经典规则/ALLC/ALLD 后可能没有优势；频率轴某一区间的明显负 `d(k)` 足以使 fixation 很低。[演化实现](../experiments/v2_quantitative/population.py)；[fixation 方法](fixation_benchmark.md)。
2. **强先验导致筛选信号弱的可能性很高，但必须验证。** DeepSeek 官方于 2026-09-10 确认旧名 `deepseek-v4-flash` 临时转发 V4.1-Flash；用户报告的初始 80% / 40% 是待做受控配对实验的观察，不是文献确证的模型效果。当大家很早就互帮，C/C 中许多规范收益近似，稀有违规、噪声误判、向坏人合作与“惩罚坏人反被惩罚”这些必要分支既少遇到也少被选择；**适应度缺少把它们区分开的训练场景**。[DeepSeek 官方更新](https://api-docs.deepseek.com/updates/)；[当前 prompt](../experiments/v2_quantitative/prompts.py)。
3. **评估环境尚未完全同分布。** 运行脚本的演化和当前入侵/fixation 分析默认 `benefit=3,cost=1`；历史归档 benchmark 结果使用的是 `benefit=2,cost=1`，复现归档时必须显式传入 `--benefit 2`。须逐个核查实际 `evolutionary.json.config`，不能从默认值推断历史每一场都不同。演化/有限代入侵每代重置私有声誉和内部状态，而 fixation 的组成实验不跨“代”重置，先 burn-in 再取稳态；所以它们回答两个合法但不同的问题。此外，仓库把经典捐赠博弈表移植到**双人同时行动的囚徒困境**，明确不是原论文模型的精确重现。对比胜负应写成“本仓库移植后的 leading-eight 基线”。[运行参数](../experiments/run_fermi_v3.py)；[移植说明](../experiments/v2_quantitative/baselines.py)；[benchmark 协议](fixation_benchmark.md)。
4. **历史比较口径有风险。** `STRATEGY_SUPERIORITY_STANDARD.md` 仍以 `IS, SS, SJ, SC, SH, IS+, SS+, SJ+` 表述“八规范”，但当前 `baselines.py` 的 canonical 集合已是 `L1…L8`，且 `IS+` 等旧名被拒绝。必须把旧图、旧标准标为历史版本，重新冻结 L1—L8、ALLC、ALLD 的源码哈希及协议。README 中某些 0 噪声入侵曲线沿对角线，只表明处于“全好”吸收/近中性状态，不说明鲁棒；1% 噪声结果也不是无需复验的普遍胜利。[历史标准](../experiments/analysis/STRATEGY_SUPERIORITY_STANDARD.md)；[规范实现](../experiments/v2_quantitative/baselines.py)；[入侵 SOP](../INVASION_SOP.md)。
5. **fixation 推断可靠性先于胜负。** 文档已有极端 `k` 下双稳态与测量窗口漂移告警；5 个复现的标准差包络不是 95% 置信区间。`_verdict` 对 `ρ` 用绝对容差 `0.01`（在 `N=50` 的中性值 `0.02` 附近相当宽），展示词“neutral”不等于等效性证明。只有满足平稳性、置信界和预注册的阈值才能主张胜负；尤其 `ρ(candidate→L_i)>1/N` **不自动推出** `ρ(L_i→candidate)<1/N`，反向要用独立的成对次序测量。[fixation 文档](fixation_benchmark.md)；[判词实现](../experiments/analysis/invasion/run_fixation_benchmark.py)。

以上是原因假说的优先级排序；其中第 1、3、4、5 项有直接代码/文档证据，第 2 项需实验拆分“模型先验—环境—演化”三种贡献。现有较弱 fixation **已经否定“现成代表策略普遍强于 L1—L8”作为现成结论**，但没有否定改造搜索或新问题设置。

## 4. RQ2：先纠正比较，再让搜索对准失败机制

**阶段 A｜半天至数天，零或低 API 成本。** 固定每次实验的请求模型名、服务响应返回模型版本/日期（若 API 返回）、提示词哈希、温度、收益比、信息率、错误率、初始标签、fitness window、生成失败回退率、源代码哈希。只读审计当前记录；以同一组种子比较 V4/V4.1，若旧模型不可调用就把它标为历史观察，不做同条件版本差因果结论。对 ALLC、ALLD、L1—L8 和候选做 `observe/decide` 八状态覆盖测试，并记录“拒绝坏对象/惩罚是否被赦免/是否修复错误”的行为画像。消除旧基线名与参数分歧。

**阶段 B｜中成本，定位失败频段。** 候选对每个基线绘 `d_i(k)` 而非只报一次 `ρ`；重点对照 `k≈1`、中部与 `k≈N−1`，并记录 C/C、C/D、D/C、D/D 的混合频率、向坏对象捐赠概率和声誉判断一致率。给定实际随机种子，分别比较 `β`、`b/c`、噪声强度、观测概率、代间重置与稳态。对漂移组成延长 burn-in、按独立链并列出收敛诊断；必要时降低 `N` 做精确或高预算校验。**不得**挑出有利单一误差率重新定义“普遍更好”。

**阶段 C｜研究性改造。** 搜索分“同质环境合格”与“对抗泛化”两层：候选必须达到预先固定的同质合作下限；随后对少量冻结训练对手（ALLC/ALLD、挑选的 L1/L3/L7/L8、几类可解释变种）评估稀有进入/被进入的收益差与弱噪稳定性。在人口小规模先用多保真筛选，保留互不相同的 Pareto 候选，而非一上来为每次 LLM 变异做 `49 × 10 × 5 × 24000` 级别的全套仿真；最终用**未参与筛选的 L1—L8、误差、随机种子**和两方向 fixation 验证。对 LLM 做结构化反例反馈（例如“好人拒绝坏人时自己的评分为何变坏”），要求保留合法修复分支，而非提示“尽量合作”。对照组必须包括纯随机代码变异、显式 leading-eight 模板修复、旧版模型、以及同一预算下的固定代码搜索。赢家选择与测试集隔离，避免 benchmark overfitting。

预注册主结果宜改为：**在信息不全/有成本的特定设定，存在一类非经典、可解释规范，同时满足高合作、稀有入侵、抵抗反入侵；其提升是由某一机制（例如纠错/信任校准）而非更强模型先验产生。** 如观察条件不足，诚实报告失败边界，论文也可转为“高合作为何无法通过稳定性测试”的机制分析；但这与“成功演化更强规范”属于不同论文定位，不能用分析叙事冒充已达成的胜出目标。

## 5. RQ3：自由通信标签路线的价值与最小可行定义

目前 `agent-type2` 只提供 `__init__ / decide / observe`，`decide` 通过 `_ctx_opponent_id` 知道对手，`observe` 接收两人的动作和 ID；框架仍给它连续私有声誉表。**代码写出字符串作为内部状态不等于通信，更不等于自发形成共享标签。** 至少要新增明确的发信、接信接口，以及传送范围、带宽、持续期、随机噪声、冒名/撒谎约束与可计算成本。[type2 接口](../experiments/v2_quantitative/agent_full.py)；[当前提示](../experiments/v2_quantitative/prompts.py)。

近邻工作警告新意门槛：Song 等 2026 年已经联合学“声誉评估规则+行动策略”；Zhu 等 2026 年已经让代理利用开放文本 gossip 形成去中心化信任；Ashery 等 2025 年显示 LLM 群体可以形成自发语言惯例，但**共同约定一个词**与**这个词真实预测陌生人的合作行为并抗造假**是两种任务。因此可主张的差异应是“**协议和语义共同演化**：同一串消息能否在代理更替/对抗进入时维持行为可验证的间接互惠”，而不是“首次让 LLM 发明标签”。[Song et al., 2026](https://arxiv.org/abs/2606.04359)；[Zhu et al., 2026](https://arxiv.org/abs/2602.07777)；[Ashery et al., 2025](https://openaccess.city.ac.uk/id/eprint/35211/)。

最小实验：每次交互完成后，旁观者可以给其中一个 actor 发至多一个取自 **K 个长度受限符号**的消息；所有代理只能经固定观察概率收到，不能直接读取别人内部连续声誉。随代更替代码可以突变，消息随角色绑定的旧记录留存固定步数；允许“噪声/撒谎”条件但在公共真实行动日志上可审计。先在不计入主结果的校准实验中确认消息确实被不同代理理解，然后对比：无消息、固定 ± 数值/好坏、随机消息、仅私有 memory、自由符号无信誉惩罚、自由符号有消息信誉成本。分别报告标签频率熵、对外部审计行为的互信息、跨谱系可解释性/语义一致性、通信开销、抵抗谣言/冒名/策略迁移的收益及 bidirectional fixation。**共同采用某个“好”字但 ALLD 可以轻松伪装时判失败。**

有一个理论与工程陷阱：经典两类型 fixation 默认“类型内行为规则固定，给定 `k` 能测一个相对稳定的收益差”。若自由标签在同一组成内持续重新发明、语义随代变化，便无法直接把旧 `d(k)` 代入旧公式。必须先冻结演化完成的**完整代理程序+消息协议+初始词汇/记忆初始化**，再在每个固定组成校验平稳，或采用显式有限群体马尔可夫模拟估计 fixation；不能把不适用的旧公式算出的数当作证明。

## 6. 两条路线的研究判断与次序

把“当前结果已经普遍超过 L1—L8”当作核心机制，属于**已有证据反驳**，应放弃这个版本的主张，而不是承诺再调几个参数就能成功。这个判断仅针对现成候选与现有实验，不是对未来方案的否决。对“特定信息设定下针对性抗入侵”建议 **Accept with Revisions**：效能与稳健性有清晰的机制型改善途径，但尚未给出新的数据。对“内生可验证消息规范”亦为 **Accept with Revisions（风险更高）**：更像 *New Setting*，挑战预置标签假设；若只是 `agent-type2` 内部改存字符串则被现有多智能体声誉/通信工作覆盖，且不满足通信定义。

| 选择 | 最关键的可证伪试验 | 正结果可写成 | 失败后仍能回答 |
| --- | --- | --- | --- |
| A：对抗搜索社会规范 | 预注册候选对 L1—L8/ALLC/ALLD 在相同收益、噪声、私有观测条件下的 `d(k)` 与两向 `ρ`，未见种子重测 | 限定环境中的更优、可解释规范 | 失效阶段/频段及高合作≠稳定的机制 |
| B：内生通信标签 | 隔绝底层标签，在消息带宽/成本受控时，抗假消息且跨代理更替保持语义；与固定标签同预算比较 | 可演化、可验证的协议—行动共适应 | 空话均衡、语义漂移或过拟合的边界 |

优先建议：**先做 A 的协议审计与失败频段定位（利用既有代码和结果），再开 B 的最小通信原型**。不要为了得到胜过经典基线的图，把信息条件改成只对候选有利；如果最终只有局部环境优势，就在标题、图和摘要严格限定这种条件。对 B 的能力/时间判断暂时为“待用户确认”：本次未获知 API 预算、每周可投入时间、投稿节点，不能编造周期与确定的成功概率。

## 7. 结论

对 RQ1：版本升级可提高合作起点，却不会自动提供根据对象声誉实施正当拒绝与错误修复的条件规则；仓库的短期自身收益、LLM 非忠实变异、训练/测试的收益比和状态重置差异进一步放大这一距离。对 RQ2：首先对齐协议、修复基线命名并检查极端频率的稳态；随后以攻击者、行为诊断与未见条件约束搜索，不能把终局份额曲线替代 fixation。对 RQ3：自行形成**可传播且抗虚假的信息协议**有独立价值，但当前 type2 没有通信通道，且已有自发声誉与 gossip 研究，故须用“语义—行动共同演化”及严格消融界定新增贡献。

## 参考文献

1. Ohtsuki, H., Iwasa, Y., “The leading eight: Social norms that can maintain cooperation by indirect reciprocity,” *Journal of Theoretical Biology*, 2006.
2. Hilbe, C., Schmid, L., Tkadlec, J., et al., “Indirect reciprocity with private, noisy, and incomplete information,” *PNAS*, 2018.
3. Schmid, L., Ekbatani, F., Hilbe, C., Chatterjee, K., “Quantitative assessment can stabilize indirect reciprocity under imperfect information,” *Nature Communications*, 2023.
4. Michel-Mata, S., Kawakatsu, M., Sartini, J., et al., “The evolution of private reputations in information-abundant landscapes,” *Nature*, 2024.
5. Pires, A. S., Samson, L., Ghebreab, S., Santos, F. P., “How large language models judge and influence human cooperation,” arXiv:2507.00088, 2025.
6. Horibe, K., Itao, K., Toyokawa, W., “Emergence of Reputation-Based Cooperation in LLM Agents,” arXiv:2608.04507, 2026（预印本）.
7. Song, X., Huang, Y., Zhao, D., Feng, X., “Learning to cooperate with emergent reputation via multi-agent reinforcement learning,” arXiv:2606.04359, 2026（预印本）.
8. Zhu, S., Lin, Y., Kaistha, S., et al., “Talk, Judge, Cooperate: Gossip-Driven Indirect Reciprocity in Self-Interested LLM Agents,” arXiv:2602.07777, 2026.
9. Ashery, A., Aiello, L. M., Baronchelli, A., “Emergent social conventions and collective bias in LLM populations,” *Science Advances*, 2025.

**证据与范围说明：** Zotero 提供本地书目/摘要，近邻工作用期刊或 arXiv 主页面复核；2026 年预印本结论按作者报告而非独立复现处理。代码是当前脏工作区快照，本报告未更改其业务逻辑，运行中的完整配置/用户所述 80% 与 40% 尚待对照实测。
