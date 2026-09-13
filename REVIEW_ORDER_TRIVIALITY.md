# 四阶策略是否优于三阶策略：一个平凡结论的判定

**判定对象**：README 记录的"最佳策略"（`experiments/analysis/invasion/custom_strategies/readme_best.py`）
**问题**："四阶策略比三阶策略好"这个结论是否平凡？
**日期**：2026-09-10
**实验标签**：`ae0p01_oe0p01_w48`（action error = observation error = 0.01），N=100，13 档入侵规模 × 3 seeds × 50 代

---

## §0 结论摘要

| 命题 | 平凡？ | 本仓库证据 |
|---|---|---|
| P1 "四阶**不差于**三阶"（max₄ ≥ max₃） | ✅ **平凡（同义反复）** | 集合包含，无需实验 |
| P2 "**存在**四阶规则严格优于**所有**三阶规则，且增益可归因于第四路输入" | ❌ **非平凡** | **成立**：双向不对称 10 vs 95，且消融仅差第四路通道 |
| P3 "阶数越高越好"（单调性） | ❌ 非平凡，且**为假** | 反例：三阶变体 `thirdD` 对 L1–L8 的支配度高于四阶 |
| P4 "支配力可由合作水平预测" | ❌ 非平凡，且**为假** | 合作率与支配力**无单调关系**（见 §4.4） |

**一句话**：把命题限定为"裸的阶数比较"时它是平凡的；但一旦要求**严格分离 + 通道归因 + 对手依赖**，它就变得不平凡——而且在本仓库的实测中，**"高阶更好"这一单调叙事被证伪**，真正的结论是一个**对手依赖的偏序**。

---

## §1 为什么 P1 平凡

### 1.1 阶数的定义（本仓库口径）

依据 `experiments/v2_quantitative/baselines.py` 引用的规范来源，以及 Xia et al. (2023) 的汇总表：

| 阶 | 使用的信息 | 规则数 |
|---|---|---|
| 一阶 | donor 的**动作** | 4 |
| 二阶 | ＋recipient 的**声誉** | 16 |
| 三阶 | ＋donor 自身的**声誉**（standing） | **256**（即 "the leading eight" 所在的类） |
| 四阶 | ＋recipient 的**动作** | **65536**（"Kandori's rule"） |

本仓库的 `ASSESSMENT_TABLES` 用 8 行 = (actor_rep G/B) × (action C/D) × (recipient_rep G/B)，正是**三阶**信息结构。

### 1.2 平凡性的来源

每一阶都是上一阶信息的**超集**：任何四阶规则总能通过**忽略**第四路输入来逐字复现某个三阶规则。因此

$$\max_{\text{4阶}} P \;\ge\; \max_{\text{3阶}} P$$

对**任意**性能指标 $P$、**任意**环境与参数**恒成立**。这是集合包含的推论，不是经验发现。

> **因此**：如果论文的核心主张只是 P1，它就是平凡的。

### 1.3 让它非平凡所需的四要素

1. **严格性**：best-4th > best-3rd **严格**成立（不能只是 ≥）；
2. **归因**：增益来自第四路输入本身，而非参数量或搜索噪声（需 ablation）；
3. **可达性**：演化/学习过程**确实发现并维持**该规则；
4. **稳健性**：在噪声与私有评估下仍成立，并核算评估成本。

---

## §2 被测策略的阶数（实测）

对 `readme_best.observe` 做扰动敏感性测试（网格 $-1$ 到 $1$，步长 0.05，四种动作组合）：

| 输入 | 是否影响输出 | 对应信息 |
|---|---|---|
| `A_action` | ✅ | 一阶 |
| `B_rep`（recipient 声誉） | ✅ | 二阶 |
| `A_rep`（donor 自身声誉） | ✅ | **三阶** |
| `B_action`（partner 动作） | ✅ | **四阶** |
| `my_reputation` | ❌ | 未使用 |

**结论**：`readme_best` 是**四阶规则**（使用全部四路信息），`decide` 只依赖对手声誉（等价于 `cooperate iff opp_rep ≥ −0.09`）。

### 2.1 第四路通道的精确位置

`observe` 中仅有两处引用 `B_action`：

```python
# 合作分支：合作者被背叛 → 额外惩罚
if B_action == "defect":
    A_new -= 0.3
# 背叛分支：背叛合作者 → 额外惩罚
if B_action == "cooperate":
    A_new -= 0.4
```

---

## §3 行为图谱：第四路通道在哪里生效

（脚本：`tmp/_fourth_channel_map.py`）

| actor 动作 | partner 动作 | 网格单元 | 通道激活 | 平均 \|Δ\| | 最大 \|Δ\| |
|---|---|---|---|---|---|
| C | C | 40401 | **0 (0%)** | 0.0000 | 0.0000 |
| C | D | 40401 | 40200 (99.5%) | 0.2819 | 0.3000 |
| D | C | 40401 | 35150 (87.0%) | 0.3138 | 0.4000 |
| D | D | 40401 | **0 (0%)** | 0.0000 | 0.0000 |

**总体：75350 / 161604 单元激活（46.6%）；对称切片 100% 惰性。**

### 3.1 关键机制发现

第四路通道**只在非对称交互（C-D / D-C）上点火**，在对称交互（C-C / D-D）上**完全惰性**（数学上为 0，因为扣减项被 `if` 屏蔽）。

进一步，通道效应在**更新饱和处被截断**（clamping）：当 `A_rep = −1.0` 时，第三种与第四种规则输出相同（均为 −1.0），差异被 `max(−1.0, …)` 抹平。

> **可证伪预测**：第四阶相对第三阶的任何优势，都应随"非对称交互占比"变化——在全合作与全背叛两端趋于零，在混合区最大。

§4.4 将会显示：**这个预测被数据证伪**，而这本身是一个更深的发现。

### 3.2 声誉空间定位（通道在本体状态空间的生效区间）

（脚本：`tmp/_channel_reputation_space.py`）

**按 A_rep 分桶的通道激活率：**

| \|A_rep\| 区间 | 激活 / 总数 | 比例 |
|---|---|---|---|
| 0.0–0.1 | 7638 / 15276 | 50.0% |
| 0.1–0.5 | 8040 / 16080 | 50.0% |
| 0.5–0.9 | 7030 / 16080 | 43.7% |
| 0.9–1.0 | 7432 / 17688 | 42.0% |

**仅看非对称切片（C-D、D-C），按 A_rep 取值：**

| A_rep | 激活率 |
|---|---|
| −1.00 | **0.0%**（完全被 clamping 抹平） |
| −0.90 … −0.60 | 74.9% |
| −0.40 … +1.00 | **100.0%** |

**通道效应幅度（C-D 切片，B_rep = +0.5）：**

| A_rep | 三阶 | 四阶 | Δ |
|---|---|---|---|
| −1.00 | −1.000 | −1.000 | **+0.000**（clamped） |
| −0.90 | −0.860 | −1.000 | −0.140 |
| −0.70 … +0.90 | … | … | **−0.300（恒定）** |
| +1.00 | +1.000 | +0.700 | −0.300 |

**定位结论**：
- 通道在**几乎整个声誉空间**激活（A_rep ≥ −0.4 时非对称切片 100% 激活）；
- **唯一的失效边界是极端负声誉**（A_rep → −1.0），此处 `max(−1.0, …)` 把差异截断；
- 在激活区内，通道效应是**恒定 −0.300**（合作者被背叛）或 **−0.400**（背叛合作者），
  即一个**与声誉无关的固定惩罚**，而非依赖状态的精细化调制。

> 因此第四路输入的作用是：**在非对称交互上施加一个固定额外惩罚**。
> 这解释了 §4.4(c) 的"双刃剑"性质——它是固定罚金，因此过度惩罚无法自适应。

---

## §4 实验

### 4.1 消融设计

`fourth` 与 `third` **仅**在 `B_action` 两项上不同，其余（含 `decide`）**逐字节相同**。因此任何性能差异**只**能归因于第四路通道——这满足 §1.3 的归因要求。

| 标签 | 文件 | 第四路输入处理 |
|---|---|---|
| `fourth` | `readme_best.py` | 原生使用 |
| `third` | `readme_best_3rd.py` | **移除**两项（严格降阶到三阶） |
| `thirdC` | `readme_best_3rd_freezeC.py` | 冻结为"合作"（乐观） |
| `thirdD` | `readme_best_3rd_freezeD.py` | 冻结为"背叛"（悲观） |

`third` 是**原则上正确的降阶**（移除对第四路变量的依赖），`thirdC/thirdD` 是"冻结"变体，用于探测第四路信息的方向性价值。

### 4.2 判决实验：双向成对入侵

（引擎：`experiments/analysis/invasion/run_invasion.py --residents`；468 runs）

**核心判决（fourth vs third）：**

| 方向 | AUC | 达 50% 所需初始入侵数 | n=1 | n=50 | n=99 |
|---|---|---|---|---|---|
| **fourth 入侵 third** | **87** | **10** | 0.150 | 0.887 | 1.000 |
| **third 入侵 fourth** | **11** | **95** | 0.000 | 0.047 | 0.897 |

**强烈的双向不对称**：四阶只需 10% 初始占比即可取代三阶；三阶需要 **95%** 才能取代四阶。这是 P2 的直接证据，且因消融仅差第四路通道，增益**可归因**（§1.3 要素 1、2 满足）。

**完整曲线：**

| 初始入侵数 | fourth→third | third→fourth |
|---|---|---|
| 1 | 0.150 | 0.000 |
| 5 | 0.437 | 0.000 |
| 10 | 0.833 | 0.000 |
| 20 | 0.897 | 0.000 |
| 50 | 0.887 | 0.047 |
| 90 | 1.000 | 0.257 |
| 95 | 1.000 | 0.707 |
| 99 | 1.000 | 0.897 |

**全互相对抗的阈值矩阵**（行=入侵者，列=防御者，数值=达 50% 所需初始占比）：

```
              fourth    third   thirdC   thirdD
  fourth         —        10       10       30
  third         95         —       80       99
  thirdC        20        20        —       40
  thirdD        80         5       70        —
```

### 4.3 全序：fourth ≻ thirdC ≻ thirdD ≻ third

判定规则：A ≻ B 当且仅当 `thr(A→B) < thr(B→A)`。

| 比较 | thr(A→B) | thr(B→A) | 结论 |
|---|---|---|---|
| fourth vs third | 10 | 95 | fourth ≻ third |
| fourth vs thirdC | 10 | 20 | fourth ≻ thirdC |
| fourth vs thirdD | 30 | 80 | fourth ≻ thirdD |
| thirdC vs thirdD | 40 | 70 | thirdC ≻ thirdD |
| thirdD vs third | 5 | 99 | thirdD ≻ third |

**构成全序**：`fourth ≻ thirdC ≻ thirdD ≻ third`。四阶严格居首。

### 4.4 但"高阶更好"是假的：对手依赖与制度依赖

**(a) 对规范 Leading Eight 的支配度排序与成对排序不一致**（norm sweep，3120 runs）：

| 策略 | 平均入侵阈值 | 平均防御阈值 | 支配差 (防御−入侵) | 平均合作率 |
|---|---|---|---|---|
| `fourth` | 11.0 | 94.0 | +83.0 | 0.9403 |
| `thirdC` | 21.0 | 89.0 | +68.0 | 0.9266 |
| **`thirdD`** | **6.1** | **99.0** | **+92.9** | 0.7629 |
| `third` | 20.0 | 72.0 | +52.0 | **0.9821** |

按"对 L1–L8 的支配差"，**`thirdD` 优于 `fourth`**（92.9 > 83.0）——与成对全序**相反**。

> **结论**：不存在一个标量"阶数优势"。支配力是**对手集合的函数**：面对规范的**合作型** norm，偏背叛的 `thirdD` 最占优；但在直接成对中，`thirdD` 被 `fourth` 和 `thirdC` 击败。

**(b) 合作率与支配力无单调关系**（这是对朴素直觉的直接反驳）：

- `third` 合作率**最高**（0.9821），却是**最弱**的（支配差 +52.0，成对全序末位）；
- `thirdD` 合作率**最低**（0.7629），却在对 norm 的支配上**最强**。

**(c) §3.1 的制度区预测被证伪**：

| 方向 | 优势 @ coop ≥ 0.95 | 优势 @ coop < 0.95 |
|---|---|---|
| fourth(入侵) vs third | **+0.0341** | **−0.0363** |
| fourth(入侵) vs thirdD | +1.7576 | +0.9052 |

优势在**高合作区为正、在混合区为负**——与"通道在混合区点火更多 → 优势更大"的预测**方向相反**。第四路通道实为**双刃剑**：在近均匀合作时提供更精细的判别力，在混合人群中则因额外惩罚（对"合作者被背叛"也扣分）而过惩罚，反而有害。

**(d) 方法学警示**：瞬态平均适应度在两种策略间**高频变号**（`tmp/_fitness_coherence.py`），说明该制度下选择接近**近中性漂移**——**均值适应度不是充分统计量**，判决必须依赖**频率轨迹**（synchronous deterministic imitation + 频率依赖选择）。

### 4.5 规范 norm 层面的对照（补充）

策略入侵 L1–L8 的阈值：

| 策略 | L1–L8 | ALLC | ALLD |
|---|---|---|---|
| `fourth` | **10** | 10 | 20 |
| `thirdC` | 20 | 20 | 30 |
| `thirdD` | 5 | 1 | 20 |
| `third` | 20 | 20 | 20 |

**注**：`fourth` 对 8 个 L-norm 的曲线**完全重合**（阈值均为 10）——因高合作锁定时所有 norm 的 `decide` 都落在其 GG 行（全部合作），规范差异被抹平；这与 `seed0_coop` 的早期观察一致。**要展示 norm 间区分度，必须用会触发低声誉行的对抗性策略。**

---

## §5 文献判断：该方向尚无定论

（Zotero 检索，库内相关条目）

### 5.1 支持"高阶有机制性必要性"（唯一强证据）

**Ohtsuki & Iwasa (2004)**，经 Coopeval 转述：
> "higher-order information **can be helpful for eliminating higher-order free riders**—such as second-order free riders (players that always cooperate)."

### 5.2 明确的反向证据（高阶反而有害）

**Tewolde et al. (2025), Coopeval**（Zotero `E5LAACE5`）：
> "There is **no consensus** in the literature on whether the summary of the past ought to include higher-order information… Human behavior seems to be better explained by **first-order** decision rules (Wedekind & Milinski 2000)… Our results… indicate that higher-order information about a co-player's past **does more harm than good** to the cooperative propensities of our tested LLM models."

> 注意：这与本仓库 §4.4(c) 的"混合区为负"在**方向上一致**。

### 5.3 三阶优势在噪声下会失效

**Hilbe et al. (2018)**，经 **Xia et al. (2023)** *Physics of Life Reviews* 46, 8–45 转述：
> "the leading eight strategies **may lose the advantages** in the evolution of cooperation when the related information is erroneous or noisy."

**Schmid, Ekbatani, Hilbe & Chatterjee (2023)**, *Nature Communications*：量化评估在**不完美信息**下稳定间接互惠——说明性能取决于**评估制度**，而非单纯的阶数。

### 5.4 该方向被明确标记为"尚未解决"

**Fujimoto & Ohtsuki (2023)**, PNAS（Zotero `BHYZGP3Q`）：
> "humans may use more complex norms than second-order ones. Studying the effect of higher-order information… **would further deepen our understanding**."

### 5.5 高阶有成本，须做净额比较

**Santos, Santos & Pacheco (2018)**：*Social norms of cooperation with costly reputation building*（`3ICZYJDS`）与 *Indirect reciprocity and costly assessment in multiagent systems*（`JRXELW8X`）。第四路信息不是免费的，"更好"必须**扣除评估成本**。

### 5.6 低阶也能促进合作（说明非单调阶梯）

- **Dong, Sun, Xia & Perc (2019)**（`9C4R6VDK`）：二阶声誉促进空间 PD 中的合作；
- **Yang, Wang & Xia (2019)**（`PYU9YMQ6`）：三阶声誉评估促进空间 PGG 中的合作。

各阶都有增益报告，但**不存在"阶数越高越好"的定理**。

### 5.7 综述定位

**Xia, Wang, Perc & Wang (2023)** *Physics of Life Reviews* 46, 8–45 的表 2 给出阶数层级（4→16→256→65536），其中 256 对应 "the leading eight rules"，65536 对应 "Kandori's rule"。

**Okada (2020)**（`XPFKYF4Y`、`DI48MGRR`）：间接互惠理论综述。

---

## §6 最终判定

1. **P1（"四阶不差于三阶"）平凡**——集合包含的同义反复，无需实验，也不是发现。
2. **P2（"四阶严格优于三阶且可归因"）非平凡，且在本仓库成立**：
   - 双向不对称 **10 vs 95**；
   - 消融仅差 `B_action` 两项，**归因唯一**；
   - 满足 §1.3 的要素 1、2。
3. **"阶数越高越好"为假**：
   - 反例 `thirdD`（同为三阶）对 L1–L8 的支配差（+92.9）**高于**四阶（+83.0）；
   - 支配力**对手依赖**，不存在标量阶数优势；
   - 第四路通道是**双刃剑**：高合作区有利（+0.034），混合区有害（−0.036）。
4. **合作水平不能预测支配力**：合作率最高者（`third`, 0.982）最弱；最低者（`thirdD`, 0.763）在对 norm 的支配上最强。

### 可发表的非平凡表述

> 第四路（recipient 动作）信息在本制度下**严格、可归因地**提升了策略对**合作型规范**的可入侵性（双向不对称 10 vs 95），但该增益**不是**由通道的点火频率单调决定——它在近均匀合作区为正、在混合区转为负，且**被另一个三阶规则在规范支配度上超越**。因此"更高阶更好"不成立；正确的陈述是**对手依赖的偏序**。

---

## §7 复现

```powershell
# 阶数判定（敏感性 + 符号依赖 + decide 真值表）
uv run python tmp/_order_readme_best_v3.py
uv run python tmp/_order_confirm.py
uv run python tmp/_verify_readme_best_equiv.py

# 行为图谱
uv run python tmp/_fourth_channel_map.py
uv run python tmp/_channel_reputation_space.py

# 成对判决（每个无序对一条，468 runs）——统一入口的 --residents 模式
uv run run-invasion `
  --source fourth=experiments/analysis/invasion/custom_strategies/readme_best.py `
  --source third=experiments/analysis/invasion/custom_strategies/readme_best_3rd.py `
  --source thirdC=experiments/analysis/invasion/custom_strategies/readme_best_3rd_freezeC.py `
  --source thirdD=experiments/analysis/invasion/custom_strategies/readme_best_3rd_freezeD.py `
  --residents fourth>third --residents fourth>thirdC --residents fourth>thirdD `
  --residents third>thirdC --residents third>thirdD --residents thirdC>thirdD `
  --workers 48 --action-error 0.01 --observation-error 0.01 `
  --output results/quantitative_baseline/invasion/pairwise_order_ablation_ae0p01_oe0p01_w48

# 对规范 norm（3120 runs）——同一入口的 --norms 模式
uv run run-invasion `
  --source fourth=... --source third=... --source thirdC=... --source thirdD=... `
  --workers 48 --action-error 0.01 --observation-error 0.01 `
  --output results/quantitative_baseline/invasion/n100_order_ablation_norms_ae0p01_oe0p01_w48

# 汇总 / 制度区定位
uv run python tmp/_summarize_pairwise.py
uv run python tmp/_summarize_order_norms.py
uv run python tmp/_regime_localization.py
uv run python tmp/_characterize_strategies.py
uv run python tmp/_fitness_coherence.py

# 图（已按 2026-09-11 要求移除，以下命令可重新生成）
uv run python -m experiments.analysis.plot_pairwise_invasion `
  --summary results/quantitative_baseline/invasion/pairwise_order_ablation_ae0p01_oe0p01_w48/summary.json `
  --output README.assets/pairwise_order_ablation_ae0p01_oe0p01.png
uv run python -m experiments.analysis.plot_n100_invasion_count_sweep `
  --summary results/quantitative_baseline/invasion/n100_order_ablation_norms_ae0p01_oe0p01_w48/summary.json `
  --output README.assets/n100_order_ablation_norms_ae0p01_oe0p01.png
```

### 产物

| 类型 | 路径 |
|---|---|
| 策略源码 | `experiments/analysis/invasion/custom_strategies/readme_best{,_3rd,_3rd_freezeC,_3rd_freezeD}.py` |
| 成对引擎 | `experiments/analysis/invasion/run_invasion.py`（`--residents A>B`） |
| 成对结果（数据） | `results/quantitative_baseline/invasion/pairwise_order_ablation_ae0p01_oe0p01_w48/` |
| norm 结果（数据） | `results/quantitative_baseline/invasion/n100_order_ablation_norms_ae0p01_oe0p01_w48/` |
| 图（已移除） | 本报告的两张图已于 2026-09-11 按要求删除；数据保留，可由上方命令重绘 |

---

## §8 声明与局限

- 本报告的成对与 norm 结果均为 **N=100、3 seeds、50 代、ae=oe=0.01** 下的**描述性**结论；未做多重比较校正，不构成"演化稳定"或"优于 L8"的性能主张（参见 `STRATEGY_SUPERIORITY_STANDARD`）。
- `thirdC/thirdD` 是**冻结式**变体，用于探测第四路信息的方向性价值；原则上正确的降阶是 `third`（移除依赖）。
- §4.4(d) 的近中性漂移意味着小样本下的成对阈值对种子敏感；本报告使用 3 seeds × 13 档规模的**聚合**曲线以降低该风险，但未做正式的置信区间估计。
