# 私人声誉矩阵热图（Private-reputation matrix heatmap）

把一次演化实验的**最终种群**在私人声誉上"谁对谁怎么看"直接画出来。

回答的问题：**所有观察者对同一个目标的看法，数值上是否一致？**
—— 不是"符号是否一致"，而是"给的是不是同一个数"。

---

## 1. 为什么需要这个视图

只看"符号"（好 / 坏）的判断，对**"每个观察者一个固定宽严偏移"完全免疫**。
而 LLM 演化出的声誉矩阵往往是**每个 agent 对所有目标给同一个值**，所以分歧几乎全部是这种偏移。

实例（5 个观察者给同一批目标打分）：

| | 观察者 A | B | C | D | E |
|---|---|---|---|---|---|
| 打分 | +0.95 | +0.90 | +0.85 | +0.80 | +0.10 |
| 只看符号 | 好 | 好 | 好 | 好 | 好 |

- **符号上完全一致**（5 个都 ≥ 0，"张三算好人"）
- **数值上差得很远**：A 与 E 相差 **0.85**

E 给 0.10 意味着"勉强算好人"，A 给 0.95 意味着"是圣人"。
若阈值改成 `≥ 0.2`，E 立刻翻脸——**共识瞬间蒸发**。

热图把标量所依据的**原始矩阵**画出来，让这件事一眼可见。

---

## 2. 命令

```bash
# 每个 --source 一列（建议 3-6 列）
uv run python -m experiments.analysis.consensus.plot_private_reputation_matrix \
  --source N16v2_s0=results/quantitative_baseline/LLM_..._5seed_seed0/evolutionary.json \
  --source N16v2_s1=results/quantitative_baseline/LLM_..._5seed_seed1/evolutionary.json \
  --output README.assets/private_reputation_matrix_N16v2_5seed.png \
  --seed 0 --interactions 1000
```

也可用已注册入口：`uv run plot-private-reputation-matrix ...`

**常用选项**

| 选项 | 说明 |
|---|---|
| `--source LABEL=PATH` | 可重复；每个是一列。`PATH` 必须是 `evolutionary.json` |
| `--output` | 输出 `.png`（同时写同名 `.pdf`） |
| `--seed` | 重放的随机种子（默认 0） |
| `--interactions` | 每代交互数（默认 1000） |
| `--condition` | **仅作标签**；实际条件由下面几个开关决定 |
| `--action-error` | 动作误差概率 |
| `--observation-error` | 观察误差概率 |
| `--observability` | `full` / `partial` / `private` |
| `--observability-p` | `partial` 时第三方观察概率 |
| `--annotate` | 强制把数值写进每个格（`n > 12` 时会糊） |

> ⚠️ 只支持 `agent_type == "agent-type1"`。type-2 让 LLM 自持任意内部状态，
> "私有声誉矩阵"没有良定义，`load_run` 会直接报错。

---

## 3. 图怎么读

每个 run 一列，纵向两张方阵热图：

| 位置 | 内容 |
|---|---|
| **上行** | **声誉矩阵** `[观察者, 目标]`；颜色＝该观察者私下持有的分数 |
| **上行右侧** | 该观察者的**行均值**条形图 ＝ 它对外的一贯宽严 |
| **下行** | **两两分歧矩阵** `[观察者, 观察者]`；格 `(i,j)` ＝ 两人共同评过的目标上的 `mean∣score_i − score_j∣` |

**样式约定**（都是为了"不把结构看错"）：

- **灰色格 ＝ 没有记录**（对角线＝不给自己当第三方打分；部分可观测下未观测到的也是灰）。
  刻意与 `RdBu_r` 中点的白（＝分数**恰为 0.0**，即 GOOD 阈值）区分开——两者含义完全相反。
- **白色细网格**分隔单元格。若用 `imshow` 且无网格，相邻同色格会糊成一块，行内结构反而看不出来。
- **列下方**给出唯一的指标 `D`。
- 两行色标**固定不自动缩放**（声誉 `[-1,1]`，分歧 `[0,1]`），保证各列可横向比较。

### 读图配方

1. 上行**行内是否纯色**？纯色 ⇒ 该 agent 对所有目标一个态度。
2. 纯色的行**彼此颜色是否相同**？不同 ⇒ 存在**观察者宽严偏移**。
3. 下行红色有多深？深红格 ＝ 该对观察者已严重分歧（`> 0.50`）。
4. `D` 把以上合并成一个数，见 §4。

> ⚠️ **白 ≠ 灰**：声誉矩阵里白 ＝ 分数恰为 0.0（GOOD 阈值，有意义的取值），
> 灰 ＝ 没有记录（对角线／未观测）。分歧矩阵里白 ＝ 零分歧。看错会得出相反结论。
>
> ⚠️ **不要只看"符号是否一致"下结论**。符号级判断对观察者固定偏移完全免疫。
> 本模块因此把结论压成一个数值指标 `D`。

---

## 4. 指标

只有一个数 **`D`**，单位就是**声誉刻度**（与游戏同用 `[-1, 1]`），可直接读数不需换算。

实现在 `core.py::disagreement`：

$$D=\sqrt{\frac{1}{|\Omega|}\sum_{(i,t)\in\Omega}\left(v_{it}-\bar v_t\right)^2},
\qquad \bar v_t=\frac{1}{|\{i:(i,t)\in\Omega\}|}\sum_i v_{it}$$

$\Omega$ ＝ 被观测到的非对角格；$\bar v_t$ ＝ 目标 $t$ 的列均值（只在评过它的观察者上平均）。

**为什么是这个式子**：把"所有人都同意"定义成"每一列都是常数"，
$\bar v_t$ 就是最小二乘意义下**离观测矩阵最近的那个一致矩阵**，$D$ 是残差的 RMS。
于是 $D$ 的读法是直接的：

| 取值 | 含义 |
|---|---|
| `D = 0` | 每个观察者对每个目标都给出同一个数 —— 完全一致 |
| `D → 1` | 刻度允许的最极端分裂（一半 −1、一半 +1） |

$D \in [0,1]$ 不用另外证明：取值都在 $[-1,1]$，任何点到列均值的偏差不超过 1。

没有共同评过的目标时 `D = NaN`（图上显示 "no shared targets observed"）。

**为什么不用"两两平均 $|a-b|$"**：那个量不是范数，分母按"谁碰巧评过"加权，
两个数之间也没有天然上界。改成"投影到最近一致矩阵 + 取残差范数"，
直觉完全一样，却只需要一次列均值（不必 $O(n^2)$ 配对扫描），并且天然落在 $[0,1]$。

上行右侧的条形图只是**画面**（每个观察者的行均值），不构成一个指标。

### 为什么不再用 ICC(2,1)

曾经算过 `ICC(2,1)`，现已删除，原因有三：

1. **本仓库没有它的文献依据**——库里只收录间接互惠领域论文，没有信度方法学
   （Shrout & Fleiss 1979 / McGraw & Wong 1996 均不在库）；
2. **在"每行常数"的矩阵上它不可解释**：每个观察者对所有目标全并列，
   “排序信息”本身不存在，低 ICC 并不等于“排序不一致”；
3. **它是冗余的**：`D` 已直接回答“给的是不是同一个数”，
   而且单位就是声誉刻度，比一个无量纲方差比好读得多。

同时删除了两个**随手取、无出处**的常量：`AGREEMENT_TOLERANCES = (0.05, 0.10, 0.20)`
（原仅用来在色标上画容差虚线）与 `NEGLIGIBLE_SPREAD = 1e-3`（原仅用于 ICC 的退化守卫）。

### 退化情形（刻意行为，不是 bug）

| 情形 | 处理 |
|---|---|
| `reputation` 非方阵 | `ValueError` |
| 没有共同评过的目标 | `D = NaN` |

---

## 5. 库 API

```python
from experiments.analysis.consensus.core import (
    load_run, build_from_codes, play_generation, disagreement,
)
from experiments.analysis.consensus.plot_private_reputation_matrix import (
    observer_matrix, effective_matrix, row_severity,
    pairwise_disagreement_matrix, plot,
)

run = load_run("results/.../evolutionary.json")      # 仅 agent-type1
rep, obs, ids = observer_matrix(run.codes, run.size, seed=0, interactions=1000)

disagreement(rep, obs)                      # 唯一的数值指标 D
```

`core.py` **不引入 matplotlib**；绘图只在 `plot_private_reputation_matrix.py`。
`core.py` 只依赖 numpy（无 scipy / pandas）。

---

## 6. 验证状态

| 组件 | 验证方式 | 状态 |
|---|---|---|
| 重放 / 加载 / 池构造 | `tests/test_private_reputation_consensus.py` | ✅ |
| 热图 helper（掩蔽、偏移、两两分歧） | 同上 | ✅ |
| 指标 `D`（零、列均值、上界、掩蔽、非方阵） | 同上 | ✅ 21 tests |
| 端到端出图 | 单列与多列均实跑 | ✅ |
| 与规范基线一致 | `L1` 齐质种群 → `D = 0`（数值完美一致） | ✅ |
| 与旧指标一致 | 5 个真实 run 上 `D` 与旧 `mean∣a-b∣` **排序完全相同** | ✅ |

```bash
uv run pytest tests/test_private_reputation_consensus.py -q   # 21 passed
```

---

## 7. 已知限制

- **只支持 `agent-type1`**（有私有 `reputations` 矩阵）；type-2 会显式报错。
- **只重放一代**：种群组成固定为最终代。回答"最终种群这个配置能否维持数值共识"，
  不是"共识能否被演化选出来"。
- **逐代重置声誉**：共识只能**一代之内**谈。
- **未观测项不计入统计**（走 `live` 掩蔽）；`private` 条件下覆盖率低，
  指标只在被评过的对上计算，避免被中性先验稀释。
- **`--condition` 只是标签**，不会自动设置误差参数；用 `--action-error` /
  `--observation-error` / `--observability` 真正指定条件。
- 每列只有**一个读取**（一份矩阵），不是多次独立重放的汇总；
  需要多种子比较就多传几个 `--source`。
