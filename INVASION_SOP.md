# 入侵检测 SOP：演化策略 vs Leading Eight（入侵扫描 + 固定概率 + 报告）

> 状态：✅ 已验证（2026-09-08 首次；2026-09-13 并入固定概率与报告生产）
> 更新：🔒 **噪声强制**（2026-09-08）——入侵实验**必须带噪声**，`--action-error` /
> `--observation-error` 不再是可选项，须 ≥ `0.01`（推荐 1%）。
> 更新：🔓 **移除方向维度**（2026-09-11）——实验只有一条频率轴，不再区分单/双向；
> 结果目录与 summary schema 均已扁平化（旧结果仍可读取，见 §3）。
> 更新：📐 **两个绘图脚本的分工已写明**（2026-09-13）——norm 扫描用
> `plot_n100_invasion_count_sweep`（§5.3），策略对策略用 `plot_pairwise_invasion`（§5.4），
> 选择依据见 §5.1 对照表。
> 更新：⚖️ **种群规模 N 成为一等参数**（2026-09-13）——不再固定 100；默认 **N=20**，
> 宿主规模由 `--population-size` 指定，结果目录与图标题都带 `n{N}`。
> 更新：🔬 **每代交互预算提高到 10000**（2026-09-13）——这是**结论有效性的前提**，
> 不是性能调优；1000 配对/代会制造**假入侵**，详见 §0.1。
> 更新：📄 **并入固定概率 benchmark 与报告生产**（2026-09-13）——本文现在覆盖
> 「入侵扫描 → 固定概率 → 可视化 → 报告」的完整链路。
>
> 适用范围：对已完成的 `V2EvolutionaryPopulation` 演化运行（schema v4）中提取出的
> **主导策略**，用两套独立方法评估其对抗能力：
> 1. **入侵扫描**（§3）——有限代数内的频率响应，回答「能不能在群里站住」；
> 2. **固定概率 benchmark**（§4）——Traulsen–Hauert ρ，回答「稀有时能否被选择推动」。
>
> 与 `ANALYSIS_SOP.md` 相互独立——那份只覆盖演化运行本身的聚类/谱系/曲线，
> 本文只规定**入侵对抗**部分。底层原语见 `experiments/analysis/invasion/core.py`、
> `experiments/analysis/invasion/run_invasion.py`、
> `experiments/analysis/invasion/run_fixation_benchmark.py`。
> 方法学背景见 `docs/fixation_benchmark.md`。

---

## 0. 关键设计约定（必须先理解）

### 0.1 每代交互预算（结论有效性的前提）

**`--interactions` ≥ `10000`。** 这不是性能选项，而是结论是否成立的分界：

- `--interactions` = **整个种群每代的配对总数**（不是每人配对次数）。N=20 时每人每代
  约 `2 × 10000 / 20 = 1000` 次；计分窗口 `fitness_window_fraction=0.2` 只取最后 20%，
  即约 2000 对计入适应度。
- 预算过低时计分窗口的样本量不足 → 适应度估计**信噪比过低** → 阈值型复制规则
  （`strictly_higher_fitness_always_copied`）会把噪声放大成**单向棘轮** →
  不管有没有真实优势都跑到 100%。这是**假入侵**。
- 实测对照（同 N、同代数、同 seeds，只改预算）：低预算下候选策略对全部 8 个 norm
  都"固定"到 100%；高预算下全部收敛到**内部共存平台**且不固定。
  → **低预算的入侵结论必须作废重跑。**
- 推论：改变 `--interactions` 会**改变结论**，因此每个结果都会把实际值写进
  `config.interactions_per_generation`，缓存也按它比对。历史低预算结果不会被
  自动重跑，**引用前必须核对这个字段**。

### 0.2 种群规模 N 是一等参数

- 默认 `--population-size 20`。宿主规模**不必**与演化时的规模（N=12/16/50）一致：
  入侵扫描看的是**频率**，演化策略只是作为代码注入宿主，二者唯一耦合点是主导策略的提取逻辑。
- **不要**因为演化规模是 N 就新建一个 `run_n{N}` 脚本——复用 `run_invasion.py`，
  把 `--population-size` 传进去即可（脚本已参数化）。
- 改 N 会连带改变**初始入侵者计数网格**：必须满足 `1 ≤ count < N`，
  `count/N` 才是频率。默认网格已按 N=20 设定；换 N 时用 `--invader-counts` 重给。
- 结果目录与 summary 都带 `n{N}`；绘图器从 summary 读 N 并自动改标题，
  所以同一套绘图代码能画任意 N。

### 0.3 两套方法的分工

| | 入侵扫描（§3）| 固定概率 benchmark（§4）|
|---|---|---|
| 回答 | 有限代数后**站不站得住** | 稀有时**选择推不推得动** |
| 量 | 终值频率响应曲线 | Traulsen–Hauert ρ |
| 声誉 | **每代重置** | **跨轮累积**（跑到稳态）|
| 强项 | 直观、能看阈值与共存 | 与基准可比、有中性线判据 |
| 弱项 | 是**瞬态**，不是稳定性判据 | 慢，且受 stationarity 约束 |

**两者必须一致才可下结论。** 若不一致，优先怀疑**交互预算**（§0.1），
其次怀疑**stationarity 未收敛**（§4.3）。两者机制不同的根源：
ρ 把适应度差曲线沿组成**逐点积分**（有序列乘积），而入侵扫描用的**阈值型逐对复制**
在两个适应度分布的**交叉点**达到流量平衡，与"均值相等"不等价。

### 0.4 噪声强制

**🔒 入侵实验必须带噪声（强制约定）**：`--action-error` 与 `--observation-error` **都不能省**，
且各自**必须 ≥ `0.01`**（推荐 `0.01`，即 1%）。原因（重要，写论文时须引用这一动机）：
- 在 **0 噪声**下，连续声誉会收敛到"全好声誉稳态"（reputations clamp 到 +1.0），各 norm 与演化策略
  的 `observe`/`decide` 分支几乎一致 → 行为趋同 → 入侵曲线全部贴合"无频率变化"
  对角线（README §9 中 `agent-type1` 的中性结果即源于此）。
- 加上噪声后，声誉**不再稳定锁死在全好值**，隐藏的选择差异才显现出来；此时
  演化策略对 norm 的**真实优势/劣势**才可观测（README §10 证明同策略在加噪后从对角线破出）。
- 因此：**不带噪声的入侵结果会被全好稳态污染，结论不可信**。这是本 SOP 的硬性前置条件，
  任何不带噪声的入侵扫描都应视为无效产出。

> 注意：`run_fixation_benchmark` 的 `--action-error` / `--observation-error` 默认是 `0.0`。
> **两个方法必须用同一噪声水平**才能互相印证，所以固定概率那一步也要显式传 `0.01`（§4）。

### 0.5 声誉语义

**连续值声誉基线**：各 norm（`experiments/v2_quantitative/baselines.py`）使用**连续值声誉**
（`dict[int, float]`，初始 `0.0`，按 `STEP=0.333` / `BIG_STEP=0.5` 增减并 clamp 到 `[-1,1]`）；
`Good/Bad` 仅作为 `>0` / `≤0` 的阈值判断条件，写入的仍是连续数值。评估时须理解这一"软声誉"语义。

**声誉跨代重置（2026-09-09 明确）**：入侵扫描每个周期的 `reset_for_generation()` 现在会将整个
`reputations` 矩阵重置回初始 `{agent_id: INITIAL_REPUTATION}`（即 `{agent_id: 0.0}`），
**不再**跨代累积声誉。受影响位置：`experiments/v2_quantitative/agent.py`、
`experiments/v2_quantitative/agent_full.py`、`experiments/analysis/invasion/core.py`。
行为由 `tests/test_verify_reinit.py` 固定（断言重建后 `reputations == {aid: INITIAL_REPUTATION}`）。
这保证了每代从"中性声誉"起步，避免跨代声誉残留影响选择压力。

> ⚠️ **固定概率 benchmark 与此相反**：它**不重置**声誉（Schmid et al. 2023 方法），
> 因此必须先 burn-in 再测量。这是两个方法不可互换的核心差异。

---

## 1. 输入与前置条件

| 项 | 约定 |
|---|---|
| 演化结果 | `results/quantitative_baseline/$LABEL_seed<N>/evolutionary.json`（schema v4，必须含 `final_population` + `lineage_events`）|
| 宿主规模 | `--population-size` 指定，默认 `20`；与演化规模无关（§0.2）|
| 交互预算 | `--interactions` 须 ≥ `10000`（§0.1，**结论有效性的前提**）|
| 噪声 | 两法均须显式给 `--action-error 0.01 --observation-error 0.01`（§0.4）|
| norm 集合 | `NORMS = (*LEADING_EIGHT, "ALLC", "ALLD")`（8 个 Leading Eight + 两个无条件基准，`core.py`）|
| CLI 入口 | `run-invasion` / `run-fixation-benchmark`（run）；`plot-n100-invasion-count-sweep` / `plot-pairwise-invasion` / `plot-fixation-benchmark`（plot，见 §5.1 对照表）|

> legacy schema v3（无 `lineage_events`，如 `LLM_v3_*` agent-type2 run）：`load_representative_from_path`
> 会自动回退到"按相同 code 分组、取最大组、组内最高 fitness"的代表选取（`_representative_without_lineage`）。
> 此时 `lineage_id` / `root_lineage_id` 记为 `-1`，报告中不应引用谱系字段。

前置检查：

1. schema v4 + `lineage_events` 非空（`load_representative_from_path` 需要它重建谱系）。
2. `.env` 中 DeepSeek key **不需要**——入侵对抗全是本地模拟（无 LLM 调用）。
3. 命令统一 `uv run python -m ...`（或 `uv run <entry-point>`），工作目录 = 仓库根。
4. 一次性诊断脚本放 `tmp/`（已被 `.gitignore` 忽略）；**但报告里的数字/图不得依赖它们**（§6.4）。

---

## 2. 主导策略提取（每 seed 一个代表）

入侵的"来源"是演化实验最终种群的**主导策略**。现成的提取函数：
`experiments/analysis/invasion/run_invasion.py::load_representative_from_path`
`experiments/analysis/invasion/run_invasion.py::parse_sources`

提取规则（固定，勿事后更改）：
- 对每个 seed 的 `final_population`，按 `lineage_events` 把每个 member 归属到 **root lineage family**；
- 取**成员最多的家族**；若并列取 root_lineage_id 最小者；
- 在该家族内取 **fitness 最高**（并列取 agent_id 最小）的成员作为该 seed 的主导策略代表。

这样每个 seed 给出一个 `EvolvedSource`（含 code + shasum），作为入侵扫描的 `--source`。

---

## 3. 运行入侵扫描

**统一入口 `run-invasion`**：一次运行 = **一个候选策略 vs 一个对手（resident）**。
对手是 norm 还是另一个策略，只差一个 flag。

多策略（多 seed）扫描不需要新脚本：把每个 seed 的 `evolutionary.json` 各传一个
`--source`，一次运行就产出多面板 summary。

```powershell
# (a) 候选策略 vs 规范 norm（默认模式）
#     种群规模与交互预算都要显式给，不要依赖默认值
uv run run-invasion `
  --output "results/quantitative_baseline/invasion/$RUN_DIR" `
  --source "<SRC_LABEL>=agent-type1=$OUT/<LABEL>`_seed$S/evolutionary.json" `
  --population-size 20 `
  --interactions 10000 `
  --generations 50 `
  --action-error 0.01 --observation-error 0.01 `
  --workers 48

# 多策略（本例 5 个 seed）：重复 --source 即可
#   --source b2_seed0=agent-type1=.../seed0/evolutionary.json `
#   --source b2_seed1=agent-type1=.../seed1/evolutionary.json `  ...

# (b) 候选策略 vs 另一个任意策略（策略对策略）
uv run run-invasion `
  --output "results/quantitative_baseline/invasion/$RUN_DIR" `
  --source "A=agent-type1=a.json" --source "B=agent-type1=b.json" `
  --residents "A>B" `
  --population-size 20 --interactions 10000 `
  --action-error 0.01 --observation-error 0.01 --workers 48
```

`--source` 同时接受 `evolutionary.json`（自动取主导谱系代表）和手写 `.py`；
agent type 可省略（默认 `agent-type1`）。

关键参数（`--source` 必填，噪声**必填**，其余可选）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--output` | `.../n{N}_invasion_count_sweep`（带噪声时 `..._aeX_oeY`）| 结果目录；**建议目录名带 `n{N}` 与噪声后缀** |
| `--source LABEL=[AGENT_TYPE=]PATH` | （必填）| 候选策略；`.json` 自动取主导谱系代表，`.py` 直接读源码；可重复 |
| `--population-size` | `20` | 宿主种群规模 N；须 ≥3。结果目录/标题均带 `n{N}`（§0.2）|
| `--norms` | 全部 10 个（L1–L8 + ALLC + ALLD） | 作为对手的规范 norm 子集（默认模式）|
| `--residents CAND>RES` | — | 策略对策略模式；与 `--norms` 互斥，可重复 |
| `--invader-counts` | `(1,2,3,4,5,6,8,10,12,14,16,18,19)` | 初始入侵者**数量**；每个都必须满足 `1 ≤ count < N`（`count/N` 即初始频率）。换 N 时须用本参数重给 |
| `--seeds` | `(0,1,2)` | 每个 (pair,count) 重复的随机种子数 |
| `--generations` | `50` | 每轮博弈代数（§3.1）|
| `--interactions` | `10000` | **每代全种群配对总数**（不是每人次数）。**必须 ≥10000**，见 §0.1 |
| `--fitness-window-fraction` | `0.2` | 计入适应度的最后交互占比（其余为 burn-in）|
| `--action-error` / `--observation-error` | `0`（但**本 SOP 强制 ≥ `0.01`**）| 动作/观测噪声；推荐 `0.01`（1%）。<br>**不可为 0**——0 噪声会落入全好稳态掩盖差异（见 §0.4）|
| `--workers` | `8` | 进程数；近 CPU 核数即可（扫描是纯本地模拟）|
| `--force` | 否 | 忽略缓存重跑 |
| `--smoke` | 否 | 快速冒烟：2 count、1 seed、2 代 × 20 局 |

运行规模 = `len(source) × len(norms 或 residents) × len(counts) × len(seeds)`。

### 3.1 代数够不够？

`--generations 50` 是默认值，不能假设它对**任何**预算都够。检验方法：对少数
(source, norm) 用较大代数（如 `--generations 400`）跑对照，看曲线是否在 50 代前已平台化。

- 若 50 代时仍在明显上升 → 该配置下 50 代**不够**，须加代数或降低难度。
- 若 50 代前已平台 → 50 代可用。

> 实测经验：**高交互预算下平台通常很早到达**，50 代足够；**低交互预算下轨迹会持续
> 单调上升**（那正是 §0.1 的棘轮，不是真实动力学）。所以"没跑够代数"与"预算不足"
> 会在曲线上表现得很像，**必须先固定预算再判断代数**。

运行规模 = `len(source) × len(norms 或 residents) × len(counts) × len(seeds)`。
例：5 主导策略 × 10 norm × 13 个 count × 3 种子 = **1950 次运行**（N=20 的典型规模）。

> ⚠️ **单条频率轴（2026-09-10 定稿，2026-09-11 移除方向维度）**
> 扫描的是一条频率轴：在 count=k 时种群里是 k 个候选策略 + (N−k) 个 norm，
> 所以 count=1..N−1 已覆盖整条频率轴。
> **不再保留"方向"这一维度**——早期实现曾额外跑一条反向曲线，但实测 1050 个 cell
> 的 |dirA−dirB| 均值 0.089 ≈ 同方向内 seed-sd 0.093，即反向曲线与正向曲线的差异
> 与"同方向换 seed"的噪声同量级，不构成独立实验，故该维度已整体删除。
> 候选策略**自身是否可被入侵**由固定概率 benchmark 直接测量，见 §4。
> 因此扫描是**单向**的：只测"演化策略→norm"。反向（"norm→演化策略"）没有直接测过，
> 写论文时不得声称已测；只能从高频起步时演化策略守得住来间接推断。

> ⚠️ 输出目录命名建议**同时带上 N 与噪声标记**（如
> `n20_invasion_evolved_10000inter_b2_5seed_ae0p01_oe0p01`），避免与旧 N=100、
> 干净版、低预算版混淆；图文件名建议**沿用 RUN_DIR**。
> 有结果缓存（`--force` 强制重跑）；重复跑同参数会命中缓存秒回。
> 输出目录按 `$output/$LABEL/invades_$RESIDENT/n$count_seed$seed/invasion.json` 组织
> （把对手写进路径，避免不同 pair 互相覆盖）。
>
> **旧结果兼容**：历史上有两种目录布局——扁平
> `$output/$LABEL/$norm/...` 与带方向层级 `$output/$LABEL/evolved_invades_norm/$norm/...`。
> 新代码写入上面的新路径，但**读取时会依次回退到旧路径**，因此旧缓存仍然命中、
> 无需重跑（`existing_result_path` / `legacy_result_paths`）；
> 绘图脚本同样能读旧的嵌套 summary（`_norm_groups`）。
>
> ⚠️ **旧缓存不会自动作废**：缓存只按参数比对，**不**知道交互预算已提高。
> 复用旧目录前先核查每个 `invasion.json` 的 `config.interactions_per_generation`；
> 低于 10000 的结果必须丢弃（§0.1），换目录重跑。

---

## 4. 固定概率 benchmark

入侵扫描给的是**瞬态**；固定概率 benchmark 用 Schmid et al. (2023) 的方法给**稳定性**判据。
两套方法合起来才是完整结论（§0.3）。方法学细节见 `docs/fixation_benchmark.md`。

### 4.1 何时必须做

- 要声称某种**演化稳定性 / ESS**（入侵扫描本身不足以支持）；
- 要一个**与基准可比**的量（中性线 ρ=1/N，有明确判据）；
- 入侵扫描出现**内部共存平台**（没跑到 0 也没到 1），需要区分"真共存"与"瞬态没跑完"。

### 4.2 运行

**统一入口 `run-fixation-benchmark`**：三件事——两类型混合、跑至稳态后测收益、
由收益差曲线算 ρ。

```powershell
uv run run-fixation-benchmark `
  --candidate "<LABEL>=agent-type1=$OUT/<LABEL>`_seed$S/evolutionary.json" `
  --population-size 20 `
  --burn-in 12000 --measure 12000 --replicates 5 `
  --action-error 0.01 --observation-error 0.01 `
  --workers 5 `
  --output "results/quantitative_baseline/fixation/n20_evolved_$LABEL"
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--candidate LABEL=[AGENT_TYPE=]PATH` | （必填，当前只支持**一个**）| `.json` 自动取主导谱系代表，`.py` 直接读源码 |
| `--probes` | 全部 10 个 | 与之对峙的 norm 子集 |
| `--population-size` | `50` | **建议与入侵扫描同 N**，否则两套方法不可直接对照 |
| `--burn-in` / `--measure` | `10000` / `10000` | 跑至稳态的预热 + 测量交互数；**必须看 §4.3** |
| `--beta` | `1.0` | 选择强度（ρ 公式里）|
| `--replicates` | `5` | 每个组成的独立种子数（极端组成可能双模态）|
| `--action-error` / `--observation-error` | `0`（但**本 SOP 强制 ≥ `0.01`**）| 必须与入侵扫描同噪声，见 §0.4 |
| `--seed` / `--workers` / `--force` | `0` / `8` / 否 | — |

> ⚠️ `--candidate` **当前只能给一个**（多次传递会报错）；多策略扫描需**逐个**跑
> （每个候选一个独立进程，可并发）。
> 注意本步骤的**声誉不重置**，与入侵扫描相反（§0.5）。

### 4.3 stationarity —— 本方法的承重假设

ρ 由**收益差曲线**积分得到，所以曲线的**每一段都必须已收敛**，否则指数放大误差。

- runner 在测量窗口内把窗口对半切，比较两半的 d(k)：
  `stationarity.<probe>.max_half_to_half_gap`。**阈值 0.10**。
- 超过阈值会打印 `!! STATIONARITY WARNING`，并在图上标出漂移的组成。
- **处理：加大 `--burn-in` / `--measure`（如 40000/40000）并 `--force` 重跑。**
  加 burn-in 不一定收敛——有的策略声誉动力学本身就慢。

⚠️ **已知不可直接引用的结果**（引用前必须重跑或标注）：

- N=50 / `burn=measure=12000` 的旧结果里，**部分探针**未收敛（典型是 ALLC、ALLD、L8）；
  更有一个旧策略**全部 10 个探针**都漂移，其 ρ 整表不可用。
- 漂移集中在 **ALLC / ALLD**（无声誉记忆或全背叛，声誉分布慢），L1–L8 通常稳。
- 因此：**看到 ρ 表不要直接引用，先查 `stationarity` 字段。**
判据：只要有一个探针 gap > 0.10，该探针的 ρ 就不能作为独立证据；
  必要时升预算重跑，而不是换个说法引用。

### 4.4 怎么读输出

`fixation_benchmark.json` 里每个探针给出：

- `rho`：Traulsen–Hauert 固定概率；与 `config.neutral_fixation_probability = 1/N` 比。
  **`rho > 1/N` = 被选择推动且 favoured，`<` = 被抑制，`≈` = 中性。**
- `curve`：逐组成的 d(k) = π_候选(k) − π_探针(k)。上方为正，**过零说明有内部平衡**。
- `mean_payoff_difference` / `min` / `max`：曲线概览。
- `stationarity`：§4.3 的收敛审计。

> ⚠️ **ρ 是逐组成积分的有序列乘积**，不是“平均优势”。低频区即使 d(k)>0，只要中高频区
> 负拖累足够长，ρ 仍会回落到中性。所以**“不能入侵”不等于“更弱”**——
> 它可能只是势均力敌的稳定共存。论文不可把 ρ≈中性写成“劣于基准”。

---

## 5. 可视化

> 所有图都用**仓库常驻绘图器**生成，不要在 `tmp/` 里另写一次性绘图脚本；
> 报告里引用图时必须是**可重跑的长期命令**（§6）。

### 5.1 选哪个绘图脚本（先看这张表）

两个 plotter 读的都是同一个 `groups[一级键][二级键][count]` 结构，差别只在
**一级键/二级键的语义**和**图的形态**。按你的实验模式选：

| 你的实验 | 绘图脚本 | CLI | 图的形态 |
|---|---|---|---|
| `--norms`（候选 vs 规范 norm）| `plot_n100_invasion_count_sweep` | `plot-n100-invasion-count-sweep` | 每个 `source` **一个面板**；每个 norm **一条曲线**（§5.3）|
| `--residents A>B`（策略对策略）| `plot_pairwise_invasion` | `plot-pairwise-invasion` | 每个**无序对**一个**子图**（网格）；两个方向各一条曲线，**另加**一张支配阈值矩阵（§5.4）|
| 固定概率 benchmark | `plot_fixation_benchmark` | `plot-fixation-benchmark` | 左：d(k) 曲线；右：ρ vs 中性线（§5.5）|

> ⚠️ **前两者不是 schema 互斥的**——都能读对方的 summary，交叉使用**会出图但不报错**，
> 只是语义错位（面板切法 / 轴标签 / 图例含义全变）。具体症状与陷阱见 §5.4 末尾。
> **按实验模式选脚本，不要交叉使用。**
> 判断标准很简单：看结果目录里 `summary.json` 的 `experiment` 字段——
> `n{ N }_{noisy_}invasion_count_sweep` → §5.3；`n{ N }_pairwise_*` → §5.4。

### 5.2 汇总 JSON

`run-invasion` 结束自动写 `$output/summary.json`，含：
- `groups[$label][$norm][$count]` → `{runs, fixations, extinctions, mean_final_invader_frequency}`
- `sources`（每个主导策略的 agent_type/path/agent_id/root_lineage_id/fitness/code_sha256）
- `population_size` / `action_error_probability` / `observation_error_probability` /
  `initial_invader_counts` / `seeds` / `selection`
- `completed_or_cached_runs`（应 = 期望运行数）

> 绘图器从 `summary.json` **读 `population_size`**，所以标题里的 `N=` 是自动的，
> 无需改代码就能画任意 N。

### 5.3 入侵百分比曲线图

```powershell
# 图文件名建议沿用 RUN_DIR，确保带噪声后缀可辨识
uv run python -m experiments.analysis.plot_n100_invasion_count_sweep `
  --summary "$output/summary.json" `
  --output "README.assets/${RUN_DIR}.png"
```

产物：`${RUN_DIR}.png` + `.pdf`。图为若干列（**每列一个主导策略**）：
- x 轴 = 初始入侵份额，y 轴 = 均值最终入侵份额（>对角线 = 入侵成功/固定，<对角线 = 被淘汰）
- 每个 norm 一条曲线（L1–L8 + ALLC + ALLD）

> **噪声与 N 自动标注**：`plot_n100` 会读 `summary.json` 的
> `action_error_probability` / `observation_error_probability`（非 0 时写入标题）与
> `population_size`（标题的 `N=`）。因此**噪声版与干净版、不同 N 的图自然区分**。
> 图按 `initial_invader_counts` 读数：终值停在 0.5 附近而非 0/1 = **内部共存**（不是没跑完）。
> PNG 建议同时复制一份到 `$output/` 便于下游引用。

### 5.4 策略对策略图（`plot_pairwise_invasion`）

用于 `--residents A>B` 模式（§3(b)）产出的 summary。关键区别：这里的一级键是
**入侵者**、二级键是**居民**，而且**它是方向敏感的**——`groups[A][B]` 表示
"A 作为入侵者 vs B 作为居民"。若两个方向都跑了，同一个子图里会画出两条曲线，
两者相对对角线的**不对称**正是这张图要回答的问题（一方远高于对角线、另一方远低于
→ 严格成对支配）。这与 §5.3 的 norm 扫描（那条已废弃的"方向"维度）**不是同一回事**：
norm 扫描的反向曲线只是同一组成的重复实验；策略对策略的两个方向是两个**不同**的实验。

```powershell
# 默认：strategy_labels() 的全部无序对，展开成网格子图
uv run plot-pairwise-invasion `
  --summary "results/quantitative_baseline/invasion/<PAIRWISE_RUN_DIR>/summary.json" `
  --output "README.assets/<PAIRWISE_RUN_DIR>.png"

# 只画指定方向（可重复；A>B 表示 "A 作为入侵者"）
uv run plot-pairwise-invasion `
  --summary ".../summary.json" --output "..." `
  --pairs fourth>third --pairs fourth>thirdC
```

| 参数 | 必填 | 说明 |
|---|---|---|
| `--summary` | ✅ | **无默认值**（§5.3 的脚本有默认值，本脚本没有）|
| `--output` | ✅ | 主图落盘路径；同目录自动再写一份 `.pdf` |
| `--pairs A>B` | 否 | 只画给定的**有向**对，可重复；不给 = `strategy_labels()` 的**全部无序对** |

产物是**两个**文件：

1. `<output>`（`.png` + `.pdf`）—— 响应曲线网格。x = 初始入侵者份额（0–100%），
   y = 均值最终入侵者份额，虚线 = 无频率变化。每个标签一条曲线，
   子图标题 `"<A> vs <B>: invasion response"`，总标题写明噪声
   （`Pairwise invasion with 1% action + 1% observation error`，与 §5.3 一样从 summary 读）。
   图例里每条曲线标出阈值（`thr n=N` = 达到 50% 所需的最小初始份额；`no thr` = 99% 也达不到）。
   > ⚠️ ylabel 里那句 `Mean final invader share (3 seeds)` 的 "3 seeds" 是**硬编码**的
   > （`plot_pairwise_invasion.py` 的 `set_ylabel`），换 seeds 数不会自动更新——
   > 引用该图时请以 `summary.json` 的 `seeds` 为准。
2. `<stem>_threshold_matrix`（`.png` + `.pdf`）—— **支配阈值矩阵**。
   行 = 入侵者，列 = 居民（defender），格值 = 达到 50% 终值所需的**最小初始份额**（百分比）；
   `∞` 表示 99% 都达不到。**绿/低 = 容易入侵，红/高 = 难**。对角线画 `—`。
   > 阈值是**描述性**读数，不是演化稳定性判据。稳定性/ESS 必须用
   > `run-fixation-benchmark` + `plot-fixation-benchmark`（§4、§5.5）——
   > 模仿扫描的终点频率会给瞬态增长当成入侵的假阳性。

数据契约与兼容：

- `groups[入侵者][居民][count]`；
- 轴标签来自 `strategy_labels()`，它依次尝试 `strategies`（归档的旧 pairwise summary）
  → `sources` + `residents`（统一 runner）→ `groups` 的键。
  把 `residents` 也算进去是必要的：`groups` 只以**入侵者**为一级键，
  一个"只当居民、从不当入侵者"的策略否则会整个从矩阵里消失。
- `_has_pair()` 守门：**没有模拟过的方向直接不画**，绝不镜像、不推断另一方向。
  所以图上的空白 = 没跑过，不是中性。

> ⚠️ **陷阱：两个 plotter 能互相读对方的 summary，但语义会错位。**
>
> - **norm summary 喂给 `plot_pairwise_invasion`**：能跑通，且策略对语义恰好正确
>   （`b2_seed0` 当入侵者、`L1` 当居民）。但 `strategy_labels()` 会把
>   `sources`(5) + `residents`(10) 合成 **15 个标签**，默认展开 **105 个无序对 →
>   105 个子图**，其中只有 50 个有真实曲线（`_has_pair` 过滤后），其余 55 个只剩一条对角线。
>   要这么用**必须**逐个显式 `--pairs b2_seed0>L1`。
> - **pairwise summary 喂给 `plot_n100_invasion_count_sweep`**：只要**每个 source 都跑过
>   全部居民**，它的完整性校验 `len(sources)×len(norms)×len(counts)×len(seeds)` 会恰好成立
>   （实测 order-ablation：4×3×13×3 = 468 = `completed_or_cached_runs`），于是静默出图。
>   但面板按 `sources` 切、图例变成居民名、标题仍写 "Candidate strategy invades host"、
>   ylabel 写 "candidate share" —— **标签语义不对**。若某个 source 缺一个居民 key，
>   还会直接 `KeyError`。
>
> 结论：**按 §5.1 对照表选脚本**（或看 summary 的 `experiment` 字段），不要交叉使用。

### 5.5 固定概率图（`plot_fixation_benchmark`）

```powershell
uv run plot-fixation-benchmark `
  --summary "results/quantitative_baseline/fixation/<RUN_DIR>/fixation_benchmark.json" `
  --output "README.assets/fixation_benchmark_<RUN_DIR>.png"
```

产物：`<output>` + `.pdf`。两个面板：

1. **左：d(k) 曲线**（每个 probe 一条）——候选在组成 k 下的收益优势。
   在 0 上方 = 候选更适应；**与 0 相交 = 有内部平衡**（只在稀有时占优 → 可共存；
   只在常见时占优 → 双稳）。阴影带 = 多个 replicate 的 ±1 s.d.（宽带 = 该组成双模态，均值不稳）。
   若某 probe 未收敛，标题会加 `(!) ... raise --burn-in` 并在图上标出漂移组成（§4.3）。
2. **右：ρ 柱状图** —— 与中性线 `1/N` 对比。柱高于虚线 = 候选可入侵该 probe。
   误差须 = ρ 在收益差 ±1 s.d. 处的取值，**是警示（基态敏感性），不是置信区间**。

> 批量画多个策略时，**循环调用上面这条长期命令**（每个候选一次）；
> 不要在 `tmp/` 里另写一个绘图脚本（§5 开头的约定）。

---

## 6. 报告生产

本节规定如何把上面两步的产出写成一份可核验的报告（标杆：
`results/quantitative_baseline/invasion/EVOLVED_5SEED_10000INTER_INVASION_FIXATION_REPORT.md`）。

### 6.1 必备章节

| 节 | 内容 | 不要写什么 |
|---|---|---|
| 0. 结论速览 | 3–5 条可判定的话 | 不要模糊的“效果良好” |
| 1. 实验配置 | 两法参数**对照表** + 复现命令 | 不要省复现命令 |
| 2. 入侵扫描 | 曲线图 + **定性**归纳（阈值型/共存型/被反推）|
| 3. 固定概率 | ρ 图 + 与中性线的关系 | 不要把中性写成“劣于”|
| 4. 交叉验证 | 两法是否一致；不一致时归因 | 不要只报一致的那一半 |
| 5. Caveat | 未收敛探针、分辨不出的对手、未测方向 | **不要省略** |
| 6. 数据与图片清单 | 结果目录 + 图文件名 | 图必须可重跑 |

> **不需要数值表。** 报告只需能支撑结论的**汇总表**（如每个策略的判定计数）；
> 完整 ρ 矩阵、逐组成 d(k) 这类明细应留在 `fixation_benchmark.json` 里，
> 用路径引用即可，**不要抄进报告**（会与结果文件脱节且无法校验）。

### 6.2 图片必须内联

报告里的图一律用 `![](相对路径)`，**不要**用 `[链接](路径)` 的表格形式——
否则在 Markdown 预览里看不到图。相对路径基于**报告文件所在目录**（不是仓库根），
写错会变成死图。若报告在 `results/.../invasion/` 下，引用 `README.assets/` 需要 `../../../`。

### 6.3 完稿必做：链接校验

每张图、每个路径引用都可能因为改名/挪目录而失效。完稿前**逐条**验证：

```powershell
# 抽所有 ![]() / []() 目标，逐个 Test-Path（相对报告文件所在目录）
Select-String -Path "results/.../<REPORT>.md" -Pattern '!?\[[^\]]*\]\(([^)]+)\)' -AllMatches |
  ForEach-Object { $_.Matches } | ForEach-Object {
    $rel = $_.Groups[1].Value
    if ($rel -notmatch '^(#|https?://)') {
      $p = Join-Path (Split-Path "results/.../<REPORT>.md") $rel
      "  {0,-6} {1}" -f (Test-Path $p), $rel
    }
  }
```

必须 **0 个 `False`**。历史教训：曾在一份 README 里写了指向**不存在文件**的引用
（“详见汇总目录外的诊断记录”），是悬空引用；**写完引用后必须验证目标存在**。

### 6.4 生成脚本的可追溯性

报告里每个数字/图都应能追到一条**仓库内的常驻命令**：

- 图 → 常驻绘图器（§5）；
- 表/判定 → 常驻 runner 的 `summary.json` / `fixation_benchmark.json` 字段；
- 复现命令写进报告的 §1。

> ⚠️ **不要用 `tmp/` 里的一次性脚本生成报告里的数字。**
> `tmp/` 被 `.gitignore` 忽略，**不会进版本库** —— 这样做会让报告的表格
> 变成不可复现的孤儿（三个月后的自己只能重跑实验再手算）。
> 若某张表确实需要一层聚合，要么把它写进**常驻**脚本（`tools/` 或 runner 内），
> 要么**不放进报告**。

---

## 7. 验收标准

### 7.1 入侵扫描

1. **噪声强制**：`summary.json` 的 `config.action_error_probability` **与** `observation_error_probability` **均须 ≥ `0.01`**；任一为 `0` 即判定无效（全好稳态污染，见 §0.4）。
2. **交互预算**：每个 `invasion.json` 的 `config.interactions_per_generation` **须 ≥ `10000`**；低于此值的结果**整批作废**（§0.1）。
3. `summary.json` 的 `population_size` 与本次声明的 N 一致，且 `completed_or_cached_runs` 等于期望运行数（`sources × norms或residents × counts × seeds`）。
4. 全部运行 exit code = 0，无 traceback；worker 崩溃会被捕获并打印 `FAILED`。
5. `<label>/<resident>/n<count>_seed<seed>/invasion.json` 每个都有合法
   `final_invader_frequency ∈ [0,1]`，且 `invader_fixed`/`invader_extinct` 与它自洽。
6. 绘图脚本通过 `population_size`（读自 summary，须 ≥3）、`initial_invader_counts ∈ [1, N)`、
   `selection==synchronous_deterministic_payoff_imitation` 校验。
   > 这组校验**只在 `plot_n100_invasion_count_sweep`（§5.3）里**；
   > `plot_pairwise_invasion`（§5.4）**不做任何 schema 校验**，传错 summary 会静默出图。
   > 策略对策略实验的验收按 3–5 条走，第 6 条不适用。
7. `README.assets/${RUN_DIR}.png` 生成，且曲线形状与结论一致。
8. 若需与干净版对照，干净版必须**仅作对照**并在图/文档中标注"0 噪声——全好稳态，结论仅供对照"，
   不得作为主结论。

### 7.2 固定概率 benchmark

9. `fixation_benchmark.json` 的 `config` 里 `population_size` / `burn_in_interactions` /
   `measure_interactions` / `beta` / `replicates` / 两个噪声值均已记录，且噪声 ≥ `0.01`。
10. **stationarity 全过**：每个 probe 的 `stationarity.<probe>.max_half_to_half_gap ≤ 0.10`。
    凡超过的 probe，其 ρ **不得写入报告结论**（要么升预算 `--force` 重跑，要么在 caveat 里点名）。
11. `Neutral fixation probability == 1/N`，且 ρ 的解读引用了这条中性线。
12. 若与入侵扫描对照，**两者 N 与噪声必须相同**，否则不得直接对比（§0.3）。

### 7.3 报告

13. 报告里每个 `![]( )` / `[]( )` 路径均存在（**0 个 `False`**，见 §6.3）。
14. 报告里所有图均由**常驻绘图器**生成，复现命令已写入报告 §1；
    **不存在只靠 `tmp/` 脚本才能得到的数字或图**（§6.4）。
15. caveat 节已列明：未收敛的 probe、分辨不出的对手、未测的反向方向（§4.3、§3 的注、§0.3）。

---

## 8. 常见坑速查

| 现象 | 处理 |
|---|---|
| **忘了加噪声，曲线全部贴对角线** | 这是 0 噪声下的**全好稳态**（见 §0.4），结果无效。必须重跑：加 `--action-error 0.01 --observation-error 0.01` |
| **曲线全部跑到 100%、看起来很“能入侵”** | 先查 `config.interactions_per_generation`。低于 10000 时这是**假入侵**（噪声棘轮，§0.1），不是真实优势 |
| **重建了旧目录，结果没变** | 缓存只比参数，**不知道预算已提高**；旧目录里的低预算结果会原样命中。换目录重跑或删旧结果 |
| **ρ 表看起来“中性”但 stationarity 在报警** | 该 probe 的 ρ 不可用（§4.3）。加大 `--burn-in`/`--measure` 后 `--force` 重跑 |
| **`--candidate` 传了两次报错** | 固定概率 runner 当前**只支持一个**候选；多策略扫描要逐进程跑（§4.2）|
| **把 ρ≈中性写成“劣于基准”** | ρ 是逐组成积分的**有序列乘积**，中性 = 共存，不是劣势（§4.4）|
| **结果数远小于期望** | 检查 `--invader-counts` 是否超出 `1..N−1`（含 `0`/`N` 即非法）；是否 `--smoke` 遗留 |
| **换了 N 但图标题/网格不对** | 换 N 时`--invader-counts` 必须重给（默认网格只适合 N=20，§0.2）|
| 输出目录被覆盖 | 多策略用 `--output` 指定独立目录；`--source` 的 LABEL 必须唯一 |
| 图例显示 "LLM_v3" 误导 | `plot_cross_experiment_clusters` 的 `_experiment_label` 只认 `LLM_v2/v3` 前缀；本文入侵图用 `$LABEL` 列名，无此问题 |
| 重复跑不更新 | 缓存命中；加 `--force` 强制重跑 |
| worker 崩溃但主进程继续 | 属预期；查看 FAILED 行定位具体 (source,norm,count,seed) |
| **报告里的图/链接打不开** | 相对路径基于**报告文件所在目录**而非仓库根（§6.2）；完稿前必须跑 §6.3 的校验 |
| **报告里的数字无法复现** | 很可能用了 `tmp/` 里的一次性脚本算的（§6.4）。要么改由常驻脚本产出，要么从报告里删掉 |
| **用错绘图脚本**（策略对策略数据配 `plot-n100-invasion-count-sweep`，或反过来）| 两者都能读、都不报错，只是标签/面板语义错位；`plot_n100` 的完整性校验还可能在"每源都跑过全部居民"时**恰好通过**，掩盖错误。按 §5.1 对照表或 summary 的 `experiment` 字段选（§5.4 末尾有完整症状清单）|
| 成对图里出现大量只有对角线的空面板 | `plot_pairwise_invasion` 默认展开全部无序对；用 `--pairs A>B` 逐个指定。空白 = **没跑过**这个方向，不是中性结果 |
| 阈值矩阵整列 `∞` | 该居民到 99% 初始份额都守得住（≥50% 终值）；这不等于"不可入侵"——稳定性问题请用固定概率 benchmark（§4）|
