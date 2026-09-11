# 入侵分析 SOP：主导策略 vs Leading Eight 的入侵扫描

> 状态：✅ 已验证（2026-09-08，agent-type1 N=12/3seed 主导策略注入 N=100 宿主）
> 更新：🔒 **噪声强制**（2026-09-08）——入侵实验**必须带噪声**，`--action-error` /
> `--observation-error` 不再是可选项，须 ≥ `0.01`（推荐 1%）。
> 更新：🔓 **移除方向维度**（2026-09-11）——实验只有一条频率轴，不再区分单/双向；
> 结果目录与 summary schema 均已扁平化（旧结果仍可读取，见 §3）。
> 适用范围：对已完成的 `V2EvolutionaryPopulation` 演化运行（schema v4）中提取出的
> **主导策略**（Evolution 出的 norm），做其对 8 个 Leading Eight 基准 norm 的
> **入侵百分比扫描**，并生成可视化。
> 与 `ANALYSIS_SOP.md` 相互独立——那份只覆盖演化运行本身的聚类/谱系/曲线,
> 本文只规定**入侵对抗**部分。底层原语见 `experiments/analysis/invasion/core.py`、
> `experiments/analysis/invasion/run_n100_invasion_count_sweep.py`。

---

## 0. 关键设计约定（必须先理解）

**宿主种群规模固定 = 100**，不同演化实验的种群规模一致无关紧要：
- 入侵实验看的是**入侵百分比**（invader count / 100），**不需要**与演化时的种群规模（如 N=12 / N=16 / N=50）匹配。
- 演化 run 里提取出的策略代码（主导策略）只是作为 `--source` 的 `code` 注入 N=100 宿主；
  二者唯一的耦合点是主导策略的提取逻辑，与宿主规模无关。

因此，**不要**因为演化规模是 N 就新建一个宿主规模=N 的入侵脚本（例如 `run_n12`）——那是错误的。
正确的做法是复用 `run_n100_invasion_count_sweep.py`（宿主恒为 100），把任意演化规模的主导策略作为
`--source` 传入。

**🔒 入侵实验必须带噪声（强制约定）**：`--action-error` 与 `--observation-error` **都不能省**，
且各自**必须 ≥ `0.01`**（推荐 `0.01`，即 1%）。原因（重要，写论文时须引用这一动机）：
- 在 **0 噪声**下，连续声誉会收敛到"全好声誉稳态"（reputations clamp 到 +1.0），各 norm 与演化策略
  的 `observe`/`decide` 分支几乎一致 → 行为趋同 → 入侵曲线全部贴合"无频率变化"
  对角线（README §10 中 `agent-type1` 的中性结果即源于此）。
- 加上噪声后，声誉**不再稳定锁死在全好值**，隐藏的选择差异才显现出来；此时
  演化策略对 norm 的**真实优势/劣势**才可观测（README §11 证明同策略在加噪后从对角线破出）。
- 因此：**不带噪声的入侵结果会被全好稳态污染，结论不可信**。这是本 SOP 的硬性前置条件，
  任何不带噪声的入侵扫描都应视为无效产出。

**连续值声誉基线**：L8 各 norm（`experiments/v2_quantitative/baselines.py`）使用**连续值声誉**
（`dict[int, float]`，初始 `0.0`，按 `STEP=0.333` / `BIG_STEP=0.5` 增减并 clamp 到 `[-1,1]`）；
`Good/Bad` 仅作为 `>0` / `≤0` 的阈值判断条件，写入的仍是连续数值。评估时须理解这一"软声誉"语义。

**声誉跨代重置（2026-09-09 明确）**：每个演化/入侵周期的 `reset_for_generation()` 现在会将整个
`reputations` 矩阵重置回初始 `{agent_id: INITIAL_REPUTATION}`（即 `{agent_id: 0.0}`），
**不再**跨代累积声誉。受影响位置：`experiments/v2_quantitative/agent.py`、
`experiments/v2_quantitative/agent_full.py`、`experiments/analysis/invasion/core.py`。
行为由 `tests/test_verify_reinit.py` 固定（断言重建后 `reputations == {aid: INITIAL_REPUTATION}`）。
这保证了每代从"中性声誉"起步，避免跨代声誉残留影响选择压力。

---

## 1. 输入与前置条件

| 项 | 约定 |
|---|---|
| 演化结果 | `results/quantitative_baseline/$LABEL_seed<N>/evolutionary.json`（schema v4，必须含 `final_population` + `lineage_events`）|
| 宿主规模 | 恒为 `POPULATION_SIZE = 100`（`run_n100` 内硬编码，勿改）|
| norm 集合 | `NORMS = (*LEADING_EIGHT, "ALLC", "ALLD")`（8 个 Leading Eight + 两个无条件基准，`core.py`）|
| CLI 入口 | `run-n100-invasion-count-sweep`（run）+ `plot-n100-invasion-count-sweep`（plot） |

> legacy schema v3（无 `lineage_events`，如 `LLM_v3_*` agent-type2 run）：`load_representative_from_path`
> 会自动回退到"按相同 code 分组、取最大组、组内最高 fitness"的代表选取（`_representative_without_lineage`）。
> 此时 `lineage_id` / `root_lineage_id` 记为 `-1`，报告中不应引用谱系字段。

前置检查：

1. schema v4 + `lineage_events` 非空（`load_representative_from_path` 需要它重建谱系）。
2. `.env` 中 DeepSeek key **不需要**——入侵对抗全是本地模拟（无 LLM 调用）。
3. 命令统一 `uv run python -m ...`，工作目录 = 仓库根。

---

## 2. 主导策略提取（每 seed 一个代表）

入侵的"来源"是演化实验最终种群的**主导策略**。现成的提取函数：
`experiments/analysis/invasion/run_n100_invasion_count_sweep.py::load_representative_from_path`
`experiments/analysis/invasion/run_n100_invasion_count_sweep.py::parse_sources`

提取规则（固定，勿事后更改）：
- 对每个 seed 的 `final_population`，按 `lineage_events` 把每个 member 归属到 **root lineage family**；
- 取**成员最多的家族**；若并列取 root_lineage_id 最小者；
- 在该家族内取 **fitness 最高**（并列取 agent_id 最小）的成员作为该 seed 的主导策略代表。

这样每个 seed 给出一个 `EvolvedSource`（含 code + shasum），作为入侵扫描的 `--source`。

---

## 3. 运行入侵扫描（宿主 N=100）

`run_n100` 的关键参数（`--source` 必填，噪声**必填**，其余可选）：

```powershell
uv run python -m experiments.analysis.invasion.run_n100_invasion_count_sweep `
  --output "results/quantitative_baseline/invasion/$RUN_DIR" `
  --source "<SRC_LABEL>=agent-type1=$OUT/<LABEL>`_seed$S/evolutionary.json" `
  --action-error 0.01 --observation-error 0.01 `
  --workers 48
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--output` | `.../n100_invasion_count_sweep` | 结果目录；多策略时需指定独立目录，避免覆盖现有全 norm 扫描 |
| `--source LABEL=AGENT_TYPE=PATH` | （必填）| 每个主导策略一份；`LABEL` 唯一，如 `seed0_coop` |
| `--norms` | 全部 10 个（L1–L8 + ALLC + ALLD） | 参与对比的 norm 子集 |
| `--invader-counts` | `(1,5,10,…,99)` | 初始入侵者数量（宿主=100 时即百分比）|
| `--seeds` | `(0,1,2)` | 每个 (source,norm,count) 重复的随机种子数 |
| `--generations` | `50` | 每轮博弈代数 |
| `--interactions` | `1000` | 每代随机配对局数 |
| `--fitness-interactions` | `200` | 计入适应度的最后局数（其余为 burn-in）|
| `--action-error` / `--observation-error` | **必填**，须 ≥ `0.01` | 动作/观测噪声；推荐 `0.01`（1%）。<br>**不可为 0**——0 噪声会落入全好稳态掩盖差异（见 §0）|
| `--smoke` | 否 | 快速冒烟：1 norm、2 count、1 seed、2 代 × 20 局 |

运行规模 = `len(source) × len(norms) × len(counts) × len(seeds)`。
例：3 主导策略 × 10 norm × 13 百分比 × 3 种子 = **1170 次运行**。

> ⚠️ **单条频率轴（2026-09-10 定稿，2026-09-11 移除方向维度）**
> 扫描的是一条频率轴：在 count=k 时种群里是 k 个候选策略 + (100−k) 个 norm，
> 所以 count=1..99 已覆盖整条频率轴。
> **不再保留"方向"这一维度**——早期实现曾额外跑一条反向曲线，但实测 1050 个 cell
> 的 |dirA−dirB| 均值 0.089 ≈ 同方向内 seed-sd 0.093，即反向曲线与正向曲线的差异
> 与"同方向换 seed"的噪声同量级，不构成独立实验，故该维度已整体删除。
> 候选策略**自身是否可被入侵**由固定概率 benchmark 直接测量，见
> `docs/fixation_benchmark.md`。

> ⚠️ 输出目录命名建议**带上噪声标记**（如 `..._ae0p01_oe0p01`），避免与干净版混淆；
> 图同样加后缀（如 `n100_noisy_invasion_count_sweep_ae0p01_oe0p01`）。
> `run_n100` 有结果缓存（`--force` 强制重跑）；重复跑同参数会命中缓存秒回。
> 输出目录按 `$output/$LABEL/$norm/n$count_seed$seed/invasion.json` 组织。
>
> **旧结果兼容**：2026-09-11 之前的结果含一层方向层级
> （`$output/$LABEL/evolved_invades_norm/$norm/...`）。新代码写入扁平路径，但**读取时会
> 自动回退到旧路径**，因此旧缓存仍然命中、无需重跑（`existing_result_path`）；
> 绘图脚本同样能读旧的嵌套 summary（`_norm_groups`）。

---

## 4. 汇总与可视化

### 4a. 汇总 JSON

`run_n100` 结束自动写 `$output/summary.json`，含：
- `groups[$label][$norm][$count]` → `{runs, fixations, extinctions, mean_final_invader_frequency}`
- `sources`（每个主导策略的 agent_type/path/agent_id/root_lineage_id/fitness/code_sha256）
- `completed_or_cached_runs`（应 = 期望运行数）

### 4b. 入侵百分比曲线图

```powershell
# 图文件名建议沿用 RUN_DIR，确保带噪声后缀可辨识
uv run python -m experiments.analysis.plot_n100_invasion_count_sweep `
  --summary "$output/summary.json" `
  --output "README.assets/${RUN_DIR}.png"
```

产物：`${RUN_DIR}.png` + `.pdf`。图为 N 列（每列一个主导策略）：
- x 轴 = 初始入侵份额，y 轴 = 均值最终入侵份额（>对角线 = 入侵成功/固定，<对角线 = 被淘汰）
- 每个 norm 一条曲线（L1–L8 + ALLC + ALLD）

> **噪声自动标注**：`plot_n100` 会读取 `summary.json` 的 `action_error_probability` /
> `observation_error_probability`，非 0 时在标题写明（如 "N=100 invasion ability with
> 1% action error + 1% observation error"）。因此**噪声版与干净版标题自然区分**。
> 图按固定百分比集读数：e.g. `count=50`（50%）是"半群入侵"的关键阈值。
> PNG 建议同时复制一份到 `$output/` 便于下游引用（`Copy-Item README.assets/..._invasion_count_sweep.png $output/`）。

---

## 5. 关联：同质鲁棒性（不同于入侵扫描，谨慎引用）

与入侵扫描相邻但**目的不同**的评估是
`run_perturbation_robustness.py`（CLI：`run-perturbation-robustness`）：
- 它测的是**同质群体**（100 个相同策略）在 8 组 (action,observation) 扰动下的**收益保持率**，
  看候选是否鲁棒、是否优于 L8。
- 它**不是**入侵对抗（不混合两种策略），结论措辞遵循
  `STRATEGY_SUPERIORITY_STANDARD.md`。
- 入侵扫描（本文）测的是**混合群体**中谁吃谁，二者互补、勿混用。

---

## 6. 验收标准

1. **噪声强制**：`summary.json` 的 `config.action_error_probability` **与** `observation_error_probability` **均须 ≥ `0.01`**；任一为 `0` 即判定无效（全好稳态污染，见 §0）。
2. `summary.json` 的 `completed_or_cached_runs` 等于期望运行数（`sources × norms × counts × seeds`）。
3. 全部运行 exit code = 0，无 traceback；worker 崩溃会被捕获并打印 `FAILED`。
4. `<label>/<norm>/n<count>_seed<seed>/invasion.json` 每个都有合法
   `final_invader_frequency ∈ [0,1]`，且 `invader_fixed`/`invader_extinct` 与它自洽。
5. 绘图脚本通过 `population_size==100`、`selection==synchronous_deterministic_payoff_imitation` 校验。
6. `README.assets/${RUN_DIR}.png` 生成，且可读到 `assert 结论` 相关的曲线分离现象。
7. 若需与干净版对照，干净版必须**仅作对照**并在图/文档中标注"0 噪声——全好稳态，结论仅供对照"，
   不得作为主结论。

---

## 7. 常见坑速查

| 现象 | 处理 |
|---|---|
| **忘了加噪声，曲线全部贴对角线** | 这是 0 噪声下的**全好稳态**（见 §0），结果无效。必须重跑：加 `--action-error 0.01 --observation-error 0.01` |
| 结果数远小于期望 | 检查 `--invader-counts` 是否含 `0`/`100`（必须 1–99）；是否 `--smoke` 遗留 |
| 输出目录被覆盖 | 多策略用 `--output` 指定独立目录；`--source` 的 LABEL 必须唯一 |
| 图例显示 "LLM_v3" 误导 | `plot_cross_experiment_clusters` 的 `_experiment_label` 只认 `LLM_v2/v3` 前缀；本文入侵图用 `$LABEL` 列名，无此问题 |
| 重复跑不更新 | 缓存命中；加 `--force` 强制重跑 |
| worker 崩溃但主进程继续 | 属预期；查看 FAILED 行定位具体 (source,norm,count,seed) |
| C++ 图保存路径 | 输出目录需存在（脚本会自动创建）|
