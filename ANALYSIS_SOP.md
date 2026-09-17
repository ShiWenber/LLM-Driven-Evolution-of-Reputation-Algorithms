# 分析 SOP：Fermi 演化实验的完整可视化分析流程

> 状态：✅ 已验证（2026-09-02，agent-type1 g100/10000inter/N16/upd4/5seed 全套产出）
> 适用范围：对一次已完成的 `V2EvolutionaryPopulation` 演化运行（schema v4，
> 含 `lineage_events`）做完整可视化 + 聚类 + 谱系分析，并把产物整理到统一布局。
> 底层工具细节见 `experiments/analysis/README.md`；本文只规定**执行顺序、命令、
> 命名约定与验收标准**，是可直接照做的标准操作流程。
>
> **§1–§9**：逐 seed 聚类/谱系可视化（回答"策略空间与谱系"）。
> **§6**：分两步 —— **§6.1** 跨 seed 联合聚类；**§6.2** 用 6.1 的 `run_id` 做**联合标签谱系树**
> （5 个 seed 共享同一套 cluster 颜色，可跨 seed 比较谱系）。§6.2 依赖 6.1，不可跳过。
> **§10**：逐目标私人声誉矩阵热图（回答"所有人对同一目标的看法在数值上是否一致"）。
> 三段互相独立，可只跑需要的部分。
>
> 📌 **§6.2 是 2026-09-14 新增**：此前 `plot_lineage` 的 `--cluster-run-id` 能力
> 虽已在代码中存在，但未写入 SOP，导致"跨 seed 可比的谱系树"实际无人产出。
> 注意 §4（逐 seed 独立聚类谱系树）与 §6.2（联合标签谱系树）是**两套不同产物**，
> 用文件名 `__joint` 后缀区分，不可混用。


---

## 0. 输入与前置条件

| 项 | 约定 |
|---|---|
| 结果根目录 | `results/quantitative_baseline/`（下称 `$OUT`）|
| 运行 label | 例如 `LLM_agent-type1_fermi_z_v3_g100_10000inter_N16_genreset_upd4`（下称 `$LABEL`）|
| 数据布局 | `$OUT/$LABEL\_seed<N>/evolutionary.json`，每个 seed 一份 |
| seed 列表 | 例如 `$SEEDS = 0..4` |

前置检查（不满足即停）：

1. **schema v4 + lineage**：每个 `evolutionary.json` 的 `config.schema_version == 4`
   且 `lineage_events` 非空（否则 `plot_lineage` 直接拒绝）。
2. **embedding 环境**：CUDA 可用（Code embedding `Salesforce/SFR-Embedding-Code-400M_R`），
   共享缓存 `results/.analysis_cache/strategy_analysis.sqlite3` 存在。
3. **ffmpeg** 可用（PCA 动画 MP4 需要；GIF 不需要）。
4. 命令统一用 `uv run python -m ...`，工作目录 = 仓库根。

预检示例（单行，PowerShell 可用）：

```powershell
uv run python -c "import json,glob; [print(p, json.load(open(p,encoding='utf-8'))['config'].get('schema_version'), len(json.load(open(p,encoding='utf-8')).get('lineage_events',[]))) for p in glob.glob('results/quantitative_baseline/LLM_agent-type1_fermi_z_v3_g100_10000inter_N16_genreset_upd4_seed*/evolutionary.json')]"
```

---

## 1. 逐 seed：全世代聚类演化（堆叠图 + PCA 动画）

对每个 seed 运行（**会自动选 K**，K 范围 2–20 取 silhouette 最大）：

```powershell
uv run python -m experiments.analysis.plot_strategy_cluster_evolution `
  --json "$OUT/$LABEL`_seed$s/evolutionary.json"
```

产物（写入该 seed 目录）：

- `plot_strategy_cluster_composition_per_generation.png`
- `plot_strategy_pca_evolution.gif`
- `plot_strategy_pca_evolution.mp4`

> 该步会触发所有代代码的 embedding（首次 ~1–3 min/seed），并把聚类 run 写入缓存
> （`clustering_runs` / `cluster_assignments` / `generation_cluster_stats`），后续步骤
> 可复用缓存，显著加速。

---

## 2. 逐 seed：最终代聚类 PCA 散点

```powershell
uv run python -m experiments.analysis.plot_strategy_clusters `
  --json "$OUT/$LABEL`_seed$s/evolutionary.json" `
  --out "$OUT/$LABEL`_seed$s/final_strategy_clusters_pca.png"
```

> 每个 seed 独立自动选 K（seed 间 K 不同是预期行为，勿跨 seed 比较 cluster 编号）。

---

## 3. 逐 seed：最终代聚类表 + 层次树状图

```powershell
# 3a. 文本聚类表（per-agent cluster、cluster sizes、语义名）
uv run python -m experiments.analysis.clustering.cluster_cli `
  --json "$OUT/$LABEL`_seed$s/evolutionary.json"

# 3b. 层次树状图（Ward，自动转 hierarchical）
uv run python -m experiments.analysis.plot_strategy_dendrogram `
  --json "$OUT/$LABEL`_seed$s/evolutionary.json" `
  --out "$OUT/$LABEL`_seed$s/strategy_dendrogram.png"
```

> ⚠️ 若单个 seed 因异常中断而漏产物，单独重跑该 seed 即可（embedding 命中缓存）。
> 完成后核对 seed 目录文件数一致（标准 10 项，见 §9 产物清单）。

---

## 4. 逐 seed：谱系可视化 + lineage 数据

```powershell
# 4a. 全分支视图（默认）：lineage_survival.png + lineage_tree.png
uv run python -m experiments.analysis.plot_lineage `
  --json "$OUT/$LABEL`_seed$s/evolutionary.json"

# 4b. 幸存者祖先视图：加 --survivors-only 与 --out-suffix
uv run python -m experiments.analysis.plot_lineage `
  --survivors-only --out-suffix "_survivors" `
  --json "$OUT/$LABEL`_seed$s/evolutionary.json"

# 4c. lineage 数据（JSON，供后续分析/报告引用）
uv run python -m experiments.analysis.lineage.build `
  --json "$OUT/$LABEL`_seed$s/evolutionary.json" `
  --out "$OUT/$LABEL`_seed$s/lineage.json"
```

### 命名整理（必须）

`plot_lineage` 输出名与仓库规范名不同，需重命名并清理多余文件：

```powershell
$d = "$OUT/$LABEL`_seed$s"
Rename-Item "$d/lineage_tree.png"            "full_birth_event_tree.png"     -Force
Rename-Item "$d/lineage_tree__survivors.png" "final_survivor_ancestry_tree.png" -Force
Remove-Item  "$d/lineage_survival__survivors.png" -Force -ErrorAction SilentlyContinue
```

保留 `lineage_survival.png`（存活区间图，两视图里只有它是唯一一份，不需要后缀版本）。

---

## 5. 汇总图：合作率进化曲线

`plot_evolution_curves` 画**单面板**：一个 run label 的逐 seed 合作率曲线 + 均值 + ±1 标准差带。
不限 agent 类型 —— 任何 run label 都能画，传 `--label` 即可。

```powershell
uv run python -m experiments.analysis.plot_evolution_curves `
  --label "$LABEL" --seeds 0 1 2 3 4 `
  --output "$OUT/analysis_<描述>/evolution_curves.png"
```

产物：`.png` + `.pdf`。

可选参数：`--title` 覆盖标题、`--color` 改颜色、`--no-pdf` 跳过 PDF、
`--results-dir` 覆盖结果根目录（默认 `results/quantitative_baseline/`）。

库函数同样可用：`plot_evolution_curves(label, seeds, ...)` 返回输出路径。

---

## 6. 跨 seed 联合聚类与联合谱系（同一策略空间对比）

> 本节含两步，**必须按顺序做**：先 6.1 得到联合 `run_id`，再用该 `run_id` 做 6.2。

### 6.1 联合聚类

将所有 seed 的 json 一起传入，**共享一次 embedding + PCA + K-means**（K 全局自动选），
保证 cluster 编号/名称跨 seed 可比：

```powershell
uv run python -m experiments.analysis.plot_cross_experiment_clusters `
  --json $OUT/$LABEL`_seed0/evolutionary.json $OUT/$LABEL`_seed1/evolutionary.json `
         $OUT/$LABEL`_seed2/evolutionary.json $OUT/$LABEL`_seed3/evolutionary.json `
         $OUT/$LABEL`_seed4/evolutionary.json `
  --out-dir "$OUT/analysis_<描述>/joint_<n>seed"
```

产物（`joint_<n>seed/` 下）：

- `cross_experiment_cluster_composition.png` — 各 seed 每代 cluster 构成对比
- `cross_experiment_strategy_space.png` — 共享 PCA 空间散点
- `*_seed<N>_cluster_composition.png` — 每 seed 单独构成图
- `source_manifest.json` — 输入路径清单
- 打印 `run_id`（← 记录到汇总报告，**并且必须传给 6.2**）

> ⚠️ 已知限制：脚本 `_experiment_label()` 只按 `LLM_v2`/`LLM_v3` 前缀区分类型，
> 非 `LLM_v2` 一律显示为 "LLM_v3"。agent-type1 run 的图例标签会有误导，解读时注意。

### 6.2 联合标签谱系树（用 6.1 的 run_id）

把 **6.1 的 `run_id`** 传给 `plot_lineage`，使 5 个 seed 的谱系树**共享同一套 cluster
颜色与名称**，从而可以跨 seed 横向比较谱系构成。

`plot_lineage` **没有 `--out-dir`**（输出固定写到 `--json` 所在目录），因此推荐用
**单进程脚本**直接调库函数、把产物写到汇总目录：

```python
# 单进程：一次加载 sklearn/scipy，循环 5 个 seed（避免 CLI 反复重载被中断）
# 下文的 $OUT / $LABEL / $RID 是本文档的占位符，用时替换为实际值。
from pathlib import Path
from experiments.analysis.clustering.cache import AnalysisCache
from experiments.analysis.plot_lineage import lineage_backtrack_tree, lineage_survival_plot
from experiments.evolution_log import F_CODE, F_LINEAGE_ID, F_POPULATION, K_TRAJECTORY, load_evolution_json

JOINT = Path("$OUT/analysis_<描述>/joint_<n>seed");  JOINT.mkdir(parents=True, exist_ok=True)
cache = AnalysisCache(None)                       # None ⇒ 默认共享缓存
code_labels, names = cache.get_clustering_run_labels("$RID")

for seed in range(5):
    data = load_evolution_json(f"$OUT/$LABEL_seed{seed}/evolutionary.json")
    code_by_lineage = {}
    for gen in data.get(K_TRAJECTORY, []):
        for a in gen.get(F_POPULATION, []):
            lid = a.get(F_LINEAGE_ID)
            if lid is not None and lid not in code_by_lineage:
                code_by_lineage[lid] = a.get(F_CODE, "")
    clusters = ({l: code_labels[c] for l, c in code_by_lineage.items() if c in code_labels},
                names, "stored Code embedding (joint run)", "$RID")
    lineage_survival_plot(data, JOINT / f"lineage_survival__joint_seed{seed}.png", clusters=clusters)
    lineage_backtrack_tree(data, JOINT / f"survivor_ancestry_tree__joint_seed{seed}.png",
                           clusters=clusters, survivors_only=True)
    lineage_backtrack_tree(data, JOINT / f"full_birth_event_tree__joint_seed{seed}.png",
                           clusters=clusters, survivors_only=False)
```

命令行等价形式（产物落在 seed 目录，需手工搬到汇总目录并改名）：

```powershell
uv run python -m experiments.analysis.plot_lineage --cluster-run-id $RID `
  --survivors-only --out-suffix jsurv --json "$OUT/$LABEL`_seed$s/evolutionary.json"
uv run python -m experiments.analysis.plot_lineage --cluster-run-id $RID `
  --out-suffix jfull --json "$OUT/$LABEL`_seed$s/evolutionary.json"
```

命名约定（必须带 `__joint` 后缀，与 §4 的独立聚类版区分）：

| 输出 | 汇总目录下的规范名 |
|---|---|
| 幸存者祖先树 | `survivor_ancestry_tree__joint_seed<N>.png` |
| 全分支出生事件树 | `full_birth_event_tree__joint_seed<N>.png` |
| 谱系存活区间 | `lineage_survival__joint_seed<N>.png` |

> 🚨 **最容易搞错的一点**：`plot_lineage` **默认对每个 seed 各自重新聚类**。
> 不传 `--cluster-run-id` 时产出的是**逐 seed 独立标签**版，
> **不能**跨 seed 比较 cluster 编号。§4 的产物就是这种独立版。
> 两套必须用文件名区分（§4 无 `__joint` 后缀，§6.2 有）。

> ⚠️ 不要在 PowerShell 里循环调 CLI：每次启动都要重新加载 sklearn/scipy/numpy，
> 5 seed × 2 次 = 10 次重载，实测多次 `KeyboardInterrupt`。用上面的单进程脚本。

覆盖度自查（应输出 0）：

```python
miss = sum(1 for code in code_by_lineage.values() if code not in code_labels)
```

---

## 7. 汇总 README（面向论文/协作的记录）

汇总目录 `$OUT/analysis_<描述>/README.md`，内容与数据来源：

| 章节 | 数据来源 |
|---|---|
| 摘要表（gen0/最终 coop、fitness、唯一策略数、lineage 事件数）| 逐 seed `evolutionary.json`（`trajectory[0]` / `trajectory[-1]` / `final_population` / `lineage_events`）|
| 每 seed 产物索引表 | §1–§4 产物，相对链接到 seed 目录 |
| 联合聚类参数与 cluster 名表 | SQLite 缓存 `clustering_runs`（`cluster_names_json`、`cluster_count`），run_id 来自 §6.1 打印 |
| 最终代联合构成（每 seed 各 cluster 人数）| SQLite `cluster_assignments`，`WHERE run_id=? AND generation=99` |
| 联合标签谱系树索引（5 seed × 3 图）| §6.2 产物，相对链接到 `joint_<n>seed/`；附各 seed 存活谱系数 / top-1 占比 / 存活根 origin |

从缓存读取元数据的参考语句：

```python
import sqlite3, json
con = sqlite3.connect("results/.analysis_cache/strategy_analysis.sqlite3")
# 最新 run
con.execute("SELECT run_id, cluster_count, cluster_names_json, created_at "
            "FROM clustering_runs ORDER BY created_at DESC LIMIT 1").fetchone()
# 某 run 的 gen99 每 seed cluster 构成
con.execute("SELECT experiment_id, cluster_id, COUNT(*) FROM cluster_assignments "
            "WHERE run_id=? AND generation=99 GROUP BY experiment_id, cluster_id", (run_id,))
```

---

## 8. 验收标准

1. 每个 seed 目录恰好含 **10 项**（8 图 + 2 数据文件）：
   `plot_strategy_cluster_composition_per_generation.png`、
   `plot_strategy_pca_evolution.gif`、`plot_strategy_pca_evolution.mp4`、
   `final_strategy_clusters_pca.png`、`strategy_dendrogram.png`、
   `lineage_survival.png`、`full_birth_event_tree.png`、
   `final_survivor_ancestry_tree.png`、`lineage.json`、`evolutionary.json`。
   ⚠️ 这 10 项里的谱系图是 **§4 的逐 seed 独立聚类版**（无 `__joint` 后缀），
   不要把它们当成联合标签版。
2. 全部命令 exit code = 0，无 traceback；`plot_lineage` 不报 schema < 4。
3. `lineage.json` 输出 `n_events`/`n_lineages` 与 `lineage_events` 长度自洽。
4. 汇总目录存在 `README.md` + 进化曲线 + `joint_*` 输出。
5. 缓存中能按 run_id 查到聚类名称与最终构成（用于报告追溯）。
6. **§6.2 联合标签谱系树齐备**：`joint_<n>seed/` 下恰好 `survivor_ancestry_tree__joint_seed<N>.png`、
   `full_birth_event_tree__joint_seed<N>.png`、`lineage_survival__joint_seed<N>.png` 各 n 份
   （n = seed 数），且**文件名带 `__joint` 后缀**以区别于 §4。
7. **联合标签覆盖度 = 100%**：每个 seed 的 `lineage_id -> code` 全部命中
   `--cluster-run-id` 的 `code_labels`（unlabelled = 0）。若有缺号，说明 6.1 的
   run 与 6.2 的 json 不匹配（例如 6.1 之后有新 run 覆盖了缓存），需重跑 6.1。
8. **颜色可比性**：5 个 seed 的 6.2 图例（cluster 名 → 颜色）完全一致，
   即同一 cluster 名在 5 张图里同色。

---

## 9. 常见坑速查

| 现象 | 处理 |
|---|---|
| PowerShell 中 `$LABEL_seed` 变量粘连 | 用反引号转义：`"$OUT/${LABEL}_seed$s/..."` 或 `` "$OUT/$LABEL`_seed$s/..." `` |
| 单 seed 产物缺失 | 单独重跑该命令；embedding/cache 命中，秒级~分钟级 |
| `plot_evolution_curves` 报 `Missing required runs` | `--seeds` 里有 seed 目录缺 `evolutionary.json`；从 `--seeds` 中移除该 seed |
| 想要跨 seed 可比的谱系树 | 必须给 `plot_lineage` 传 `--cluster-run-id`（§6.2）；不传就是逐 seed 独立聚类 |
| §6.2 里有谱系是灰色/无颜色 | 该谱系的代码不在 6.1 的标签里；检查 unlabelled 计数，必要时重跑 6.1 |
| 循环调 `plot_lineage` CLI 被中断（`KeyboardInterrupt`） | 每次 CLI 都重载 sklearn/scipy/numpy；改成**单进程脚本**调库函数（§6.2）|
| 聚类名缓存未命中导致额外 DeepSeek 调用 | 属正常；同代码复跑会命中（`--refresh-cluster-names` 可强制重命名）|
| `--json` 多文件时用 PowerShell 反引号续行 | 每行末加 `` ` ``，勿留尾随空格 |

---

## 10. 逐目标私人声誉矩阵热图（§1–§9 之外的新增可视化层）

§1–§9 回答的是"策略空间长什么样、谱系怎么分叉"。**本节回答的是另一个问题**：
最终种群在**私人声誉数值上**是否一致——不是"符号是否一致"，而是"给的是不是同一个数"。
两者互不依赖，可只跑本节。

> 仅支持 `agent_type == "agent-type1"`。type-2 让 LLM 自持任意内部状态，
> "私有声誉矩阵"没有良定义，`load_run` 会直接报错——**不要试图绕过**。

### 10.1 逐目标矩阵热图

**热图把声誉矩阵本身画出来**——这是判断"符号一致是否等于数值一致"的唯一直接证据。

```powershell
uv run python -m experiments.analysis.consensus.plot_private_reputation_matrix `
  --output "README.assets/private_reputation_matrix_$LABEL.png" `
  --seed 0 --interactions 1000 `
  --source N16v2_s0="$OUT/$LABEL`_seed0/evolutionary.json" `
  --source N16v2_s1="$OUT/$LABEL`_seed1/evolutionary.json"
```

- **每个 `--source` 一列**（建议 3–6 列；>8 列会太宽）。
- 每列两行：上行＝声誉矩阵 `[观察者, 目标]`，下行＝两两分歧矩阵 `[观察者, 观察者]`。
- 声誉色标固定 `[-1,1]`（红好/蓝坏/灰=无记录或自身），分歧色标固定 `[0,1]`；
  固定而非自动缩放，保证各列可横向比较。
- 命令会同时打印每列的 `D`。

### 10.2 读图配方（判定"数值是否一致"）

1. 上行**行内是否纯色**？纯色 ⇒ 该 agent 对所有目标一个态度。
2. 纯色的行**彼此颜色是否相同**？不同 ⇒ 存在**观察者宽严偏移** ⇒
   **符号一致但数值不一致**（这是最常见的情形）。
3. 下行红色有多深？深红格 = 该对观察者已严重分歧（`> 0.50`）。
4. 上行右侧 `observer mean` 条形图的跨度 = 各观察者一贯宽严的极差。
5. 列下方的 `D` 把 1–3 合并成一个数（`D = 0` 完全一致；越大越分裂）。

> ⚠️ **不要只看"符号是否一致"下结论**。符号级判断对观察者固定偏移**完全免疫**，
> 而演化种群恰恰是这种形态，所以本模块**不计算它们**，只输出一个数值指标 `D`。
> 实测出现过"符号全一致"同时 `D = 0.34`。

> ⚠️ **白 ≠ 灰**：声誉矩阵里白 = 分数恰为 0.0（即 GOOD 阈值，有意义的取值），
> 灰 = 没有记录（对角线／未观测）。分歧矩阵里白 = 零分歧。看错会得出相反结论。

### 10.3 本节验收标准

1. 命令 exit code = 0；不出现 `agent_type` 报错。
2. 矩阵热图 `.png` + `.pdf` 均生成。
3. 命令打印的每列 `D` 与图上列下方一致
   （可用 `--seed 0 --interactions 1000` 逐列核对）。
4. 若某列显示 `no shared targets observed`，说明没有任何两个观察者共同评过同一目标，
   `D = NaN`，属正常。
5. 测试全绿：`uv run pytest tests/test_private_reputation_consensus.py -q`

### 10.4 本节常见坑

| 现象 | 处理 |
|---|---|
| `load_run` 报 `agent_type='agent-type2'` | 该 run 不适用本节；换 type-1 run 或跳过 |
| `--condition` 设了但条件没变 | `--condition` **只是标签**；用 `--action-error` / `--observation-error` / `--observability` 真正指定 |
| 某列整片 `D ≈ 0` | 数值上完全一致（饱和稳态 ），属正常 |
| 图太宽/数字看不清 | 减少 `--source` 列数；`n > 12` 时自动关闭格内标注（`--annotate` 可强制开启） |

