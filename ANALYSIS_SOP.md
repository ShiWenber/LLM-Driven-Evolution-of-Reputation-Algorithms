# 分析 SOP：Fermi 演化实验的完整可视化分析流程

> 状态：✅ 已验证（2026-09-02，agent-type1 g100/10000inter/N16/upd4/5seed 全套产出）
> 适用范围：对一次已完成的 `V2EvolutionaryPopulation` 演化运行（schema v4，
> 含 `lineage_events`）做完整可视化 + 聚类 + 谱系分析，并把产物整理到统一布局。
> 底层工具细节见 `experiments/analysis/README.md`；本文只规定**执行顺序、命令、
> 命名约定与验收标准**，是可直接照做的标准操作流程。

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

## 5. 汇总图：5-seed 合作率进化曲线

`plot_evolution_curves` 是**双面板**（agent-type1 vs agent-type2）设计。若只有单 agent
类型的新数据，把两侧 label 都指向它即可得到单面板等效图；若确需双类型对比，则
`--agent-type2-*` 指向旧 run label。

```powershell
uv run python -m experiments.analysis.plot_evolution_curves `
  --agent-type1-label "$LABEL" --agent-type1-seeds 0 1 2 3 4 `
  --agent-type2-label "$LABEL" --agent-type2-seeds 0 1 2 `
  --output "$OUT/analysis_<描述>/evolution_curves.png"
```

产物：`.png` + `.pdf`。

---

## 6. 跨 seed 联合聚类（同一策略空间对比）

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
- 打印 `run_id`（← 记录到汇总报告，用于回查缓存）

> ⚠️ 已知限制：脚本 `_experiment_label()` 只按 `LLM_v2`/`LLM_v3` 前缀区分类型，
> 非 `LLM_v2` 一律显示为 "LLM_v3"。agent-type1 run 的图例标签会有误导，解读时注意。

---

## 7. 汇总 README（面向论文/协作的记录）

汇总目录 `$OUT/analysis_<描述>/README.md`，内容与数据来源：

| 章节 | 数据来源 |
|---|---|
| 摘要表（gen0/最终 coop、fitness、唯一策略数、lineage 事件数）| 逐 seed `evolutionary.json`（`trajectory[0]` / `trajectory[-1]` / `final_population` / `lineage_events`）|
| 每 seed 产物索引表 | §1–§4 产物，相对链接到 seed 目录 |
| 联合聚类参数与 cluster 名表 | SQLite 缓存 `clustering_runs`（`cluster_names_json`、`cluster_count`），run_id 来自 §6 打印 |
| 最终代联合构成（每 seed 各 cluster 人数）| SQLite `cluster_assignments`，`WHERE run_id=? AND generation=99` |

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
2. 全部命令 exit code = 0，无 traceback；`plot_lineage` 不报 schema < 4。
3. `lineage.json` 输出 `n_events`/`n_lineages` 与 `lineage_events` 长度自洽。
4. 汇总目录存在 `README.md` + 进化曲线 + `joint_*` 输出。
5. 缓存中能按 run_id 查到聚类名称与最终构成（用于报告追溯）。

---

## 9. 常见坑速查

| 现象 | 处理 |
|---|---|
| PowerShell 中 `$LABEL_seed` 变量粘连 | 用反引号转义：`"$OUT/${LABEL}_seed$s/..."` 或 `` "$OUT/$LABEL`_seed$s/..." `` |
| 单 seed 产物缺失 | 单独重跑该命令；embedding/cache 命中，秒级~分钟级 |
| `plot_evolution_curves` 缺另一类型 run | 双面板共用同一 label，或跳过该步 |
| 聚类名缓存未命中导致额外 DeepSeek 调用 | 属正常；同代码复跑会命中（`--refresh-cluster-names` 可强制重命名）|
| `--json` 多文件时用 PowerShell 反引号续行 | 每行末加 `` ` ``，勿留尾随空格 |
