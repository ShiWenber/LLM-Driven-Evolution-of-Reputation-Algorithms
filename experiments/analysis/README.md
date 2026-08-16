# 演化树与代码分析方案总结

本文档总结本仓库用于**策略代码聚类**与**演化谱系（进化树）构建**的完整分析方案，
涵盖数据来源、分析流水线、模块结构、数据格式、使用方法与已验证的关键发现。

---

## 1. 方案总览

研究问题：在 LLM 驱动的捐赠博弈演化实验中，最终种群的策略代码形成了哪些
「家族」，它们之间如何继承，以及占主导的策略究竟是**直接互惠**还是**间接互惠**。

方案由三部分组成，彼此解耦：

```text
┌─────────────────────────────────────────────────────────────────┐
│ ① 框架记录（数据来源）                                          │
│    experiments/v2_quantitative/population.py                    │
│    在每代生成时直接记录「继承 / 完全重写」关系 → schema v4      │
└──────────────────────────────┬──────────────────────────────────┘
                               │ evolutionary.json (lineage 字段)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ ② 代码分析（策略聚类）                                          │
│    experiments/analysis/clustering/pipeline.py                  │
│    本地 Code embedding → 去重 → K-means → LLM 家族标签          │
└──────────────────────────────┬──────────────────────────────────┘
                               │ family 标签
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ ③ 演化树（谱系构建 + 可视化）                                   │
│    experiments/analysis/lineage/build.py                        │
│    experiments/analysis/plot_lineage.py                         │
│    出生事件 → 谱系折叠 → 存活图 / 回溯树                        │
└─────────────────────────────────────────────────────────────────┘
```

关键设计原则：

- **不做事后推断**。继承关系由演化框架在生成时显式记录，而非用代码相似度反推。
- **先聚类再判家族**。使用本地 Code embedding 后做 K-means；
  由 DeepSeek 根据质心附近的代表代码生成语义家族名。
- **库代码零硬编码**。所有库函数纯参数化，CLI 的默认路径/输出目录仅限 `--selftest`。

---

## 2. 数据来源：框架层面的血统记录

演化框架 `experiments/v2_quantitative/population.py` 中的 `V2EvolutionaryPopulation`
在 Fermi 更新点（`_select_and_reproduce_fermi`）**直接捕获继承关系**。

### 2.1 记录的语义（`origin` 取值）

| `origin` | 触发时机 | 是否有父代 |
|---|---|---|
| `initial` | 第 0 代初始化 | 无（根） |
| `imitate` | Fermi 1-μ 路径：对角色模型 j 的代码做小变异 | 有（父 = j） |
| `independent_init` | Fermi μ 路径：独立 LLM 初始化（完全重写，无父参照） | 无（新根） |
| `mutate` | 遗留 tournament 路径：对幸存者的变异副本 | 有 |

### 2.2 记录的数据结构

每个 agent（`trajectory[*].population` 与 `final_population`）新增字段：

```jsonc
{
  "agent_id": 0,
  "lineage_id": 4,              // 每次出生事件分配、永不复用
  "parent_id": 3,               // 父槽位（origin=imitate/mutate 时）
  "parent_lineage_id": 3,       // 父血统（origin=imitate/mutate 时）
  "origin": "imitate",          // initial / imitate / independent_init / mutate
  "birth_gen": 1
}
```

顶层新增 `"lineage_events"`：完整出生事件日志（含已灭绝血统），
`config.schema_version` 升为 **4**。

> **重要**：旧数据为 schema v3，无血统字段。需用更新后的框架重跑演化才能拿到
> lineage 数据（见 §7）。

---

## 3. 代码分析：策略聚类

### 3.1 自适应聚类（`clustering/pipeline.py` / `clustering/cluster_cli.py`）

- **先按代码字符串去重**再聚类（避免多个 agent 逐字节相同导致的「重复点」
  病理：`silhouette` 高分与 `ConvergenceWarning`）。
- 低层原语 `cluster_codes` 使用本地 Code embedding，默认模型为
  `Salesforce/SFR-Embedding-Code-400M_R`，对去重后的代码批量推理并做 L2 归一化。
  CUDA 可用时使用 FP16，运行时保留 CPU FP32 fallback；超过 8192 tokens 的完整代码
  会分块编码并聚合，不静默截断。
- 使用 K-means 与 `silhouette` 自动选 K。每簇选取离质心最近的代表代码，
  所有簇通过一次 DeepSeek 请求统一生成简短语义名称。
- **数据边界**：embedding 全程在本地完成；聚类命名会把每簇最多两段完整代表代码
  发送给 `.env` 中配置的 DeepSeek 服务。
- 完整代码、跨实验出现位置、模型向量和 DeepSeek 名称默认缓存在
  `results/.analysis_cache/strategy_analysis.sqlite3`。内容与模型配置完全命中时不加载
  embedding 模型；可用 `--no-embedding-cache` 禁用。
- 同一 SQLite 还保存每次聚类运行的逐 agent/逐代 label、每代各簇数量和占比、聚类名称与参数，
  以及生成的 PNG/GIF/MP4 路径和文件元数据。后续可用 `run_id` 关联
  `clustering_runs`、`cluster_assignments`、`generation_cluster_stats` 和
  `analysis_artifacts`，进行跨实验分析。
- 高层 `cluster_strategies`：跨全部世代拟合一次全局 Code embedding + K-means；
  使用中心化 PCA 投影，使每代策略落在同一个二维平面。
- `clustering/cluster_cli.py` 只是消费上述原语的 CLI
  （打印最终代逐 agent 聚类表）。

### 3.2 已验证的关键发现

在 seed2 最终代（15 个 agent、6 个唯一代码）上，用
`clustering/cluster_cli.py` 复跑：

- silhouette 自动选 **K=2**（K=2: 0.3939 > K=3: 0.3497）。
- 两簇：`self+opp_id+state`（逐对手记忆，12 人）与
  `self+reputation_scores`（声誉制，3 人，agent 0/6/10）。
- 即：**逐对手策略占 80%，利用声誉信息的少数派以 agent 0/6/10 残存**，
  与手工读码结论一致。

---

## 4. 演化树：谱系折叠与可视化

### 4.1 两种节点语义

| 定义 | 节点 = ? | 规模（100 代 × N=15）|
|---|---|---|
| A：出生事件树 | 每次出生（含每次小变异）| ~1500 节点 |
| B：谱系森林（折叠）| 连续 `imitate` 链合并为一条谱系，`independent_init` 才断 | ~根 + 少量独立 init |

方案采用**底层无损（A）+ 可视化折叠（B）**：`lineage_events` 保留完整出生事件，
`build_lineage_tree` 事后折叠成谱系视图，两者都可用。

### 4.2 谱系构建（`lineage/build.py`）

`build_lineage_tree(data)` 输出：

- `parent_of` / `root_of`：出生事件的父关系与根映射（折叠模仿链）。
- `lineages`：折叠后的谱系，每条含 `birth_gen`、`death_gen`（最后一次出现在任何
  槽位的世代）、`origin`、`members`、`n_members`。
- `survivors`：每个最终幸存者的祖先回溯路径（leaf → root 顺序）。

### 4.3 可视化（`analysis/plot_lineage.py`）

| 图 | 内容 | 用途 |
|---|---|---|
| `lineage_survival_plot` | x=世代，y=谱系，横条=存活区间，颜色=家族 | 看存续/灭绝/独立重写注入 |
| `lineage_backtrack_tree` | 最终幸存者的祖先森林（边=父血统→子血统）| 看家族谱系分叉 |

颜色映射使用 `clustering.pipeline.summarize_cluster_names` 生成的家族标签。

---

## 5. 模块结构与职责

```text
experiments/analysis/
├── __init__.py              # 轻量 facade（无副作用，不 eager 导入重依赖）
├── plot_lineage.py          # 演化树可视化（matplotlib）
├── plot_strategy_clusters.py    # 最终代策略 SVD 散点图
├── plot_strategy_cluster_evolution.py  # 世代堆叠柱状图 + PCA 动图
├── make_figures.py
├── plot_agent2_schmid_invasion.py
├── plot_evolution_curves.py
├── clustering/              # 可复用聚类原语（sklearn，无 matplotlib）
│   ├── io.py                # 解析 evolutionary.json → 世代 population
│   ├── pipeline.py          # embedding / cluster_codes / LLM naming / cluster_strategies
│   ├── cli_args.py          # 各聚类入口共用的命令行参数
│   └── cluster_cli.py       # 最终代聚类 CLI（消费 pipeline）
└── lineage/                 # 纯分析包（零 matplotlib 依赖）
    ├── __init__.py          # 仅惰性暴露 build_lineage_tree（PEP 562 __getattr__）
    └── build.py             # 谱系树数据构建（纯逻辑）
```

职责边界：

- `clustering/`：**可复用聚类原语 + 聚类 CLI**，不引入 matplotlib。
- `lineage/`：**谱系数据构建**，不引入 matplotlib。
- `analysis/plot_*.py`：**画图职责**，消费 `clustering` / `lineage` 的接口。
- `analysis/__init__.py`：轻量 facade，`import experiments.analysis` 零副作用。

---

## 6. 数据格式（schema v4）——接口约定

`evolutionary.json` 的唯一契约定义在 **`experiments/evolution_log.py`**
（writer 与 reader 共用，避免字段名漂移）：

- **Writer 侧**（`experiments/v2_quantitative/population.py`，v2 `QuantitativeAgent`
  与 v3 `FullAgent` 共用同一 `run_evolution()` 输出路径）：
  - 记录由 `population_entry()` / `trajectory_entry()` / `lineage_event()` /
    `make_config()` 构建，经 `build_evolution_results()` 装配并自动盖章
    `schema_version`。
  - CLI runner 通过 `write_evolution_json()` 原子写盘（tmp + rename），
    落盘路径统一为 `evolution_json_path(output_root, label, seed)`
    → `<output_root>/<label>_seed<N>/evolutionary.json`。
- **Reader 侧**（`experiments/analysis/*`）：一律用 `load_evolution_json()`
  （可校验）读取，字段名引用 `F_*` / `K_*` 常量，不再硬编码字符串。

顶层结构：

```jsonc
{
  "trajectory": [ ... ],        // 每代 population 含 lineage 字段（见 §2.2）
  "final_population": [ ... ],  // 同上
  "lineage_events": [ ... ],    // 完整出生事件日志
  "config": { "schema_version": 4, "agent_type": "v2" | "v3", ... }
}
```

约定要点：

- `origin` 枚举：`initial` / `imitate` / `independent_init` / `mutate`
  （对应 `ORIGIN_*` 常量）。
- `config.agent_type` 只能是 `"v2"` 或 `"v3"`（`AGENT_TYPES`）。
- 增加字段不破坏 reader（reader 用 `.get`）；删除或**重命名**字段必须
  递增 `SCHEMA_VERSION` 并同步更新迁移感知的 reader。
- `evolution_json_path()` 的目录约定也可经 `analysis.paths.run_dir()` /
  `analysis.paths.evolution_json_path()` 复用。

---

## 7. 使用方法

### 7.1 重跑演化（生成 v4 数据）

```bash
# 5 代 smoke（快速端到端验证）
uv run python _smoke_lineage_5gen.py

# 完整 100 代 seed2（约 5-7 小时）
uv run python _run_fermi_3seed_100gen_v3.py 2
```

### 7.2 分析命令

```bash
# 最终代策略聚类表（silhouette 自动选 K）
uv run python -m experiments.analysis.clustering.cluster_cli \
  --json results/.../evolutionary.json

# 最终代策略散点图（Code embedding + 中心化 PCA）
uv run python -m experiments.analysis.plot_strategy_clusters \
  --json results/.../evolutionary.json --out results/.../clusters_pca.png

# 全世代策略聚类演化图（堆叠柱状图 + PCA 动图）
uv run python -m experiments.analysis.plot_strategy_cluster_evolution \
  --json results/.../evolutionary.json --k 10

# 谱系数据构建（含 selftest）
uv run python -m experiments.analysis.lineage.build \
  --json results/.../evolutionary.json --out results/.../lineage.json
uv run python -m experiments.analysis.lineage.build --selftest

# 演化树可视化（含 selftest）
uv run python -m experiments.analysis.plot_lineage \
  --json results/.../evolutionary.json
uv run python -m experiments.analysis.plot_lineage --selftest
```

### 7.3 库 API

```python
from experiments.analysis.lineage import build_lineage_tree
from experiments.analysis.clustering.io import load_generations
from experiments.analysis.clustering.pipeline import cluster_codes, cluster_strategies
from experiments.analysis.plot_lineage import lineage_survival_plot, lineage_backtrack_tree
```

---

## 8. 验证状态

| 组件 | 验证方式 | 状态 |
|---|---|---|
| 血统记录（框架） | 桩替换 LLM 的单元测试，断言 origin/parent/血统链 | ✅ |
| Code embedding 聚类 | 本地模型批量编码、归一化与 K-means 接口测试 | ✅ |
| 谱系折叠 + death_gen | `build --selftest` 合成数据断言 | ✅ |
| 可视化 | `plot_lineage --selftest` 出图成功 | ✅ |
| 端到端 | `_smoke_lineage_5gen.py`（真实 LLM，3.3 min） | ✅ |

---

## 9. 已知限制与后续

- **重跑成本**：完整 100 代需约 5-7 小时/seed + DeepSeek API 费用。
- **本地模型**：Code embedding 首次运行需下载模型；默认模型采用 CC BY-NC 4.0
  许可证，仅适用于非商业研究用途。
- **重跑随机性**：新数据的「直接/间接互惠比例」与旧数据可能略有差异（不同随机
  路径），但新数据带精确继承关系，结论更硬。
- **特征提取的容错**：LLM 代码的语法多样性较大，`extract_features` 解析失败时
  `parse_ok=False` 并返回全零特征，需在分析时留意。
- **待办**：正式投稿如需独立 `.bib`，可将 `thebibliography` 迁移为 BibTeX（框架已
  支持，见 `paper_zh/reference.bib`）。
