# 入侵 runner 去冗余重构方案

> 状态：⛔ **已被取代（2026-09-11）**。冗余已按下文诊断的理由消除，但采用的不是
> 本文建议的"保留三个 runner + 抽出声明式 `GridSweep`"，而是**合并为单一入口**
> `experiments/analysis/invasion/run_invasion.py`。
>
> 为什么换方案：本文的诊断是对的（三个 runner 的 `main()` 编排几乎全是复制粘贴），
> 但把编排参数化仍会保留三份"描述自己的网格"。进一步检查发现**三个 flavour 的差异
> 可以完全消掉**——norm 就是一个 `BASELINES[norm]` 的 `EvolvedSource`，于是
> "候选 vs norm"与"候选 vs 任意策略"变成同一件事的两种参数，连 `kind` 分支
> （`Competitor.create` 里的 `if kind == KIND_NORM`）都不再需要。
> 实测：同一个复制器（采纳 model 的完整身份）在 norm 模式下与旧实现结果**逐位相同**
> （0 mismatch），而在策略对策略模式下修复了旧复制器的身份错配 bug。
>
> 最终结构：
>
> ```
> experiments/analysis/invasion/
> ├── core.py                        # 统一原语；norm_source()；一个复制器 + 一个对局循环
> ├── run_invasion.py                # ★ 唯一引擎 + 唯一 CLI（--norms / --residents）
> ├── run_n100_invasion_count_sweep.py  # 弃用 shim（~80 行，转调 run_invasion）
> ├── run_n100_invasion_custom_code.py  # 弃用 shim
> ├── run_pairwise_invasion.py          # 弃用 shim
> └── run_fixation_benchmark.py      # 独立方法（Schmid ρ），未受影响
> ```
>
> 旧入口点、旧路径布局、旧 summary schema 均保持可读（`legacy_result_paths` /
> `_norm_groups`），已验证能命中真实归档结果、无需重跑。
>
> 以下为原始方案，保留作为诊断记录。

> 目标：消除 `run_n100_invasion_count_sweep.py` / `run_n100_invasion_custom_code.py` /
> `run_pairwise_invasion.py` 三者的重复，且**行为完全不变**。
> 状态：方案（未实施）。测量数据见文末「附录 A：冗余测量」。

---

## 0. 一句话结论

三个 runner 里 **`main()` 的编排部分几乎全是复制粘贴**（`custom` 与 `sweep` 相似度
**0.81**，其中 `record()` 46 行**逐字节相同**）。真正属于各 flavour 的只有
**"任务怎么枚举"** 和 **"一个任务怎么算"** 两件事。

因此方案的核心是：**把编排抽成声明式的 `GridSweep`，让三个 runner 只描述自己的网格。**

---

## 1. 冗余的四个来源（已量化）

| # | 重复内容 | 位置 | 规模 |
|---|---|---|---|
| R1 | `record()` 闭包 + 缓存检查循环 + `ProcessPool` 分派 | `sweep.main` ↔ `custom.main` | 46 + 15 + 9 行，**逐字节相同** |
| R2 | 三层 `setdefault` 分组 + 求均值 | `sweep.write_summary` ↔ `pairwise.main` | 算法相同，仅变量名不同 |
| R3 | 从 `.py` 构建 `EvolvedSource` | `custom.load_custom_source` ⊂ `pairwise.load_source` | custom 是 pairwise 的**真子集** |
| R4 | `sha256(...).hexdigest()` | 三文件共 **7 处**；`pairwise` 已包成 `_sha()` | — |

另外 argparse 的 `--norms/--invader-counts/--seeds/--generations/--interactions/
--fitness-window-fraction/--workers/--force/--action-error/--observation-error`
十个参数定义 + 三项校验，在三处重复。

---

## 2. 目标结构

```
experiments/analysis/invasion/
├── core.py                          # 不变，+1 个 sha256_hex()
├── sweep.py                         # ★ 新增：声明式网格编排（~150 行）
├── run_n100_invasion_count_sweep.py # 638 → ~430
├── run_n100_invasion_custom_code.py # 221 → ~110
└── run_pairwise_invasion.py         # 423 → ~250
```

### 2.1 新增 `sweep.py` —— 四个导出

```python
def sha256_hex(text: str) -> str: ...

def load_strategy_source(label: str, agent_type: str, raw_path: str) -> EvolvedSource:
    """按后缀分派：.json → 主导谱系代表；.py → 直接读源码。
    取代 custom.load_custom_source（子集）与 pairwise.load_source（重复实现）。"""

@dataclass(frozen=True)
class GridSweep:
    """一次网格扫描的声明。任务恒为 (key_0, ..., key_n, count, seed)。"""
    banner: str
    key_names: tuple[str, ...]        # ("agent_type", "norm") | ("invader_label","resident_label")
    layout: str                       # 结果路径模板，如 "{agent_type}/{norm}/n{count}_seed{seed}/invasion.json"
    run_payload: Callable             # 模块级执行器（ProcessPool 要求可 pickle by reference）
    summary_name: str

def execute_sweep(sweep, tasks, args) -> list[dict]:
    """缓存检查 → 进程池 → 写结果 → 返回 rows。取代 R1 的全部三块。"""

def group_rows(rows, key_names, summary_name) -> dict:
    """按 key_names 分层聚合 + 求均值。取代 R2 的两处实现。"""

def add_sweep_arguments(parser, *, include_smoke=False, output_required=False) -> None: ...
def validate_sweep_arguments(args, parser) -> None: ...
```

### 2.2 关键设计点：路径与行字段**自动派生**

这是能把 savings 做大的原因。现状里三处 `result_path` 和各写一遍的 row dict
其实是同一个东西的不同拼法：

```python
# 现在（sweep）
return output / agent_type / norm / f"n{count}_seed{seed}" / "invasion.json"
rows.append({"agent_type": kind, "norm": norm, "initial_invader_count": count, ...})

# 现在（pairwise）
return output / label_a / f"invades_{label_b}" / f"n{count}_seed{seed}" / "invasion.json"
rows.append({"invader_label": la, "resident_label": lb, "initial_invader_count": count, ...})
```

改成声明后二者统一：

```python
LAYOUT_NORM   = "{agent_type}/{norm}/n{count}_seed{seed}/invasion.json"
LAYOUT_PAIR   = "{invader_label}/invades_{resident_label}/n{count}_seed{seed}/invasion.json"
```

```python
def result_path_for(sweep, task, output) -> Path:
    key, (count, seed) = task[:-2], task[-2:]
    return output / sweep.layout.format(**dict(zip(sweep.key_names, key)),
                                       count=count, seed=seed)

def row_for(sweep, task, result, status, path) -> dict:
    key, (count, seed) = task[:-2], task[-2:]
    return {**dict(zip(sweep.key_names, key)),
            "initial_invader_count": count, "seed": seed,
            "final_invader_frequency": result["final_invader_frequency"],
            "invader_fixed": result["invader_fixed"],
            "invader_extinct": result["invader_extinct"],
            "status": status, "path": str(path.relative_to(output))}
```

> ⚠️ `row_for` 输出的字段名必须与现状**完全一致**，否则 `summary.json` 的 schema 会变。
> 现有 `"agent_type"` / `"norm"` 与 `"invader_label"` / `"resident_label"`
> 恰好就等于 `key_names`，所以能自动对上——这也是选 declarative 方案的前提。

### 2.3 ⚠️ Windows spawn 约束（必须遵守）

进程池提交的函数**必须是模块级**（可 pickle by reference）：

```python
# sweep.py —— 内部调用，不能传 lambda 进 pool
with ProcessPoolExecutor(max_workers=args.workers) as pool:
    futures = [pool.submit(sweep.run_payload, p) for p in pending]
```

各 runner 保留自己的模块级 `execute()` 作为 `run_payload`。**不要**把它改成闭包。

---

## 3. 分阶段执行

每阶段结束都必须 `uv run pytest -q` 全绿 + 跑一次 smoke。

### Phase 0 —— 先补**特征化测试**（前置，不可跳过）

现状测试覆盖：`sweep` 3 个测试（**只测 `noisy_output` / `parse_sources`，
不碰游戏循环**），`custom` 与 `pairwise` **零覆盖**。没有这个就别重构。

新增 `tests/test_invasion_golden.py`，用**极小参数**固定随机种子并断言精确值：

```python
SMALL = dict(generations=2, interactions=20, fitness_window_fraction=0.25, seeds=[0])

def test_norm_sweep_golden(tmp_path):
    """固定 seed 下 final_invader_frequency 必须逐位可复现。"""
    result = run_one(source_readme_best, "L1", invader_count=10, seed=0, **SMALL)
    assert result["final_invader_frequency"] == <pinned>

def test_pairwise_golden(tmp_path): ...      # run_pair
def test_custom_code_golden(tmp_path): ...   # 与 norm sweep 结果必须一致（同一 run_one）
```

> 这三个 golden 值**在重构前采集**，重构后必须不变。这是唯一的正确性保证。
> 注意 `random.seed(1_000_003 + seed)` 的全局种子扰动——测试要串行跑。

**验收**：三个 golden 测试先通过（在未改动的代码上）。

### Phase 1 —— 抽 `sweep.py`（纯搬移，改行为=0）

1. 新建 `sweep.py`：`sha256_hex` / `GridSweep` / `execute_sweep` / `group_rows` /
   `add_sweep_arguments` / `validate_sweep_arguments` / `load_strategy_source`
2. 把 `sweep.main` 的 `record()`、缓存循环、pool 块、`write_summary` 的分组
   替换为对 `execute_sweep` / `group_rows` 的调用
3. `custom` 与 `pairwise` 同样替换

**验收**：Phase 0 的三个 golden 测试全绿；三个 CLI `--help` 输出**逐字符不变**
（用 `git stash` 前后对比 `--help` 文本）；`summary.json` 对同一输入**字节级可比**。

### Phase 2 —— 统一 loader

用 `load_strategy_source` 取代 `custom.load_custom_source` 与 `pairwise.load_source`。
`custom` 的 `parse_sources`（`LABEL=PATH`）与 `pairwise.parse_strategies`
（`LABEL=[TYPE=]PATH`）合并为一个 `parse_strategy_specs(values, default_type=...)`。

**验收**：golden 全绿 + `--source`/`--strategy` 的错误信息不变
（`test_n100_invasion_sources.py` 的 `test_explicit_source_requires_label_type_and_path`
与 `test_explicit_source_labels_are_unique` 必须仍然通过）。

### Phase 3 —— 收敛 `custom`（可选，需你拍板）

Phase 1–2 之后 `custom.py` 只剩一个 `main()`（~110 行），与 `sweep` 的差别仅剩：

| 差异 | 处理 |
|---|---|
| `--output` required vs 可选 | `add_sweep_arguments(output_required=...)` 已支持 |
| `--source LABEL=PATH` vs `LABEL=TYPE=PATH` | 统一 loader 后 `TYPE` 可有默认值 |
| 无 `--smoke` | `include_smoke=False` |
| banner 文案 | `GridSweep.banner` |

也就是**功能已被 `sweep` 完全覆盖**（`.py` 后缀分派后，`sweep` 既能收
`evolutionary.json` 也能收 `.py`）。

**但删掉 `custom.py` 会移除 CLI `run-n100-invasion-custom-code`**，
且它 18 小时前才写、可能仍在演进。**建议：暂不删，先保留为薄包装。**

**验收**：无（本阶段只做决策，不改代码）。

---

## 4. 收益预估（诚实版）

| 阶段 | 行数变化 | 净收益 |
|---|---|---|
| 现状 | 638 + 221 + 423 = **1282** | — |
| Phase 1+2 | 556 + 118 + 312 + **170**(新) = **1156** | **−126（−10%）** |
| 再加 Phase 3 | 556 + 0 + 312 + 170 = **1038** | **−244（−19%）** |

**行数不是主要收益。** 真正的收益是消除 R1：

> 今天 `record()` 的 46 行在 `sweep` 和 `custom` 里**逐字节相同**。任何人改缓存状态
> 命名、改进度打印间隔、改 row 字段——**必须记得改两处**，漏一处就静默产生
> 不一致的 `summary.json`。这类 bug 不会让测试变红（因为没测），只会让两个
> 实验目录的结果**不可直接比较**。

---

## 5. 明确**不要**合并的部分

| 差异 | 为什么保留 |
|---|---|
| `pairwise_update` vs `payoff_imitation_update` | `payoff_imitation_update` 复制 **learner** 身份；A-vs-B 场景**必须复制 model 身份**。合并即引入错误。（`pairwise.py` docstring 已记录此陷阱） |
| `run_one` vs `run_pair` 主体 | 相似度 0.63，差异是 kind 标签（`candidate/norm` vs `A/B`）、更新函数、来源元数据。强行参数化会得到一堆回调、可读性反而下降 |
| `_representative_without_lineage` | legacy schema-v3 回退，只对 `evolutionary.json` 有意义 |
| 各 flavour 的 `execute()` | Windows spawn 要求模块级可 pickle |
| summary 的 `sources` 段 | `evolved_source` 单源 vs `invader_source`+`resident_source` 双源，语义不同 |

---

## 6. 风险与回滚

| 风险 | 缓解 |
|---|---|
| 无测试保护 | **Phase 0 强制前置**，未补 golden 不动一行 |
| `summary.json` schema 漂移 | Phase 1 验收含"对同一输入结果可比"；`row_for` 字段名与现状逐一核对 |
| CLI 行为变化 | 三个 `--help` 前后文本比对 |
| 与进行中的工作冲突 | 三个文件均 untracked，`pairwise`/`custom` 仅 18h 新。**建议先让作者 commit，或本方案只在 `sweep.py` 新增、不动 `main()` 的对外行为** |
| 进程池 pickling 失败 | 保持 `run_payload` 为模块级函数；Phase 1 后用 `--workers 2` 实跑一次 |

**回滚**：`sweep.py` 是新增文件，Phase 1/2 的改动限于三个 runner 的
`main()` 内部。回滚 = 恢复三个文件 + 删 `sweep.py`。建议每阶段前
`git add`（哪怕不 commit）以便 `git checkout` 单文件回退。

---

## 7. 建议的执行顺序

```
Phase 0  补 golden 测试            ← 必做，先做
   ↓
Phase 1  抽 sweep.py（搬移）        ← 稳，收益最大
   ↓
Phase 2  统一 loader               ← 稳
   ↓
暂停 ── 交给作者 review / commit
   ↓
Phase 3  决定是否收敛 custom        ← 需领域判断，不由我定
```

**我不建议一次做完。** Phase 1 是纯机械搬移（行为不变），风险最低、收益最大，
单独做完就能消掉 R1 这个最危险的重复。Phase 2 次之。Phase 3 涉及删 CLI，
应当由了解 `custom_code` 用途的人决定。

---

## 附录 A：冗余测量

重复行数（≥5 行连续块，已归一化注释/空行）：

| 组合 | 重复行 | 占较小文件 |
|---|---|---|
| `custom` ⟷ `sweep` | **86** | **44%** |
| `pairwise` ⟷ `sweep` | 67 | 18% |
| `custom` ⟷ `pairwise` | 24 | 12% |

同名函数相似度：

| 函数 | 对比 | 相似度 |
|---|---|---|
| `main` | `custom` vs `sweep` | **0.81** |
| `main` | `pairwise` vs `sweep` | 0.35 |
| `run_pair` vs `run_one` | — | 0.63 |
| `load_source` vs `load_custom_source` | — | 子集关系 |

同函数名下**逐字节相同**的最大块：

| 块 | 行数 | 位于 |
|---|---|---|
| `def record(task, result, status) -> None:` | **46** | `custom.main` ↔ `sweep.main` |
| `try:` （`--source` 解析） | 15 | 同上 |
| `with ProcessPoolExecutor(...)` | 9 | 同上 |

测量脚本思路：三文件各做 `ast` 函数清单 + `difflib.SequenceMatcher`
（对 `splitlines()` 先剥注释/空行、折叠空白），取 `get_matching_blocks()`
中 `size >= 5` 的块求和。
