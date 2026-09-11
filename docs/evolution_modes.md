# 锦标赛模式与 Fermi 模型说明

本文档说明本仓库中两种种群更新模式的核心区别：**锦标赛选择（Tournament）** 与 **Fermi 模仿过程（Fermi）**。

---

## 1. 总体定位

| 维度 | 锦标赛模式（Tournament） | Fermi 模型（Fermi） |
| --- | --- | --- |
| 核心隐喻 | 竞赛式筛选 + 变异 | Moran 过程式模仿学习 |
| 更新粒度 | 每代整体替换 | 每代多次单点更新 |
| 选择机制 | 精英保留 + 锦标赛 + 淘汰 | 基于 fitness 差异的 Fermi 接受概率 |
| 子代来源 | 对幸存者做全量 LLM 变异 | 对高 fitness 个体做小变异或独立重写 |
| 并行友好度 | 较低（历史实现串行） | 较高（job 可批量并行） |
| 代码入口 | `TournamentEvolutionRule` | `FermiEvolutionRule` |
| CLI 参数 | `--learning-method tournament` | `--learning-method fermi` |

---

## 2. 锦标赛模式（Tournament）

### 2.1 选择流程

每代执行一次完整的“筛选-替换”：

1. **精英保留**：按 fitness 排序，前 `elite_count` 个个体直接进入下一代。
2. **锦标赛补位**：从当前种群中随机抽取 `tournament_size` 个个体，取 fitness 最高者作为幸存者；重复直到幸存者数量满足 `N - num_eliminate`。
3. **淘汰**：剩余 `num_eliminate` 个个体被移出种群。
4. **生成子代**：对每个被淘汰的槽位，随机选择一个幸存者作为父代，调用 LLM 生成一个**全量变异**（`llm_mutate`，`mutation_kind="full"`）的子代。

### 2.2 关键特性

- **同步世代**：每代只发生一次种群替换，所有个体在同一轮交互后统一评估 fitness。
- **强变异**：子代是对父代代码的完整改写，变异幅度较大。
- **固定替换数**：每代恰好淘汰 `num_eliminate` 个个体，并生成等量子代。
- **父代随机**：被淘汰槽位的父代从幸存者中均匀随机抽取，不直接绑定到 fitness 排名。

### 2.3 代码位置

- 规则实现：`experiments/v2_quantitative/evolution_architecture.py::TournamentEvolutionRule`
- 调用入口：`experiments/v2_quantitative/population.py` 中 `_select_and_reproduce()` 的 tournament 分支
- 配置参数：`elite_count`、`num_eliminate`、`tournament_size`

---

## 3. Fermi 模型（Fermi）

### 3.1 选择流程

每代执行 `updates_per_gen` 次独立的“模仿-更新”事件（Moran 过程风格）：

1. **采样 learner**：从种群中无放回地随机抽取 `updates_per_gen` 个个体作为学习者。
2. **采样 role model**：对每个 learner，随机抽取一个**不同**的个体作为角色模型。
3. **计算接受概率**：

   $$
   P(i \text{ 模仿 } j) = \frac{1}{1 + \exp(-\beta \cdot (\phi_j - \phi_i))}
   $$

   其中 $\phi$ 是当前代的窗口 fitness，$\beta$ 是 `fermi_beta`。
4. **决定是否模仿**：
   - 若随机数 < 接受概率，则 learner 复制 role model 的代码；
   - 否则 learner 保持不变。
5. **变异决策**：在 accepted copy 的基础上，以 `mutation_rate_on_adoption` 的概率触发**独立 LLM 重写**（`llm_init`，无父代）；否则触发**小变异**（`llm_mutate`，`mutation_kind="small"`），即对 role model 代码做小幅修改。

### 3.2 关键特性

- **异步更新**：每代发生多次单点更新，种群在代内逐步变化。
- **概率性接受**：fitness 更高的个体更容易被模仿，但 fitness 更低的个体仍有一定概率保留原策略。
- **小变异为主**：默认以小幅修改为主，独立重写为辅。
- **slot 不变性**：Fermi 模式下 learner 的 `agent_id`（槽位号）保持不变，仅代码和 lineage 可能更新。

### 3.3 代码位置

- 规则实现：`experiments/v2_quantitative/evolution_architecture.py::FermiEvolutionRule`
- 调用入口：`experiments/v2_quantitative/population.py` 中 `_select_and_reproduce_fermi()`
- 配置参数：`fermi_beta`、`mutation_rate_on_adoption`、`updates_per_gen`

---

## 4. 核心区别总结

| 对比项 | 锦标赛模式 | Fermi 模型 |
| --- | --- | --- |
| 更新时机 | 每代一次批量替换 | 每代多次单点更新 |
| 选择逻辑 | 精英 + 锦标赛 + 淘汰 | Fermi 概率接受 |
| 子代数量 | 固定 `num_eliminate` 个 | 最多 `updates_per_gen` 个 |
| 变异幅度 | 全量改写（full） | 小变异（small）+ 独立重写（init） |
| 父代绑定 | 淘汰槽位 ↔ 随机幸存者 | learner ↔ 随机 role model |
| 保留机制 | 精英直接保留 | 未接受模仿的个体原样保留 |
| 并行粒度 | 子代间可并行，但历史实现串行 | job 天然可并行 |
| lineage 语义 | `origin=mutate` | `origin=imitate` 或 `origin=independent_init` |

---

## 5. 如何选择

- 若希望**强选择压力、快速收敛**，且接受每代较大幅度的策略改写，可选用 **Tournament**。
- 若希望**更平滑的演化动态、保留更多多样性**，且希望更好地利用并行 LLM 调用，可选用 **Fermi**。

两种模式在代码中通过 `learning_method` 切换，默认值为 `fermi`。
