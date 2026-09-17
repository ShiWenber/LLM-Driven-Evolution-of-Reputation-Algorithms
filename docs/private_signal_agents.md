# 私有 dataclass 信号代理

使用 `--agent-type agent-type2-signal` 启用。agent-type1 和旧 agent-type2
保留原有执行器、提示、观察范围和默认行为。新模式没有 `_ctx_opponent_id`。

## 规则接口

LLM 生成 `@dataclass class Signal` 与 `class LLMAgent`，字段含义由策略自行设计。
所有字段必须提供默认值或 `field(default_factory=...)`，框架直接调用 `Signal()`。
使用普通实例方法，构造函数为 `__init__(self): pass`。

```python
def observe(self, target_signal, target_action,
            partner_signal, partner_action, self_signal) -> Signal:
    ...

def decide(self, opponent_signal) -> bool:
    ...
```

框架为每位观察者保存独立的 `ID -> Signal` 表，包含对自身的信号。
两次角色对调的评价均读取同一时点的双方信号和自身信号；两次调用都成功才写回。
即使自身参与了对局，传入的三个参数也分别复制，不暴露对象别名。
`decide` 的输入修改不会写回存储。

本人参与对局的观察与第三方观察全部沿用原有分发机制；入侵和 fixation 也不排除本人。
允许比较信号内容；本设置不声称排除了经由自身经验产生的全部直接互惠。

## 执行边界

新执行器单独执行 AST 检查和运行时属性访问检查。规则不获得真实 ID、框架对象、
对象身份、反射、文件或网络能力。`self` 只用于调用实例辅助方法（可以使用下划线命名），不能保存额外状态。
信号支持有限数值、字符串、bool、None、嵌套 list/tuple/dict/set/frozenset 和 Signal。
跨策略的 Signal 类型不能混用；整数恰好与某个 ID 相等不会被拒绝。

当前运行时支持 `dataclasses` 的 dataclass/field/replace 和常用 `math` 函数，
以及常用内置函数、容器方法、条件、循环与推导式。使用普通 `@dataclass`，不使用继承、
特殊方法、方法默认参数或可变参数。`import random` 提供 random/uniform/randint/
randrange/choice；执行器使用框架随机流派生的独立种子，不以 ID 作为种子，
不暴露 seed/getstate/Random。构造时的协议检查不会消耗正式模拟的随机数。
字段写法为 `signal.field = expression`，字段增强赋值需展开。

执行预算为每次调用 20000 个插桩步骤，信号最多 4096 个数据节点、深度 32，
字符串长度最多 4096；源代码上限为 12000 字符。这些限制只适用于新变体，
旧 type1/type2 的限制未改变。该执行器提供研究所需的信息隔离和常见失控代码检查，
不是用于运行任意恶意 Python 的操作系统沙盒，也不保证拦截所有底层资源耗尽方式。

初始协议检查覆盖默认信号、四种动作组合、评价输出和决策输出。
运行时出现异常、非 bool 决策、非 Signal 评价或超限信号时明确抛错并停止该次评价，
不会静默改为背叛。生成失败沿用现有重试和计数流程；新模式的最终初始化回退
由种群 RNG 选择兼容的 ALLC 或 ALLD，记入 fallback 统计，不能视作成功生成的策略。

## 运行

演化通过现有入口启动，例如：

```powershell
.\.venv\Scripts\python.exe -m experiments.run_fermi_v3 --agent-type agent-type2-signal --seed 0 --gens 10 --population-size 15 --target-interactions 1000 --output-root results/private_signal
```

此命令会按项目现有配置调用 LLM。执行器与提示都在独立文件中，初始生成、普通变异、
小变异和 deliberate 变异均使用同一信号协议。跨代重新初始化信号；Fermi 续跑
从原有代际检查点恢复代码、适应度与谱系，不恢复代内记忆。

提供无需 LLM 的示例 `experiments/analysis/invasion/custom_strategies/private_signal_example.py`。
它用于检查调用链，不代表优于 leading eight。可直接分析示例或替换为演化日志路径：

```powershell
.\.venv\Scripts\python.exe -m experiments.analysis.invasion.run_invasion --source signal=agent-type2-signal=experiments/analysis/invasion/custom_strategies/private_signal_example.py --norms L1 ALLD --population-size 4 --invader-counts 1 2 --seeds 0 --generations 2 --interactions 40 --workers 1 --output results/private_signal/invasion_smoke

.\.venv\Scripts\python.exe -m experiments.analysis.invasion.run_fixation_benchmark --candidate signal=agent-type2-signal=experiments/analysis/invasion/custom_strategies/private_signal_example.py --probes L1 ALLD --population-size 4 --burn-in 40 --measure 80 --workers 1 --replicates 1 --output results/private_signal/fixation_smoke
```

入侵与 fixation 中，Signal 代理和原 type1 leading eight 可以直接混合竞争，
分别维护自己的表示。fixation 的预热与测量之间不会重置信号。
比较时仍需明确原有演化、入侵和 fixation 的收益参数、观察率及生命周期差异，
不能仅凭接口兼容就认为所有实验条件已经匹配。

## 记录与验证

新模式保留演化日志的公共结构：`self_reputation` 为 null，额外记录 `self_signal`、
`signal_schema`、`signal_runtime_errors`、`signal_rng_seed` 和 `signal_interface_version`。
元组、集合和任意键字典使用带类型标记的 JSON 数据，避免把自由信号伪装成标量。
原 type1/type2 的日志记录保持原样；标量声誉矩阵分析仍只适用于 type1。
入侵和 fixation 的新信号结果带接口版本，缓存会检查该版本。

`tests/test_signal_agent.py` 覆盖身份访问、信号隔离、自身评价、同时更新、类型协议、
错误行为、代际重置、两种演化方式、续跑、入侵、fixation，以及 type1 标量记录回归。
