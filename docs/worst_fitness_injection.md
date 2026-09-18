# 最低适应度个体的独立 LLM 注入实验

独立入口：`python -m experiments.run_worst_fitness_injection`。不修改日常 `run_fermi_v3` 的默认机制，也没有恢复旧的 control/resume 入口。

## 两种政策

1. `fermi_gated`：每代 16 个不同学习者各抽一个其他榜样。Fermi 接受后，0.1 概率独立 LLM 初始化该学习者，否则以榜样代码和收益为条件生成子代。
2. `worst_per_generation`：每代独立抽一次 Bernoulli(0.1)，不要求任何 Fermi 接受。命中时，用全新 LLM 初始化程序替换本代评估中适应度最低的一个体；完全相同最低收益之间随机选一个。其余位置继续 Fermi 学习，接受后只做亲代条件改写。若最低适应度位置也被 Fermi 更新选中，独立初始化覆盖这个计划中的改写，实际只生成一个新程序并提交一次。

适应度取当代交互末 20% 的人均行动收益。最低者在更新前的完整群体中确定；所有计划同步提交，随后声誉清零。独立初始化不含亲代代码或亲代收益。

0.1 在两组中的分母不同。因 σ(x)+σ(−x)=1，均匀选择榜样且每个学习者访问一次时，每代 Fermi 接受数的条件期望为 N/2=8。故原机制的独立出生期望为 0.1×8=0.8/代，新机制为0.1/代。99次代际更新分别约79.2和9.9次独立出生。不能把该实验写成只改变替换目标的单因素实验。

## 配对与记录

两组各5个演化种子，复用此前官方模型组的5套初始程序。`prepare` 在不调用 API 的情况下，逐种子重放验证初代评估与原档完全一致。两组均通过 `sili` provider 使用环境中配置的 `deepseek-ai/DeepSeek-V4-Flash`（请求地址 `https://api.siliconflow.cn/v1`），每条运行 `llm_concurrency=4`、温度0.8、关闭thinking、4000输出token上限，N16、100代、每代10000交互、b3/c1、β5、行动及观察噪声1%、完全观察、异步独立抽对。

初代高合作意味着本实验测试维持、扰动和谱系更新，不能宣称从低合作初态涌现合作。新实验两组同期重跑，避免仅拿新机制与旧脚本归档比较。每个演化seed/代的游戏随机种子按固定规则生成并在两组间共享；Fermi使用独立于游戏的群体RNG，新政策的注入及最低收益并列抽样另用键控RNG。LLM输出本身仍是随机生成，并非配对子代程序。

记录包括API返回模型别名、token用量、出生代码哈希、被替换者原谱系/适应度/合作率、Fermi计划覆盖次数、完整轨迹、谱系及源码快照。服务别名不等于固定模型版本。生成失败会终止该条运行，不以默认策略代替；中间轨迹只用于审查，没有自动续跑模式。

## 命令

```powershell
.venv/Scripts/python.exe -m experiments.run_worst_fitness_injection prepare --provider sili --llm-concurrency 4
.venv/Scripts/python.exe -m experiments.run_worst_fitness_injection run --provider sili --workers 5 --llm-concurrency 4
.venv/Scripts/python.exe -m experiments.run_worst_fitness_injection status
.venv/Scripts/python.exe tools/analyze_worst_fitness_injection.py
```

后续 `prepare` 的默认输出目录为 `results/worst_fitness_injection_sili_c4_20260918/`；当前新批次从初代开始运行。`prepare` 拒绝覆盖已有目录，新的独立实验请给 `prepare/run/status` 指定同一个新的 `--output`。默认4并发请求/运行；run的`--workers`默认5，按seed分配进程。seed0–4同时运行，每个seed依次执行fermi_gated、worst_per_generation两种机制，同一seed的两条轨迹不同时请求。总请求并发上限为20。`execution.json`记录实际调度；分析器逐条检查provider、请求地址、模型、thinking参数及并发数4。

分析：末20代合作率为主要终点；另统计全程平均、≤0.3合作的代数、相邻代从≥0.8降至≤0.3的突降、最长低合作段、独立出生与谱系存活、功能规则签名。被替换者合作与群体合作的比较是描述性关联，不能据此证明一次注入导致崩溃或恢复。每条轨迹是一个重复，不把100代或16个个体当作独立样本。

## 2026-09-18 配置纠正

官方 DeepSeek、16并发的已完成结果保留并标注 WITHDRAWN。两次 sili 批次（16并发和8并发）均未完成，已按用户要求停止并删除全部运行目录；等待结果的论文更新进程和一次性脚本也已清理。没有完成的新结果，未替换论文第五章及图表。此次取消后，用户重新授权以sili、llm_concurrency=4、五个seed并发运行；新目录与被删除的两次批次区分。

## 五个 seed 并发重跑

seed0–4分别占用一个工作进程，两种机制在各自seed进程内按顺序运行，总计10条轨迹。20项机制、配置及seed分组调度测试通过；十条轨迹的初代已逐条验证与源记录一致。API审计记录每次请求的provider与并发数。
