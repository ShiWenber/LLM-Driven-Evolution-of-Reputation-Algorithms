# Issues

> **状态更新（2026-06-03）**：本文档源自 5 月初的代码审计。当前除"问题 2"（直接互惠泄露）外，其余 4 项均已在提交前修复。下文按"是否仍存在"标注。

## 问题 1（致命） — ✅ 已修复

**位置**：`experiments/agents/code_agent.py:17`

```python
INITIAL_REPUTATION = 0.01
```

修复后改用 `INITIAL_REPUTATION = 0.01`，使 `> 0.0` 阈值也能在冷启动时通过。

## 问题 2（高） — ⚠️ 仍存在（已修复）

**原位置**：`experiments/evolution/mutation.py:135`（OpenAI 分支）与 165（Anthropic 分支）

**原问题**：
```python
"In my_history entries, use entry['action'] for your own action "
"and entry['partner_action'] for the other agent's action."
```

`partner_action` 是**直接互惠**信息通道，让 LLM 倾向在 `decide()` 中走 `if my_history[-1]['partner_action'] == 'donate': return True` 路线 —— 这是直接互惠，不是间接互惠。

**修复（2026-06-03）**：两处 system message 改为：
```python
"Focus on INDIRECT reciprocity: use observation-based reputation "
"(observation dict) to update and consult reputation scores. "
"Do NOT condition decide() on partner_action from my_history — "
"that would be direct reciprocity, which is not the target of this game."
```

⚠️ **注意**：此处修复后，所有**新**运行的 mutation 都不会再被正向引导使用 `partner_action`。但**已归档**的实验结果（`experiments/results/`）是在修复前产生的，需要重新跑才能用于 TSMC 投稿。

## 问题 3（高） — ✅ 已修复

**位置**：`experiments/agents/prompts.py:54`

`STRATEGY_INTERFACE` 已明确写 `float between -1.0 and 1.0`。

## 问题 4（中） — ✅ 已修复

mutation system message 已正确列出 `entry['action']` 与 `entry['partner_action']` 两个字段（2026-06-03 修复时一并覆盖）。

## 问题 5（中） — ✅ 已修复

`prompts.py:35-44` 已补充关于 `donor` 与 `recipient` 字段的说明。

---

## 总结

| 编号 | 问题 | 状态 | 备注 |
|---|---|---|---|
| 1 | 冷启动 `INITIAL_REPUTATION=0.0` | ✅ 已修 | `code_agent.py:21` = `0.01` |
| 2 | 直接互惠泄露 | ⚠️ 已修，需重跑 | `mutation.py:148-150 / 177-179` |
| 3 | 声誉尺度 `any float` | ✅ 已修 | `prompts.py:54` |
| 4 | `my_history` 字段不对称 | ✅ 已修 | 与问题 2 一起 |
| 5 | observation 字段说明 | ✅ 已修 | `prompts.py:35-44` |

## 下一步

1. **重跑实验**：所有归档结果在修复前产生，需要重跑 `Experiment 1-4` 并替换 `experiments/results/`
2. **样本量扩到 ≥ 10 seeds**（TSMC 投稿要求）
3. **至少加入 1 个 LLM family 对比**（如 DeepSeek + Claude 或 GPT-4o）
