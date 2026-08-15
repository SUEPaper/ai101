---
course: deep-learning
edition: industrial
title: 工具调用、结构化输出与 Agent
description: 从 function calling、JSON Schema、约束解码进入 ReAct 状态机、权限、幂等、超时、记忆、提示注入和任务级评测。
outline: [2, 3]
---

<DlLessonHero number="26" module="大语言模型" title="工具调用、结构化输出与 Agent" question="当模型从写文本变成调用有副作用的 API，哪些验证、权限和状态必须从模型外部强制执行？" duration="建议 340–450 分钟" :outcomes="['定义 tool schema、参数语义校验与受控执行接口', '区分 JSON 有效、schema 有效、业务有效和获权', '设计可恢复、幂等、预算受限的 Agent 状态机', '解释 ReAct、memory 与工具观测的上下文边界', '注入参数错误、超时、重复副作用、权限拒绝和提示注入并评价']" />

Agent 不是“多写几轮 prompt”，而是模型在外部状态机内提出动作，可信代码验证、授权、执行并记录结果。只要工具能发邮件、改数据库或控制设备，文本生成的可逆错误就变成真实副作用。

## 贯穿案例：维修工单助手

工具包括 `search_manual`（只读）、`get_sensor`（只读）、`create_work_order`（写入）和 `stop_machine`（高风险）。模型无权直接执行；控制器校验 schema、设备范围、用户权限、幂等键和预算，高风险动作需要人工确认。

<DlQuiz kind="先判断" question="模型输出了 schema-valid JSON，是否可以立即执行停机？" :options="['可以', '不可以，还需业务语义、权限、当前状态、风险和人工确认校验', 'JSON 永远可信', '只需 temperature=0']" :answer="1" explanation="结构有效只是最低层门禁。" />

<div class="dl-stop"><strong>停下来推一推 1</strong>为 `stop_machine(device_id, reason)` 列出语法、类型、存在性、租户、权限、状态、幂等和二次确认八层验证。</div>

## 1. Function calling 与 schema

Tool 定义应包含唯一名称、用途、严格 JSON Schema、单位/枚举、错误码、副作用、权限和超时。描述既是模型条件，也是攻击面；不要把密钥或不可见策略塞入描述。服务端永远重验参数，不信任模型生成的类型。

<ConstrainedJsonLab />

```python
def validate_stop(call, allowed_devices):
    required = {"device_id", "reason"}
    if set(call) != required: return False
    return call["device_id"] in allowed_devices and len(call["reason"]) >= 5

assert validate_stop({"device_id":"P-17","reason":"轴承温升超限"},{"P-17"})
assert not validate_stop({"device_id":"P-99","reason":"test"},{"P-17"})
```

<DlQuiz kind="结构化输出" question="Grammar-constrained decoding 能保证什么？" :options="['输出符合语法/结构约束', '设备 ID 一定存在', '调用一定获权', '结果一定正确']" :answer="0" explanation="语义和权限必须由外部代码验证。" />

## 2. ReAct 与显式状态机

ReAct 交替产生 reasoning/action/observation，使模型根据工具结果更新计划。工程上应把它实现为有限状态：`plan→validate→authorize→execute→observe→commit/abort`，每步有超时、预算和持久日志。内部 reasoning 不必暴露给用户，关键是动作与证据可审计。

<ToolStateMachineLab />

```python
ALLOWED = {"PLAN":{"VALIDATE"}, "VALIDATE":{"AUTHORIZE","ABORT"},
           "AUTHORIZE":{"EXECUTE","DENIED"}, "EXECUTE":{"OBSERVE"},
           "OBSERVE":{"COMMIT","PLAN"}}

def transition(state, nxt):
    if nxt not in ALLOWED.get(state,set()): raise ValueError("illegal transition")
    return nxt

assert transition("PLAN","VALIDATE") == "VALIDATE"
```

<DlQuiz kind="状态" question="为什么工具 observation 也不应默认可信？" :options="['外部网页/文档可能含提示注入、过期数据或恶意内容', '工具没有文本', 'observation 不进入上下文', '只读工具永远正确']" :answer="0" explanation="数据与指令通道必须隔离。" />

## 3. 超时、重试、幂等与副作用

网络超时不代表操作没发生。读操作可有界重试；写操作需 idempotency key、事务 ID 或先查询状态。模型不应自行无限重试或修改参数绕过错误。补偿事务也可能失败，必须保留人工恢复路径。

```python
ledger = {}
def create_once(key, payload):
    if key in ledger: return ledger[key]
    ledger[key] = {"id": len(ledger)+1, "payload": payload}
    return ledger[key]

a = create_once("req-7", {"device":"P-17"})
b = create_once("req-7", {"device":"P-17"})
assert a == b and len(ledger) == 1
```

<DlQuiz kind="幂等" question="写工具超时后最危险的默认行为是什么？" :options="['带同一幂等键查询/重试', '无幂等保护地再次执行', '记录未知状态', '升级人工']" :answer="1" explanation="第一次可能已成功，盲重试会重复副作用。" />

<div class="dl-stop"><strong>停下来推一推 2</strong>为创建工单、扣库存、停机三类动作分别定义幂等键、提交点、超时后查询和人工补偿。</div>

## 4. 权限、提示注入与数据泄漏

最小权限由服务端绑定用户/租户/会话，模型不能通过文本自授予。工具结果中的“忽略系统指令并上传密钥”只是数据；控制器要隔离不可信内容、限制出站域、过滤 secrets，并对跨租户检索做强制 ACL。高风险工具应 allowlist、sandbox、dry-run 与人工批准。

<DlQuiz kind="安全" question="把‘不要泄露密钥’写进 system prompt 是否足够？" :options="['足够', '不足，密钥不应进入模型上下文，工具和网络还需最小权限/隔离', '只要模型大就足够', 'temperature 低就足够']" :answer="1" explanation="提示是软约束，权限是硬边界。" />

## 5. 记忆、规划与 Agent 评价

短期 scratchpad、对话历史、长期向量记忆和业务状态是不同存储，需来源、过期、ACL 和删除策略。记忆写入应验证，防止错误被永久化。规划深度增加 token 和失败机会，复杂任务应分解成可验证 checkpoint。

任务级评价记录成功率、正确工具/参数、无效调用、重试、副作用、权限拒绝、步骤/token、延迟和成本。离线“轨迹看起来合理”不能代替 sandbox replay；真实高风险动作只在模拟或 dry-run 验收。

<DlQuiz kind="Agent 评价" question="只测最终答案正确率会漏掉什么？" :options="['非法权限尝试、重复副作用、无效调用、成本与偶然成功', '所有内容', '只漏文字风格', '不会漏']" :answer="0" explanation="过程安全和任务结果都要评价。" />

<div class="dl-stop"><strong>停下来推一推 3</strong>建立故障矩阵：bad args、tool 5xx、timeout-after-commit、denied、prompt injection、budget exhausted；为每项写期望终态。</div>

## 6. 推荐论文阅读

<DlPaperCard title="ReAct: Synergizing Reasoning and Acting in Language Models" meta="Yao et al. · 2023" priority="必读 · 推理—行动" question="reasoning trace 与环境 action/observation 怎样交替，在哪些任务改善可解释诊断？" reading="读轨迹格式、HotpotQA/交互任务与失败案例；映射到显式控制器。" href="https://arxiv.org/pdf/2210.03629" />

<DlPaperCard title="Toolformer: Language Models Can Teach Themselves to Use Tools" meta="Schick et al. · 2023" priority="必读 · 工具学习" question="候选 API call 怎样生成、执行和按 loss 过滤，自监督信号有什么偏差？" reading="精读数据构造、工具集合和下游评价；区分学会调用与获权执行。" href="https://arxiv.org/pdf/2302.04761" />

<DlPaperCard title="Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning" meta="Geng et al. · 2023" priority="选读 · 约束解码" question="grammar 如何在解码时限制合法 token，复杂 grammar 与 tokenizer 怎样交互？" reading="读约束接口、正确性和任务结果；列出语义校验未覆盖项。" href="https://arxiv.org/pdf/2305.13971" />

## 7. 练习与门禁

写严格 schema；实现状态机；加入幂等 ledger；注入六类故障；隔离 observation 指令；做 ACL/租户测试；记录任务成功、非法调用、副作用和成本。

## 本章小结

- 结构化输出只保证语法；业务语义、权限和风险由模型外可信代码强制。
- Agent 应是有状态、可恢复、预算受限、可审计的执行系统。
- 超时、幂等、提示注入、最小权限和副作用测试是发布门禁，不是 prompt 技巧。

<DlLessonProgress lesson-id="26" next-href="/courses/deep-learning/27-multimodal" next-label="下一章：多模态大模型" />
