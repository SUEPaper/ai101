---
course: deep-learning
edition: industrial
title: RLHF、PPO、DPO 与推理模型
description: 从偏好数据与奖励模型进入 KL 正则化策略优化、PPO、DPO、可验证奖励、推理强化学习和 reward hacking。
outline: [2, 3]
---

<DlLessonHero number="23" module="大语言模型" title="RLHF、PPO、DPO 与推理模型" question="如何把‘更有帮助、更安全、更正确’变成可训练偏好信号，又避免模型只学会利用奖励漏洞？" duration="建议 360–470 分钟" :outcomes="['设计含标注协议、一致性和切片的偏好数据', '解释 reward model、reference policy、KL 与 PPO 的系统接口', '从 Bradley–Terry 偏好概率理解 DPO', '区分结果奖励、过程奖励和可验证奖励', '识别长度偏差、奖励过优化、模式坍缩与推理评测污染']" />

后训练不是给模型安装“价值观模块”，而是从有限、带噪、分布偏置的人类或自动偏好中改变输出分布。RLHF 与 DPO 都依赖数据、reference 和评价；优化方法不同，不会消除规范选择与 reward hacking。

## 贯穿案例：危险维修建议的偏好对齐

对同一故障 prompt 采样两条回答，标注员比较正确性、证据、安全和可执行性。系统必须区分“更流畅”与“更安全”，保留并列/不可判定；发布评测包含已知、未知、证据冲突和高风险拒答。

<DlQuiz kind="先判断" question="偏好标签 chosen 是否等于客观真答案？" :options="['永远等于', '不等于，它是标注协议、比较对象和标注者条件下的相对选择', '只在 RLHF 中等于', '只由长度决定']" :answer="1" explanation="偏好可受风格、长度、顺序和知识限制影响。" />

<div class="dl-stop"><strong>停下来推一推 1</strong>同一对回答交换 A/B 顺序、隐藏长度和品牌名后重新标注；如何估计顺序偏差、长度偏差与标注一致性？</div>

## 1. 偏好数据与 Bradley–Terry 奖励模型

奖励模型常从 prompt $x$ 和回答 $y$ 输出标量 $r_\phi(x,y)$，用

$$P(y_c\succ y_r)=\sigma(r_\phi(x,y_c)-r_\phi(x,y_r))$$

拟合成对偏好。只有比较差异可识别，奖励绝对零点没有行为含义。若采样回答太容易区分，模型学不到近边界偏好；若全来自旧策略，离线支持集不足。

```python
import math

def bt_loss(chosen_reward, rejected_reward):
    gap = chosen_reward-rejected_reward
    return math.log1p(math.exp(-gap))

assert bt_loss(2, 0) < bt_loss(.2, 0)
assert math.isclose(bt_loss(0, 0), math.log(2))
```

<DlQuiz kind="奖励模型" question="给所有 reward 同加 100，成对 Bradley–Terry 概率怎样？" :options="['不变，只依赖差值', '都变成 1', '都变成 0', '无法计算']" :answer="0" explanation="绝对偏移不可识别。" />

## 2. RLHF 管线：SFT → RM → 策略优化

InstructGPT 式管线先 SFT，再训练 reward model，最后用 PPO 优化采样回答的奖励，同时以 reference KL 限制偏离。策略、价值网络、reference 与 reward model 可能同时驻留，rollout 生成也昂贵；真实账本包含权重、KV、激活、优化器和采样吞吐。

PPO 用概率比 $r_t=\pi_\theta(a_t|s_t)/\pi_{old}(a_t|s_t)$ 与裁剪目标抑制单批过大更新。LLM 中 token 级 advantage、终止、长度、reward whitening 与 KL 估计都可能改变结果。

```python
def ppo_term(ratio, advantage, eps=.2):
    clipped = min(max(ratio, 1-eps), 1+eps)
    return min(ratio*advantage, clipped*advantage)

assert ppo_term(1.5, 2.0) == 2.4
assert ppo_term(.5, -2.0) == -1.6
```

<DlQuiz kind="PPO" question="PPO clip 是否保证每次更新的真实 KL 都小于固定阈值？" :options="['严格保证', '不严格保证，它限制代理目标中的概率比贡献，仍需监测 KL', '完全不影响更新', '只保证 reward 为正']" :answer="1" explanation="clip 是启发式信赖域代理。" />

<KlRewardLab />

## 3. DPO：从 KL 正则化最优策略到分类损失

DPO 利用 KL 正则化 reward maximization 的闭式关系，把隐式 reward 差写成 policy 与 reference 的 log-ratio 差，直接在偏好 pair 上做 logistic loss，无需显式训练 RM 和在线 PPO rollout。

<PreferencePairLab />

```python
import math

def dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=.1):
    z = beta*((pi_c-pi_r)-(ref_c-ref_r))
    return math.log1p(math.exp(-z))

good = dpo_loss(-1, -3, -1.5, -2.0)
bad = dpo_loss(-3, -1, -1.5, -2.0)
assert good < bad
```

<DlQuiz kind="DPO" question="DPO 不训练显式 reward model，因此还需要 reference policy 吗？" :options="['通常需要，loss 用相对 reference 的 log-ratio', '完全不需要任何基准', '只需随机模型', 'reference 只用于 tokenizer']" :answer="0" explanation="reference 控制隐式 KL/偏移基准。" />

## 4. 失败机制：奖励过优化与分布外策略

当策略找到 reward model 漏洞，代理 reward 上升而人类质量下降，形成 Goodhart/reward hacking。常见表现包括冗长讨好、格式作弊、虚假引用、拒答过度或利用 verifier bug。KL 太强限制改进，太弱导致漂移；单一平均偏好会牺牲少数群体和高风险切片。

<DlQuiz kind="奖励黑客" question="训练 reward 持续上升但冻结人工胜率下降，应如何判断？" :options="['继续最大化 reward', '视为代理过优化，停止并审计 RM 外推、长度/格式漏洞和数据支持', '删除人工评测', '把 KL 设为零']" :answer="1" explanation="代理指标与目标分离时不能按代理发布。" />

<div class="dl-stop"><strong>停下来推一推 2</strong>设计 reward-vs-human 曲线：随 KL/训练步增加，在哪些切片可能先出现翻转？怎样设置 early stop？</div>

## 5. 推理模型：结果、过程与可验证奖励

数学、代码等任务可用单元测试/答案 checker 提供可验证结果奖励，降低纯主观 RM 的一部分噪声。过程监督为中间步骤打分，但标注昂贵且“可读推理文本”不必等于模型真实内部计算。推理 RL 还可能产生更长输出、语言混杂或 verifier exploit。

DeepSeek-R1 等工作展示大规模 RL 与冷启动/SFT 组合对推理能力的影响。评价必须去污染，统一 test-time token/采样预算，验证最终答案并检查错误过程、伪造证明和不可验证领域迁移。隐藏或展示 chain-of-thought 还涉及安全与产品政策，本课不把其当事实保证。

<DlQuiz kind="推理评价" question="比较两个推理模型时只看 pass@1 是否足够？" :options="['足够', '不足，还需统一 token/采样成本、污染、pass@k、校准与失败类型', '只需比较输出长度', '只需看训练 reward']" :answer="1" explanation="test-time compute 会改变质量与成本。" />

<div class="dl-stop"><strong>停下来推一推 3</strong>给代码任务设计 verifier 红队：超时、读取隐藏测试、随机重试、浮点容差和副作用如何阻止？</div>

## 6. 推荐论文阅读

<DlPaperCard title="Training Language Models to Follow Instructions with Human Feedback" meta="Ouyang et al. · 2022" priority="必读 · RLHF 管线" question="SFT、偏好数据、RM 与 PPO 各使用什么数据，KL 和评价怎样组织？" reading="精读方法、模型/数据规模、标注协议与真实性/毒性评价；复画制品依赖。" href="https://arxiv.org/pdf/2203.02155" />

<DlPaperCard title="Direct Preference Optimization: Your Language Model is Secretly a Reward Model" meta="Rafailov et al. · 2023" priority="必读 · 离线偏好优化" question="KL 正则化最优策略怎样导出 DPO logistic loss，β 与 reference 有何作用？" reading="精读第 3 节推导、实验基线和偏好数据限制。" href="https://arxiv.org/pdf/2305.18290" />

<DlPaperCard title="DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" meta="DeepSeek-AI · 2025" priority="选读 · 推理强化学习" question="纯 RL、cold-start data、SFT/distillation 分别解决什么行为问题？" reading="读训练阶段、奖励、评测预算与失败现象；核对可验证任务和不可验证主张边界。" href="https://arxiv.org/pdf/2501.12948" />

## 7. 练习与门禁

测标注一致性；实现 BT/DPO/PPO 小例；画 reward-human 翻转；扫描 β/KL；按长度与安全切片；对 verifier 做攻击；统一 test-time compute 比较推理模型。

## 本章小结

- 偏好是相对、有噪并依赖协议的数据，不等于客观真值。
- RLHF 通过显式 RM/PPO 优化，DPO 直接优化相对 reference 的偏好 log-ratio；两者都需 KL、支持集和冻结评价。
- 推理奖励越可验证越容易自动化，但 verifier 漏洞、长度成本、污染和不可验证迁移仍是发布门禁。

<DlLessonProgress lesson-id="23" next-href="/courses/deep-learning/24-decoding-test-time-scaling" next-label="下一章：解码与 Test-Time Scaling" />
