<script setup lang="ts">
import { computed, ref } from 'vue'

type Fault = 'healthy' | 'high-lr' | 'bad-labels' | 'leakage' | 'norm-shift'
const fault = ref<Fault>('high-lr')
const epoch = ref(18)
const profiles: Record<Fault, { train: number[]; val: number[]; grad: string; action: string; clue: string }> = {
  healthy: { train: [1.8,1.15,.79,.58,.44,.36,.31], val: [1.84,1.22,.88,.68,.56,.51,.49], grad: '稳定下降', action: '继续按冻结协议训练', clue: '训练与验证同向改善，间隔有限' },
  'high-lr': { train: [1.8,1.25,1.64,.94,1.41,.83,1.32], val: [1.86,1.31,1.78,1.04,1.57,.98,1.49], grad: '尖峰/符号翻转', action: '降低 LR，检查 loss scale 与裁剪', clue: '损失锯齿且梯度范数尖峰同步' },
  'bad-labels': { train: [1.8,1.48,1.31,1.19,1.13,1.09,1.07], val: [1.82,1.53,1.38,1.31,1.29,1.28,1.29], grad: '有界但信号弱', action: '抽样核对标签与类别条件分布', clue: '训练很早进入高损失平台' },
  leakage: { train: [1.8,.82,.24,.06,.02,.01,.005], val: [1.78,.76,.18,.03,.008,.004,.003], grad: '异常容易', action: '冻结训练，审计时间与实体切分', clue: '几乎无泛化间隔且不合常理地完美' },
  'norm-shift': { train: [1.8,1.09,.72,.51,.39,.32,.28], val: [1.83,1.35,1.28,1.33,1.42,1.51,1.57], grad: '训练侧正常', action: '核对 eval 模式与输入统计量', clue: '训练持续改善，验证从早期开始反向恶化' }
}
const profile = computed(() => profiles[fault.value])
const index = computed(() => Math.min(6, Math.floor(epoch.value / 5)))
function points(values: number[]) { return values.map((value, i) => `${36 + i * 76},${220 - value / 2 * 170}`).join(' ') }
</script>

<template>
  <DlExperimentFrame index="03-B" title="训练曲线故障注入台" question="只看最终准确率会错过什么？先用曲线指纹提出可证伪诊断。" formula="curve fingerprint + gradient trace + data audit → next diagnostic action">
    <template #visual>
      <svg class="curves" viewBox="0 0 540 255" role="img" :aria-label="`${fault} 故障下的训练与验证损失曲线`">
        <g class="grid"><line v-for="y in [50,95,140,185,220]" :key="y" x1="36" :y1="y" x2="492" :y2="y" /></g>
        <polyline class="train" :points="points(profile.train)" /><polyline class="val" :points="points(profile.val)" />
        <line class="cursor" :x1="36 + index * 76" y1="34" :x2="36 + index * 76" y2="220" />
        <text x="38" y="30">train</text><text x="92" y="30" class="val-label">validation</text><text x="36" y="242">epoch 0</text><text x="444" y="242">epoch 30</text>
      </svg>
    </template>
    <template #controls>
      <label>注入情景<select v-model="fault"><option value="healthy">健康基线</option><option value="high-lr">学习率过高</option><option value="bad-labels">标签噪声/定义错误</option><option value="leakage">数据泄漏</option><option value="norm-shift">训练/推理归一化错位</option></select></label>
      <label>检查 epoch = {{ epoch }}<input v-model.number="epoch" type="range" min="0" max="30" step="1"></label>
    </template>
    <template #metrics>
      <div><span>训练损失</span><strong>{{ profile.train[index].toFixed(3) }}</strong></div>
      <div><span>验证损失</span><strong>{{ profile.val[index].toFixed(3) }}</strong></div>
      <div><span>梯度迹线</span><strong>{{ profile.grad }}</strong></div>
      <div><span>关键线索</span><strong>{{ profile.clue }}</strong></div>
    </template>
    <template #note>第一动作：{{ profile.action }}。曲线只能缩小原因集合，不能单独证明根因；下一步必须读取批次、梯度、数据切分或运行模式的原始证据。</template>
  </DlExperimentFrame>
</template>

<style scoped>
.curves{display:block;width:100%;min-height:280px;border:1px solid #294b5c;border-radius:7px;background:#0b172a}.grid line{stroke:#223a4f}.curves polyline{fill:none;stroke-width:3}.train{stroke:#58d3c8}.val{stroke:#f2c14e}.cursor{stroke:#df6c5b;stroke-width:2;stroke-dasharray:5 4}.curves text{fill:#58d3c8;font:10px ui-monospace,monospace}.curves .val-label{fill:#f2c14e}
</style>
