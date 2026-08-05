export type MlLessonKind = 'core' | 'supplement' | 'practice'

export interface MlLesson {
  number: string
  title: string
  href: string
  kind: MlLessonKind
}

export interface MlModule {
  title: string
  summary: string
  lessons: MlLesson[]
}

const core = (number: string, title: string, slug: string): MlLesson => ({
  number,
  title,
  href: `/courses/machine-learning/${slug}`,
  kind: 'core'
})

export const mlModules: MlModule[] = [
  {
    title: '一 · 学习与优化',
    summary: '先把任务写成风险，再分清训练、优化和泛化。',
    lessons: [
      core('01', '学习问题怎样被数学化', '01-learning-problem'),
      core('02', '参数为什么能被算法学出来', '02-optimization'),
      core('03', '训练误差为什么不等于未来误差', '03-generalization')
    ]
  },
  {
    title: '二 · 回归',
    summary: '从投影、概率和正则化三条路径重建线性回归。',
    lessons: [
      core('04', '最小二乘的几何本质', '04-ols-geometry'),
      core('05', '为什么高斯噪声导出平方损失', '05-gaussian-likelihood'),
      core('06', '正则化究竟改变了什么', '06-regularization'),
      core('07', '线性模型怎样表达非线性', '07-basis-functions')
    ]
  },
  {
    title: '三 · 分类与间隔',
    summary: '从决策代价、概率似然和最大间隔得到分类器。',
    lessons: [
      core('08', '分类首先是一个决策问题', '08-bayesian-decision'),
      core('09', '逻辑回归为什么是这个形式', '09-logistic-regression'),
      core('10', '多分类为什么得到 Softmax', '10-softmax'),
      core('11', '从犯错更新到最大间隔', '11-margin-svm'),
      core('12', '对偶问题为什么带来核技巧', '12-kernels')
    ]
  },
  {
    title: '四 · 神经网络',
    summary: '沿计算图推导 MLP、卷积、序列递推和注意力。',
    lessons: [
      core('13', '复合线性层怎样表达复杂函数', '13-mlp'),
      core('14', '反向传播为什么高效', '14-backpropagation'),
      core('15', 'CNN 为什么适合网格数据', '15-cnn'),
      core('16', 'RNN 怎样对序列共享状态', '16-rnn'),
      core('17', 'Transformer 为什么不需要递归', '17-transformer')
    ]
  },
  {
    title: '五 · 非参数与集成',
    summary: '观察局部方法、树与集成如何交换偏差、方差和计算。',
    lessons: [
      core('18', '不写参数方程也能学习吗', '18-knn'),
      core('19', '决策树的划分准则从哪里来', '19-decision-trees'),
      core('20', 'Bagging 与随机森林为什么稳定', '20-random-forests'),
      core('21', 'Boosting 为什么能把弱学习器变强', '21-boosting')
    ]
  },
  {
    title: '六 · 概率与潜变量',
    summary: '从似然、先验和看不见的类别推导概率模型。',
    lessons: [
      core('22', 'MLE 与 MAP 为什么反复出现', '22-mle-map'),
      core('23', '生成式分类器怎样得到决策边界', '23-generative-classifiers'),
      core('24', '看不见的变量怎样被估计', '24-em')
    ]
  },
  {
    title: '七 · 降维与聚类',
    summary: '从方差、重构和簇内距离理解无监督目标。',
    lessons: [
      core('25', 'PCA 为什么等于寻找特征向量', '25-pca'),
      core('26', 'K-means 为什么交替两步', '26-kmeans'),
      core('27', '聚类没有标签怎样判断好坏', '27-clustering-evaluation')
    ]
  },
  {
    title: '八 · 学习理论',
    summary: '用概率界和可靠评价说明有限数据为何可能泛化。',
    lessons: [core('28', '有限数据为什么可能学到规律', '28-learning-theory')]
  },
  {
    title: '九 · 推荐与强化学习',
    summary: '从缺失矩阵走到会在环境中持续学习的决策系统。',
    lessons: [
      core('29', '推荐系统怎样从缺失矩阵学习', '29-recommender-systems'),
      core('30', 'Bellman 方程为什么成立', '30-mdp-bellman'),
      core('31', '没有环境模型怎样学习价值', '31-value-learning'),
      core('32', '策略怎样直接沿回报方向更新', '32-policy-gradient')
    ]
  }
]

export const mlSupplementModule: MlModule = {
  title: '选修 · 数学工具箱',
  summary: '不阻断主线，用于补全证明和数值线性代数视角。',
  lessons: [
    { number: 'S01', title: '矩阵分解', href: '/courses/machine-learning/s01-matrix-factorization', kind: 'supplement' },
    { number: 'S02', title: '矩阵微分', href: '/courses/machine-learning/s02-matrix-calculus', kind: 'supplement' },
    { number: 'S03', title: '概率统计', href: '/courses/machine-learning/s03-probability', kind: 'supplement' },
    { number: 'S04', title: '信息论', href: '/courses/machine-learning/s04-information-theory', kind: 'supplement' },
    { number: 'S05', title: '凸优化', href: '/courses/machine-learning/s05-convex-optimization', kind: 'supplement' },
    { number: 'S06', title: 'Lagrange 对偶', href: '/courses/machine-learning/s06-lagrange-duality', kind: 'supplement' }
  ]
}

export const mlPracticeModule: MlModule = {
  title: '综合实践',
  summary: '把推导、实现、验证和失效分析组合成可审查成果。',
  lessons: [
    { number: 'A1–A8', title: '八次综合作业', href: '/courses/machine-learning/assignments', kind: 'practice' },
    { number: 'PROJECT', title: '机器学习结课项目', href: '/courses/machine-learning/project', kind: 'practice' },
    { number: 'SOURCE', title: '来源与许可登记', href: '/courses/machine-learning/sources', kind: 'practice' }
  ]
}

export const mlAllModules = [...mlModules, mlSupplementModule, mlPracticeModule]
export const mlCoreLessons = mlModules.flatMap((module) => module.lessons)
export const mlLessons = mlAllModules.flatMap((module) => module.lessons)

export function normalizeMlPath(path: string) {
  return path.replace(/\.html$/, '').replace(/\/$/, '')
}
