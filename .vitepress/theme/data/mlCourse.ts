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
      core('01', '监督学习与经验风险最小化', '01-learning-problem'),
      core('02', '梯度下降、牛顿法与随机优化', '02-optimization'),
      core('03', '泛化、偏差—方差与模型选择', '03-generalization')
    ]
  },
  {
    title: '二 · 回归',
    summary: '从投影、概率和正则化三条路径重建线性回归。',
    lessons: [
      core('04', '普通最小二乘法（OLS）', '04-ols-geometry'),
      core('05', '线性回归：高斯似然与统计推断', '05-gaussian-likelihood'),
      core('06', 'Ridge 回归与 Lasso', '06-regularization'),
      core('07', '基函数回归与局部加权回归', '07-basis-functions')
    ]
  },
  {
    title: '三 · 分类与间隔',
    summary: '从决策代价、概率似然和最大间隔得到分类器。',
    lessons: [
      core('08', '贝叶斯决策与最优分类器', '08-bayesian-decision'),
      core('09', '逻辑回归', '09-logistic-regression'),
      core('10', 'Softmax 回归', '10-softmax'),
      core('11', '感知机与支持向量机', '11-margin-svm'),
      core('12', '核方法与核 SVM', '12-kernels')
    ]
  },
  {
    title: '四 · 神经网络',
    summary: '沿计算图推导 MLP、卷积、序列递推和注意力。',
    lessons: [
      core('13', '多层感知机（MLP）', '13-mlp'),
      core('14', '反向传播', '14-backpropagation'),
      core('15', '卷积神经网络（CNN）', '15-cnn'),
      core('16', '循环神经网络（RNN）', '16-rnn'),
      core('17', 'Transformer 与自注意力', '17-transformer')
    ]
  },
  {
    title: '五 · 非参数与集成',
    summary: '观察局部方法、树与集成如何交换偏差、方差和计算。',
    lessons: [
      core('18', 'k 近邻算法（kNN）', '18-knn'),
      core('19', '决策树', '19-decision-trees'),
      core('20', 'Bagging 与随机森林', '20-random-forests'),
      core('21', 'AdaBoost 与梯度提升', '21-boosting')
    ]
  },
  {
    title: '六 · 概率与潜变量',
    summary: '从似然、先验和看不见的类别推导概率模型。',
    lessons: [
      core('22', '最大似然估计与最大后验估计', '22-mle-map'),
      core('23', '高斯判别分析与朴素贝叶斯', '23-generative-classifiers'),
      core('24', '期望最大化算法（EM）', '24-em')
    ]
  },
  {
    title: '七 · 降维与聚类',
    summary: '从方差、重构和簇内距离理解无监督目标。',
    lessons: [
      core('25', '主成分分析（PCA）', '25-pca'),
      core('26', 'K-means 聚类', '26-kmeans'),
      core('27', '聚类评估与模型选择', '27-clustering-evaluation')
    ]
  },
  {
    title: '八 · 学习理论',
    summary: '用概率界和可靠评价说明有限数据为何可能泛化。',
    lessons: [core('28', 'PAC 学习与 VC 维', '28-learning-theory')]
  },
  {
    title: '九 · 推荐系统',
    summary: '从缺失矩阵、隐式反馈与排序目标建立可审计的个性化推荐。',
    lessons: [
      core('29', '矩阵分解与 BPR 推荐', '29-recommender-systems')
    ]
  },
  {
    title: '十 · 强化学习',
    summary: '从已知模型的动态规划走到无模型价值学习与策略优化。',
    lessons: [
      core('30', '马尔可夫决策过程与动态规划', '30-mdp-bellman'),
      core('31', 'SARSA 与 Q-learning', '31-value-learning'),
      core('32', 'REINFORCE 与 Actor–Critic', '32-policy-gradient')
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
