export interface DsLesson {
  number: string
  title: string
  href: string
  available: boolean
}

export interface DsModule {
  title: string
  summary: string
  lessons: DsLesson[]
}

export const dsModules: DsModule[] = [
  {
    title: '起步',
    summary: '补齐这门课真正会用到的工具和数学直觉。',
    lessons: [
      { number: 'P0', title: '开始前的工具与数学桥', href: '/courses/data-science/00-preparation', available: true }
    ]
  },
  {
    title: '数据科学基础',
    summary: '把现实问题变成结构清晰、可以计算的数据任务。',
    lessons: [
      { number: '01', title: '数据科学生命周期', href: '/courses/data-science/01-lifecycle', available: true },
      { number: '02', title: 'Pandas：表、列与索引', href: '/courses/data-science/02-pandas-structures', available: true },
      { number: '03', title: 'Pandas：筛选与变换', href: '/courses/data-science/03-pandas-transform', available: true },
      { number: '04', title: 'Pandas：分组、透视与连接', href: '/courses/data-science/04-pandas-aggregate', available: true }
    ]
  },
  {
    title: '理解和表达数据',
    summary: '检查数据质量，用图形和抽样理解数据中的信号。',
    lessons: [
      { number: '05', title: '数据清洗与探索性分析', href: '/courses/data-science/05-cleaning-eda', available: true },
      { number: '06', title: '文本与正则表达式', href: '/courses/data-science/06-regex', available: true },
      { number: '07', title: '可视化：看见分布', href: '/courses/data-science/07-visualization-distributions', available: true },
      { number: '08', title: '可视化：关系与表达', href: '/courses/data-science/08-visualization-relationships', available: true },
      { number: '09', title: '抽样与偏差', href: '/courses/data-science/09-sampling', available: true }
    ]
  },
  {
    title: '回归与模型训练',
    summary: '从一条直线走到可以评价和约束的预测模型。',
    lessons: [
      { number: '10', title: '建模与简单线性回归', href: '/courses/data-science/10-simple-linear-regression', available: true },
      { number: '11', title: '损失函数与数据变换', href: '/courses/data-science/11-loss-transformations', available: true },
      { number: '12', title: '多元回归与最小二乘', href: '/courses/data-science/12-ordinary-least-squares', available: true },
      { number: '13', title: 'sklearn 与梯度下降', href: '/courses/data-science/13-gradient-descent', available: true },
      { number: '14', title: '特征工程与随机梯度下降', href: '/courses/data-science/14-feature-engineering', available: true },
      { number: '15', title: '案例：房产估价与公平性', href: '/courses/data-science/15-fairness-case', available: true },
      { number: '16', title: '交叉验证与正则化', href: '/courses/data-science/16-cross-validation', available: true }
    ]
  },
  {
    title: '概率与统计推断',
    summary: '理解样本为什么会变化，以及结论有多大不确定性。',
    lessons: [
      { number: '17', title: '随机变量', href: '/courses/data-science/17-random-variables', available: true },
      { number: '18', title: '估计量、偏差与方差', href: '/courses/data-science/18-estimators', available: true },
      { number: '19', title: 'Bootstrap 与参数推断', href: '/courses/data-science/19-bootstrap', available: true }
    ]
  },
  {
    title: '数据库',
    summary: '在关系数据库中筛选、聚合和组合数据。',
    lessons: [
      { number: '20', title: 'SQL：查询一张表', href: '/courses/data-science/20-sql-basics', available: true },
      { number: '21', title: 'SQL：分组与连接', href: '/courses/data-science/21-sql-joins', available: true }
    ]
  },
  {
    title: '分类',
    summary: '从概率预测走到可解释、可评价的分类决策。',
    lessons: [
      { number: '22', title: '逻辑回归', href: '/courses/data-science/22-logistic-regression', available: true },
      { number: '23', title: '分类指标与阈值', href: '/courses/data-science/23-classification-metrics', available: true }
    ]
  },
  {
    title: '无监督学习',
    summary: '在没有标签的数据中寻找低维结构和自然分组。',
    lessons: [
      { number: '24', title: '主成分分析：投影', href: '/courses/data-science/24-pca-projection', available: true },
      { number: '25', title: '主成分分析：SVD', href: '/courses/data-science/25-pca-svd', available: true },
      { number: '26', title: '聚类', href: '/courses/data-science/26-clustering', available: true }
    ]
  },
  {
    title: '图像数据桥接',
    summary: '把图像变成可检查、可变换、可测量的数组与对象表。',
    lessons: [
      { number: 'V1', title: '数字图像：从像素到数组合同', href: '/courses/data-science/v1-digital-images', available: true },
      { number: 'V2', title: '像素变换与几何变换', href: '/courses/data-science/v2-pixel-geometric-transformations', available: true },
      { number: 'V3', title: '空间操作、分割与区域测量', href: '/courses/data-science/v3-spatial-segmentation', available: true }
    ]
  },
  {
    title: '综合实践',
    summary: '用模块作业和结课项目把方法组合成完整工作流。',
    lessons: [
      { number: 'HW', title: '七次模块综合作业', href: '/courses/data-science/assignments', available: true },
      { number: 'PROJECT', title: '数据科学结课项目', href: '/courses/data-science/project', available: true }
    ]
  }
]

export const dsLessons = dsModules.flatMap((module) => module.lessons)

export function normalizeDsPath(path: string) {
  return path.replace(/\.html$/, '').replace(/\/$/, '')
}
