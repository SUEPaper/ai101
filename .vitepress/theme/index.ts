// https://vitepress.dev/guide/custom-theme
import { h } from 'vue'
import type { Theme } from 'vitepress'
import DefaultTheme from 'vitepress/theme'
import AiHome from './components/AiHome.vue'
import ArrayShapeLab from './components/ds/ArrayShapeLab.vue'
import BooleanMaskLab from './components/ds/BooleanMaskLab.vue'
import BiasVarianceLab from './components/ds/BiasVarianceLab.vue'
import BootstrapLab from './components/ds/BootstrapLab.vue'
import ClusteringLab from './components/ds/ClusteringLab.vue'
import CourseLayout from './components/CourseLayout.vue'
import CrossValidationLab from './components/ds/CrossValidationLab.vue'
import DataFrameExplorer from './components/ds/DataFrameExplorer.vue'
import DataQualityLab from './components/ds/DataQualityLab.vue'
import DsCourseMap from './components/ds/DsCourseMap.vue'
import DsLessonHero from './components/ds/DsLessonHero.vue'
import DsLessonProgress from './components/ds/DsLessonProgress.vue'
import FairnessLab from './components/ds/FairnessLab.vue'
import FeatureEngineeringLab from './components/ds/FeatureEngineeringLab.vue'
import GradientDescentLab from './components/ds/GradientDescentLab.vue'
import GroupByLab from './components/ds/GroupByLab.vue'
import LifecycleLab from './components/ds/LifecycleLab.vue'
import LogisticCurveLab from './components/ds/LogisticCurveLab.vue'
import LossComparisonLab from './components/ds/LossComparisonLab.vue'
import ProjectionLab from './components/ds/ProjectionLab.vue'
import PcaProjectionLab from './components/ds/PcaProjectionLab.vue'
import QuizBlock from './components/ds/QuizBlock.vue'
import RegexLab from './components/ds/RegexLab.vue'
import HistogramLab from './components/ds/HistogramLab.vue'
import SamplingLab from './components/ds/SamplingLab.vue'
import RandomVariableLab from './components/ds/RandomVariableLab.vue'
import RegressionLineLab from './components/ds/RegressionLineLab.vue'
import ScatterLab from './components/ds/ScatterLab.vue'
import SqlJoinLab from './components/ds/SqlJoinLab.vue'
import SqlQueryLab from './components/ds/SqlQueryLab.vue'
import SvdLayersLab from './components/ds/SvdLayersLab.vue'
import MetricThresholdLab from './components/ds/MetricThresholdLab.vue'
import MlCourseMap from './components/ml/MlCourseMap.vue'
import MlLab from './components/ml/MlLab.vue'
import MlLessonHero from './components/ml/MlLessonHero.vue'
import MlLessonProgress from './components/ml/MlLessonProgress.vue'
import MlQuiz from './components/ml/MlQuiz.vue'
import './style.css'

export default {
  extends: DefaultTheme,
  Layout: () => {
    return h(CourseLayout)
  },
  enhanceApp({ app }) {
    app.component('AiHome', AiHome)
    app.component('ArrayShapeLab', ArrayShapeLab)
    app.component('BiasVarianceLab', BiasVarianceLab)
    app.component('BooleanMaskLab', BooleanMaskLab)
    app.component('BootstrapLab', BootstrapLab)
    app.component('ClusteringLab', ClusteringLab)
    app.component('CrossValidationLab', CrossValidationLab)
    app.component('DataFrameExplorer', DataFrameExplorer)
    app.component('DataQualityLab', DataQualityLab)
    app.component('DsCourseMap', DsCourseMap)
    app.component('DsLessonHero', DsLessonHero)
    app.component('DsLessonProgress', DsLessonProgress)
    app.component('FairnessLab', FairnessLab)
    app.component('FeatureEngineeringLab', FeatureEngineeringLab)
    app.component('GradientDescentLab', GradientDescentLab)
    app.component('GroupByLab', GroupByLab)
    app.component('LifecycleLab', LifecycleLab)
    app.component('LossComparisonLab', LossComparisonLab)
    app.component('LogisticCurveLab', LogisticCurveLab)
    app.component('MetricThresholdLab', MetricThresholdLab)
    app.component('MlCourseMap', MlCourseMap)
    app.component('MlLab', MlLab)
    app.component('MlLessonHero', MlLessonHero)
    app.component('MlLessonProgress', MlLessonProgress)
    app.component('MlQuiz', MlQuiz)
    app.component('PcaProjectionLab', PcaProjectionLab)
    app.component('ProjectionLab', ProjectionLab)
    app.component('QuizBlock', QuizBlock)
    app.component('RegexLab', RegexLab)
    app.component('HistogramLab', HistogramLab)
    app.component('SamplingLab', SamplingLab)
    app.component('RandomVariableLab', RandomVariableLab)
    app.component('RegressionLineLab', RegressionLineLab)
    app.component('ScatterLab', ScatterLab)
    app.component('SqlJoinLab', SqlJoinLab)
    app.component('SqlQueryLab', SqlQueryLab)
    app.component('SvdLayersLab', SvdLayersLab)
  }
} satisfies Theme
