// https://vitepress.dev/guide/custom-theme
import { h } from 'vue'
import type { Theme } from 'vitepress'
import DefaultTheme from 'vitepress/theme'
import AiHome from './components/AiHome.vue'
import ArrayShapeLab from './components/ds/ArrayShapeLab.vue'
import ActivationFunctionScope from './components/ml/ActivationFunctionScope.vue'
import AttentionRoutingConsole from './components/ml/AttentionRoutingConsole.vue'
import BooleanMaskLab from './components/ds/BooleanMaskLab.vue'
import BiasVarianceTradeoffLab from './components/ml/BiasVarianceTradeoffLab.vue'
import BasisExpansionWorkbench from './components/ml/BasisExpansionWorkbench.vue'
import BatchGradientReducer from './components/ml/BatchGradientReducer.vue'
import BellmanBackupSwitchboard from './components/ml/BellmanBackupSwitchboard.vue'
import BprPairwiseSorter from './components/ml/BprPairwiseSorter.vue'
import BetaEvidenceMixer from './components/ml/BetaEvidenceMixer.vue'
import BootstrapOobCarousel from './components/ml/BootstrapOobCarousel.vue'
import AdaBoostWeightRelay from './components/ml/AdaBoostWeightRelay.vue'
import BpttGradientScope from './components/ml/BpttGradientScope.vue'
import CausalMaskLeakScanner from './components/ml/CausalMaskLeakScanner.vue'
import BayesRiskDispatchLab from './components/ml/BayesRiskDispatchLab.vue'
import CollinearitySolutionsLab from './components/ml/CollinearitySolutionsLab.vue'
import ColumnSpaceProjectionLab from './components/ml/ColumnSpaceProjectionLab.vue'
import ComputationGraphTrace from './components/ml/ComputationGraphTrace.vue'
import ConvolutionBackwardCollector from './components/ml/ConvolutionBackwardCollector.vue'
import ConvolutionScanWorkbench from './components/ml/ConvolutionScanWorkbench.vue'
import CostComplexityPruningRail from './components/ml/CostComplexityPruningRail.vue'
import DistanceConcentrationTunnel from './components/ml/DistanceConcentrationTunnel.vue'
import DistanceScaleCalibrationBay from './components/ml/DistanceScaleCalibrationBay.vue'
import ExposureMatrixLightTable from './components/ml/ExposureMatrixLightTable.vue'
import EmCollapseGuard from './components/ml/EmCollapseGuard.vue'
import EmIterationPlayback from './components/ml/EmIterationPlayback.vue'
import EmResponsibilityFlow from './components/ml/EmResponsibilityFlow.vue'
import PcaProjectionGantry from './components/ml/PcaProjectionGantry.vue'
import PcaScalingWorkbench from './components/ml/PcaScalingWorkbench.vue'
import PcaT2QMonitor from './components/ml/PcaT2QMonitor.vue'
import KMeansVoronoiSwitchyard from './components/ml/KMeansVoronoiSwitchyard.vue'
import KMeansIterationPlayback from './components/ml/KMeansIterationPlayback.vue'
import KMeansEmptyClusterRescue from './components/ml/KMeansEmptyClusterRescue.vue'
import ClusterEvidenceLedger from './components/ml/ClusterEvidenceLedger.vue'
import LinkageDendrogramBench from './components/ml/LinkageDendrogramBench.vue'
import SpectralGraphCutLab from './components/ml/SpectralGraphCutLab.vue'
import PacConfidenceCorridor from './components/ml/PacConfidenceCorridor.vue'
import VcShatteringComparator from './components/ml/VcShatteringComparator.vue'
import PairedLotBootstrapRail from './components/ml/PairedLotBootstrapRail.vue'
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
import ErmSamplingLab from './components/ml/ErmSamplingLab.vue'
import ForestCorrelationRack from './components/ml/ForestCorrelationRack.vue'
import GradientResidualCoater from './components/ml/GradientResidualCoater.vue'
import GaussianLikelihoodLab from './components/ml/GaussianLikelihoodLab.vue'
import GaussianBoundaryProbe from './components/ml/GaussianBoundaryProbe.vue'
import GradientCheckMicroscope from './components/ml/GradientCheckMicroscope.vue'
import GatedMemoryFlow from './components/ml/GatedMemoryFlow.vue'
import HeteroskedasticCoverageLab from './components/ml/HeteroskedasticCoverageLab.vue'
import InitializationSignalTunnel from './components/ml/InitializationSignalTunnel.vue'
import KernelFeatureLift from './components/ml/KernelFeatureLift.vue'
import KnnVoteInspectionBay from './components/ml/KnnVoteInspectionBay.vue'
import LatentFactorAlignmentBench from './components/ml/LatentFactorAlignmentBench.vue'
import ReceptiveFieldPlanner from './components/ml/ReceptiveFieldPlanner.vue'
import RnnStateUnroller from './components/ml/RnnStateUnroller.vue'
import LassoGeometryLab from './components/ml/LassoGeometryLab.vue'
import LocallyWeightedRegressionLab from './components/ml/LocallyWeightedRegressionLab.vue'
import LogitEvidenceConsole from './components/ml/LogitEvidenceConsole.vue'
import LikelihoodProfileScanner from './components/ml/LikelihoodProfileScanner.vue'
import MahalanobisWhiteningGauge from './components/ml/MahalanobisWhiteningGauge.vue'
import MapSamplingRiskArray from './components/ml/MapSamplingRiskArray.vue'
import MarkovStateAuditConsole from './components/ml/MarkovStateAuditConsole.vue'
import ActorCriticSignalRoom from './components/ml/ActorCriticSignalRoom.vue'
import BaselineVarianceStripChart from './components/ml/BaselineVarianceStripChart.vue'
import PolicyGradientSignalBench from './components/ml/PolicyGradientSignalBench.vue'
import QRouteLearningPlayback from './components/ml/QRouteLearningPlayback.vue'
import ReturnTargetMixer from './components/ml/ReturnTargetMixer.vue'
import SarsaQTargetRelay from './components/ml/SarsaQTargetRelay.vue'
import MlpCoherenceRouter from './components/ml/MlpCoherenceRouter.vue'
import MultiHeadPositionBoard from './components/ml/MultiHeadPositionBoard.vue'
import NaiveBayesEvidenceLedger from './components/ml/NaiveBayesEvidenceLedger.vue'
import MlLab from './components/ml/MlLab.vue'
import MlLessonHero from './components/ml/MlLessonHero.vue'
import MlLessonProgress from './components/ml/MlLessonProgress.vue'
import MlQuiz from './components/ml/MlQuiz.vue'
import OptimizationPathLab from './components/ml/OptimizationPathLab.vue'
import PerceptronUpdatePlayback from './components/ml/PerceptronUpdatePlayback.vue'
import RandomSubspaceSwitchyard from './components/ml/RandomSubspaceSwitchyard.vue'
import SecondOrderSplitFoundry from './components/ml/SecondOrderSplitFoundry.vue'
import RiskDecisionLab from './components/ml/RiskDecisionLab.vue'
import RidgeSpectralShrinkageLab from './components/ml/RidgeSpectralShrinkageLab.vue'
import RocOperatingPointLab from './components/ml/RocOperatingPointLab.vue'
import SeparationRegularizationLab from './components/ml/SeparationRegularizationLab.vue'
import SoftmaxCompetitionConsole from './components/ml/SoftmaxCompetitionConsole.vue'
import SoftmaxReferenceGauge from './components/ml/SoftmaxReferenceGauge.vue'
import StochasticGradientLab from './components/ml/StochasticGradientLab.vue'
import SvmMarginCaliper from './components/ml/SvmMarginCaliper.vue'
import SvmDualBalanceBench from './components/ml/SvmDualBalanceBench.vue'
import TreeRouteManifold from './components/ml/TreeRouteManifold.vue'
import TreeSplitFoundry from './components/ml/TreeSplitFoundry.vue'
import DynamicProgrammingPulseGrid from './components/ml/DynamicProgrammingPulseGrid.vue'
import ValidationSelectionLab from './components/ml/ValidationSelectionLab.vue'
import './style.css'

export default {
  extends: DefaultTheme,
  Layout: () => {
    return h(CourseLayout)
  },
  enhanceApp({ app }) {
    app.component('AiHome', AiHome)
    app.component('ArrayShapeLab', ArrayShapeLab)
    app.component('ActivationFunctionScopeLab', ActivationFunctionScope)
    app.component('AttentionRoutingConsoleLab', AttentionRoutingConsole)
    app.component('BiasVarianceLab', BiasVarianceLab)
    app.component('BooleanMaskLab', BooleanMaskLab)
    app.component('BiasVarianceTradeoffLab', BiasVarianceTradeoffLab)
    app.component('BasisExpansionWorkbenchLab', BasisExpansionWorkbench)
    app.component('BatchGradientReducerLab', BatchGradientReducer)
    app.component('BellmanBackupSwitchboardLab', BellmanBackupSwitchboard)
    app.component('BprPairwiseSorterLab', BprPairwiseSorter)
    app.component('BetaEvidenceMixerLab', BetaEvidenceMixer)
    app.component('BootstrapOobCarouselLab', BootstrapOobCarousel)
    app.component('AdaBoostWeightRelayLab', AdaBoostWeightRelay)
    app.component('BpttGradientScopeLab', BpttGradientScope)
    app.component('CausalMaskLeakScannerLab', CausalMaskLeakScanner)
    app.component('BayesRiskDispatchLab', BayesRiskDispatchLab)
    app.component('CollinearitySolutionsLab', CollinearitySolutionsLab)
    app.component('ColumnSpaceProjectionLab', ColumnSpaceProjectionLab)
    app.component('ComputationGraphTraceLab', ComputationGraphTrace)
    app.component('ConvolutionBackwardCollectorLab', ConvolutionBackwardCollector)
    app.component('ConvolutionScanWorkbenchLab', ConvolutionScanWorkbench)
    app.component('CostComplexityPruningRailLab', CostComplexityPruningRail)
    app.component('DistanceConcentrationTunnelLab', DistanceConcentrationTunnel)
    app.component('DistanceScaleCalibrationBayLab', DistanceScaleCalibrationBay)
    app.component('ExposureMatrixLightTableLab', ExposureMatrixLightTable)
    app.component('EmCollapseGuardLab', EmCollapseGuard)
    app.component('EmIterationPlaybackLab', EmIterationPlayback)
    app.component('EmResponsibilityFlowLab', EmResponsibilityFlow)
    app.component('PcaProjectionGantryLab', PcaProjectionGantry)
    app.component('PcaScalingWorkbenchLab', PcaScalingWorkbench)
    app.component('PcaT2QMonitorLab', PcaT2QMonitor)
    app.component('KMeansVoronoiSwitchyardLab', KMeansVoronoiSwitchyard)
    app.component('KMeansIterationPlaybackLab', KMeansIterationPlayback)
    app.component('KMeansEmptyClusterRescueLab', KMeansEmptyClusterRescue)
    app.component('ClusterEvidenceLedgerLab', ClusterEvidenceLedger)
    app.component('LinkageDendrogramBenchLab', LinkageDendrogramBench)
    app.component('SpectralGraphCutLab', SpectralGraphCutLab)
    app.component('PacConfidenceCorridorLab', PacConfidenceCorridor)
    app.component('VcShatteringComparatorLab', VcShatteringComparator)
    app.component('PairedLotBootstrapRailLab', PairedLotBootstrapRail)
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
    app.component('ErmSamplingLab', ErmSamplingLab)
    app.component('ForestCorrelationRackLab', ForestCorrelationRack)
    app.component('GradientResidualCoaterLab', GradientResidualCoater)
    app.component('GaussianLikelihoodLab', GaussianLikelihoodLab)
    app.component('GaussianBoundaryProbeLab', GaussianBoundaryProbe)
    app.component('GradientCheckMicroscopeLab', GradientCheckMicroscope)
    app.component('GatedMemoryFlowLab', GatedMemoryFlow)
    app.component('HeteroskedasticCoverageLab', HeteroskedasticCoverageLab)
    app.component('InitializationSignalTunnelLab', InitializationSignalTunnel)
    app.component('KernelFeatureLiftLab', KernelFeatureLift)
    app.component('KnnVoteInspectionBayLab', KnnVoteInspectionBay)
    app.component('LatentFactorAlignmentBenchLab', LatentFactorAlignmentBench)
    app.component('ReceptiveFieldPlannerLab', ReceptiveFieldPlanner)
    app.component('RnnStateUnrollerLab', RnnStateUnroller)
    app.component('LassoGeometryLab', LassoGeometryLab)
    app.component('LocallyWeightedRegressionLab', LocallyWeightedRegressionLab)
    app.component('LogitEvidenceConsoleLab', LogitEvidenceConsole)
    app.component('LikelihoodProfileScannerLab', LikelihoodProfileScanner)
    app.component('MahalanobisWhiteningGaugeLab', MahalanobisWhiteningGauge)
    app.component('MapSamplingRiskArrayLab', MapSamplingRiskArray)
    app.component('MarkovStateAuditConsoleLab', MarkovStateAuditConsole)
    app.component('ActorCriticSignalRoomLab', ActorCriticSignalRoom)
    app.component('BaselineVarianceStripChartLab', BaselineVarianceStripChart)
    app.component('PolicyGradientSignalBenchLab', PolicyGradientSignalBench)
    app.component('QRouteLearningPlaybackLab', QRouteLearningPlayback)
    app.component('ReturnTargetMixerLab', ReturnTargetMixer)
    app.component('SarsaQTargetRelayLab', SarsaQTargetRelay)
    app.component('MlpCoherenceRouterLab', MlpCoherenceRouter)
    app.component('MultiHeadPositionBoardLab', MultiHeadPositionBoard)
    app.component('NaiveBayesEvidenceLedgerLab', NaiveBayesEvidenceLedger)
    app.component('MlLab', MlLab)
    app.component('MlLessonHero', MlLessonHero)
    app.component('MlLessonProgress', MlLessonProgress)
    app.component('MlQuiz', MlQuiz)
    app.component('OptimizationPathLab', OptimizationPathLab)
    app.component('PerceptronUpdatePlaybackLab', PerceptronUpdatePlayback)
    app.component('RandomSubspaceSwitchyardLab', RandomSubspaceSwitchyard)
    app.component('SecondOrderSplitFoundryLab', SecondOrderSplitFoundry)
    app.component('RiskDecisionLab', RiskDecisionLab)
    app.component('RidgeSpectralShrinkageLab', RidgeSpectralShrinkageLab)
    app.component('RocOperatingPointLab', RocOperatingPointLab)
    app.component('SeparationRegularizationLab', SeparationRegularizationLab)
    app.component('SoftmaxCompetitionConsoleLab', SoftmaxCompetitionConsole)
    app.component('SoftmaxReferenceGaugeLab', SoftmaxReferenceGauge)
    app.component('StochasticGradientLab', StochasticGradientLab)
    app.component('SvmMarginCaliperLab', SvmMarginCaliper)
    app.component('SvmDualBalanceBenchLab', SvmDualBalanceBench)
    app.component('TreeRouteManifoldLab', TreeRouteManifold)
    app.component('TreeSplitFoundryLab', TreeSplitFoundry)
    app.component('DynamicProgrammingPulseGridLab', DynamicProgrammingPulseGrid)
    app.component('ValidationSelectionLab', ValidationSelectionLab)
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
