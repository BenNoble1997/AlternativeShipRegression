from knowledge_layer.multi_dataset_analysers.domain_shift_plotting_class.domain_shift_plotter import (
    DomainShiftPlotter, DatasetMetrics, FourWayMetrics
)
from knowledge_layer.multi_dataset_analysers.domain_shift_plotting_class.prediction_scatter_plotter import (
    PredictionScatterPlotter, PredictionData
)
from knowledge_layer.multi_dataset_analysers.domain_shift_plotting_class.residual_plotter import (
    ResidualPlotter, ResidualData
)

class AnalysisPlotter:

    def __init__(self):
        self.domain_shift_plotter = DomainShiftPlotter()
        self.prediction_scatter_plotter = PredictionScatterPlotter()
        self.residual_plotter = ResidualPlotter()

