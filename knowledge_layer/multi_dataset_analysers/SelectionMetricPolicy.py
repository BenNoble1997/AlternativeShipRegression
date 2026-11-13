from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class SelectionMetricPolicy:
    """
    Declare which metric governs each of the two model-selection decisions.

    Two decisions are made with different comparability properties:

    **Model form, within a predictor set** — all candidate forms are fitted
    to the same complete-case rows, so the sample size is constant and the
    log-likelihood sums are directly comparable. Information criteria
    (AIC/BIC) are valid and preferred here, since they penalise the
    parameter count that distinguishes the forms.

    **Predictor set, within a target** — candidates are fitted to different
    complete-case subsets, because predictors differ in missingness. A
    log-likelihood is a sum over observations, so a candidate fitted to
    fewer rows attains a lower AIC/BIC irrespective of fit quality. This
    biases information criteria toward the most sparsely reported
    predictor. Per-observation metrics (RRMSE, adjusted R²) do not carry
    that bias and should govern this decision.

    :param model_form_metric: Metric for the within-predictor-set decision.
    :type model_form_metric: str
    :param combination_metric: Metric for the across-predictor-set decision.
    :type combination_metric: str
    :raises ValueError: If either metric is unrecognised, or if an
        information criterion is requested for the combination decision.
    """

    model_form_metric: str = 'aic'
    combination_metric: str = 'rrmse'

    _VALID: ClassVar[frozenset[str]] = frozenset(
        {'aic', 'bic', 'rrmse', 'mape', 'r2', 'adj_r2'}
    )
    _INFORMATION_CRITERIA: ClassVar[frozenset[str]] = frozenset({'aic', 'bic'})
    _MINIMISED: ClassVar[frozenset[str]] = frozenset(
        {'aic', 'bic', 'rrmse', 'mape'}
    )

    def __post_init__(self) -> None:
        for name, metric in (('model_form_metric', self.model_form_metric),
                             ('combination_metric', self.combination_metric)):
            if metric not in self._VALID:
                raise ValueError(
                    f"{name}='{metric}' is not a recognised selection metric. "
                    f"Valid options: {sorted(self._VALID)}."
                )

        if self.combination_metric in self._INFORMATION_CRITERIA:
            raise ValueError(
                f"combination_metric='{self.combination_metric}' is an "
                f"information criterion. Predictor sets are fitted to "
                f"different complete-case subsets, so AIC/BIC favours the "
                f"predictor with the most missing data. Use one of "
                f"{sorted(self._VALID - self._INFORMATION_CRITERIA)}, or "
                f"score all candidates on a common subsample first."
            )

    def is_minimised(self, metric: str) -> bool:
        """
        Report whether lower values of a metric are better.

        :param metric: Metric name to classify.
        :type metric: str
        :returns: True if the metric is minimised, False if maximised.
        :rtype: bool
        """
        return metric in self._MINIMISED