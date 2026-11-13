from dataclasses import dataclass, field
from typing import Mapping
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScaleReference:
    """
    Immutable per-target reference scale for denominator-fixed error metrics.

    **Concept:**
    Relative error metrics divide by a measure of target magnitude. When that
    divisor is taken from the evaluation sample itself, error values computed on
    different cohorts are not commensurable: a cohort whose vessels are larger
    receives a smaller relative error for the same absolute error. Holding the
    divisor fixed at a single reference value renders the metric a linear
    rescaling of RMSE, comparable across cohorts and across targets.

    The reference is a units constant. It is never fitted, never varies by
    bootstrap replicate, and cannot influence model predictions.

    :param means: Mapping of target dimension name to reference mean.
    :param source: Human-readable provenance of the reference, recorded so that
        the scale used can be reported in table captions.
    """

    means: Mapping[str, float] = field(default_factory=dict)
    source: str = "conventional_cohort"

    @classmethod
    def from_frame(cls, data: pd.DataFrame, targets: list[str],
                   source: str = "conventional_cohort") -> "ScaleReference":
        """
        Build a reference from a cohort frame, using complete cases per target.

        :param data: Cohort from which reference means are drawn.
        :param targets: Target dimensions requiring a reference scale.
        :param source: Provenance label carried into reporting.
        :raises ValueError: If a target is absent or has no finite values.
        :returns: Populated, immutable reference.
        """
        means: dict[str, float] = {}
        for target in targets:
            if target not in data.columns:
                raise ValueError(
                    f"Target '{target}' absent from reference cohort. "
                    f"Available columns: {sorted(data.columns)}"
                )
            values = pd.to_numeric(data[target], errors="coerce").to_numpy()
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                raise ValueError(
                    f"Target '{target}' has no finite values in reference cohort; "
                    f"cannot establish a reference scale."
                )
            means[target] = float(np.mean(finite))
        return cls(means=means, source=source)

    def for_target(self, target: str) -> float:
        """
        Retrieve the reference scale for one target.

        :param target: Target dimension name.
        :raises KeyError: If no reference was established for the target.
        :returns: Reference mean.
        """
        if target not in self.means:
            raise KeyError(
                f"No reference scale for '{target}'. Established targets: "
                f"{sorted(self.means)}. Was ScaleReference built with this target?"
            )
        return self.means[target]