"""
Held-out test-set metrics for a fitted ParametricModelSelector.

Suggested location: ``knowledge_layer/model_selectors/holdout_metric_evaluator.py``
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from knowledge_layer.parametric_regression.parametric_model_factory import ModelFactory
from knowledge_layer.Metrics.regression_metrics_class import RegressionMetrics


PrepareFn = Callable[..., "tuple[np.ndarray, np.ndarray] | None"]


class HoldoutMetricEvaluator:
    """
    Recompute performance metrics for a selector's recommended models on a
    fixed train/test partition.

    The selector's own ``detailed_results`` hold cross-validated (or in-sample)
    metrics on the full dataset. The domain-shift analysis instead trains each
    recommended model on a conventional *train* partition and reports error on
    the held-out *test* partition, normalised by the full conventional mean.
    This class reproduces that second procedure so a figure can show the same
    numbers the analysis reported.

    The class holds data and a fitting policy only; it does not plot. A selector
    receives an instance and asks it for metrics (composition), so the selector
    needs no knowledge of how the split was made.

    Two constructors cover the two use cases:

    * :meth:`from_analyzer` takes the partitions directly from a fitted
      ``ParametricDomainAnalyzer``. The split, row cleaning and normalisation
      are then identical to the reported analysis by construction.
    * :meth:`from_stratified_split` rebuilds the same split recipe
      (DWT-quintile stratified, positional, seeded). It matches the analysis
      exactly only if ``data`` has the same rows in the same order as the
      analyzer's ``conventional_full``.

    :param train_data: Rows used to fit each model.
    :type train_data: pandas.DataFrame
    :param test_data: Held-out rows used to score each model.
    :type test_data: pandas.DataFrame
    :param reference_data: Rows whose target mean normalises NRMSE. ``None``
        uses ``train_data`` and ``test_data`` combined.
    :type reference_data: pandas.DataFrame | None
    :param prepare_fn: Callable ``(dataset, predictors, target, min_samples)``
        returning ``(X, y)`` or ``None``. ``None`` uses a plain ``dropna`` on the
        predictor and target columns.
    :type prepare_fn: Callable | None
    :param use_descriptor_fn: Callable ``(model_type) -> bool`` deciding whether
        a model is refitted with its stored, fixed structure (MFP,
        polynomial-lasso). ``None`` defers to the selector's
        ``_is_structure_discovering_model``.
    :type use_descriptor_fn: Callable | None
    :param min_train_samples: Minimum complete training rows per combination.
    :type min_train_samples: int
    :param min_test_samples: Minimum complete test rows per combination.
    :type min_test_samples: int
    :param source: Short description of where the split came from, used in
        debug output.
    :type source: str
    """

    def __init__(self,
                 train_data: pd.DataFrame,
                 test_data: pd.DataFrame,
                 reference_data: pd.DataFrame | None = None,
                 prepare_fn: PrepareFn | None = None,
                 use_descriptor_fn: Callable[[str], bool] | None = None,
                 min_train_samples: int = 10,
                 min_test_samples: int = 3,
                 source: str = 'custom'):
        self.train_data = train_data
        self.test_data = test_data
        self.reference_data = (reference_data if reference_data is not None
                               else pd.concat([train_data, test_data]))
        self.prepare_fn = prepare_fn
        self.use_descriptor_fn = use_descriptor_fn
        self.min_train_samples = min_train_samples
        self.min_test_samples = min_test_samples
        self.source = source
        self._cache: dict[tuple, dict | None] = {}

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_analyzer(cls, analyzer: Any) -> "HoldoutMetricEvaluator":
        """
        Build an evaluator from a fitted ``ParametricDomainAnalyzer``.

        Uses the analyzer's ``conventional_train``, ``conventional_test`` and
        ``conventional_full`` partitions, its ``foundation.prepare_dataset``
        cleaning and its ``_should_use_descriptor`` fitting policy, so the
        resulting metrics equal the analyzer's ``test_rrmse``.

        :param analyzer: Fitted domain analyzer.
        :type analyzer: ParametricDomainAnalyzer
        :returns: Evaluator bound to the analyzer's split.
        :rtype: HoldoutMetricEvaluator
        """
        prepare = getattr(analyzer.foundation, 'prepare_dataset', None)
        use_desc = getattr(analyzer, '_should_use_descriptor', None)
        return cls(
            train_data=analyzer.conventional_train,
            test_data=analyzer.conventional_test,
            reference_data=analyzer.conventional_full,
            prepare_fn=prepare,
            use_descriptor_fn=use_desc,
            source='analyzer',
        )

    @classmethod
    def from_stratified_split(cls,
                              data: pd.DataFrame,
                              test_size: float = 0.2,
                              random_state: int = 42,
                              stratify_col: str = 'DWT',
                              n_bins: int = 5,
                              debug: bool = False) -> "HoldoutMetricEvaluator":
        """
        Build an evaluator by recreating the analyzer's split recipe.

        Splits row *positions* (not values) with ``train_test_split``,
        stratified on ``pd.qcut(data[stratify_col], n_bins)``. Falls back to an
        unstratified split if stratification fails, as the analyzer does.

        :param data: Full dataset, e.g. ``selector.data``.
        :type data: pandas.DataFrame
        :param test_size: Fraction held out for testing.
        :type test_size: float
        :param random_state: Seed; use the analyzer's value.
        :type random_state: int
        :param stratify_col: Column whose quantile bins stratify the split.
        :type stratify_col: str
        :param n_bins: Number of quantile bins.
        :type n_bins: int
        :param debug: Print split sizes and the stratification outcome.
        :type debug: bool
        :returns: Evaluator bound to the new split.
        :rtype: HoldoutMetricEvaluator
        """
        from sklearn.model_selection import train_test_split

        positions = np.arange(len(data))
        stratified = False

        if stratify_col in data.columns:
            try:
                bins = pd.qcut(data[stratify_col].values, q=n_bins,
                               labels=False, duplicates='drop')
                train_idx, test_idx = train_test_split(
                    positions, test_size=test_size,
                    random_state=random_state, stratify=bins)
                stratified = True
            except Exception as exc:
                if debug:
                    print(f"[holdout] stratified split failed ({exc}); "
                          f"using random split")

        if not stratified:
            train_idx, test_idx = train_test_split(
                positions, test_size=test_size, random_state=random_state)

        if debug:
            print(f"[holdout] split: train={len(train_idx)}, "
                  f"test={len(test_idx)}, stratified={stratified}, "
                  f"random_state={random_state}")

        return cls(
            train_data=data.iloc[train_idx].copy(),
            test_data=data.iloc[test_idx].copy(),
            reference_data=data,
            source=f'stratified_split(seed={random_state})',
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_recommendations(self,
                                 selector: Any,
                                 targets: Sequence[str] | None = None,
                                 max_predictors: int | None = None,
                                 include_only_vars: Sequence[str] | None = None,
                                 debug: bool = False) -> dict[tuple, dict]:
        """
        Evaluate every recommended (combination, target) pair on the test set.

        :param selector: Fitted ``ParametricModelSelector``.
        :param targets: Restrict to these targets. ``None`` means all.
        :type targets: Sequence[str] | None
        :param max_predictors: Skip combinations with more predictors.
        :type max_predictors: int | None
        :param include_only_vars: Keep only combinations whose predictors all
            appear in this list.
        :type include_only_vars: Sequence[str] | None
        :param debug: Print one line per combination.
        :type debug: bool
        :returns: ``{(predictor_tuple, target): metrics_dict}``. Combinations
            that could not be evaluated are omitted.
        :rtype: dict[tuple, dict]
        """
        out: dict[tuple, dict] = {}

        if debug:
            print(f"[holdout] evaluating recommendations (source={self.source})")

        for (pred_combo, target), model_type in selector.model_recommendations.items():
            if targets is not None and target not in targets:
                continue
            if max_predictors is not None and len(pred_combo) > max_predictors:
                continue
            if include_only_vars is not None and not all(
                    v in include_only_vars for v in pred_combo):
                continue

            metrics = self.evaluate_combination(
                selector, pred_combo, target, model_type, debug=debug)
            if metrics is not None:
                out[(tuple(pred_combo), target)] = metrics

        if debug:
            print(f"[holdout] evaluated {len(out)} combinations")

        return out

    def evaluate_combination(self,
                             selector: Any,
                             pred_combo: Sequence[str],
                             target: str,
                             model_type: str,
                             debug: bool = False) -> dict | None:
        """
        Fit one model on the train partition and score it on the test partition.

        Results are cached per ``(combination, target, model_type)``, so
        repeated plots do not refit.

        :param selector: Fitted ``ParametricModelSelector`` (supplies stored
            descriptors and the piecewise configuration).
        :param pred_combo: Predictor names.
        :type pred_combo: Sequence[str]
        :param target: Target name.
        :type target: str
        :param model_type: Model type to fit.
        :type model_type: str
        :param debug: Print the outcome for this combination.
        :type debug: bool
        :returns: Predictive metrics on the test set (``r2``, ``adj_r2``,
            ``rmse``, ``rrmse``, ``mae``, ``mape``, ``n_samples``,
            ``n_features``), zero ``std_*`` entries, and ``n_train``; or ``None``
            if data was insufficient or fitting failed.
        :rtype: dict | None
        """
        key = (tuple(pred_combo), target)
        cache_key = (key, model_type)
        if cache_key in self._cache:
            return self._cache[cache_key]

        label = f"{' + '.join(pred_combo)} -> {target} [{model_type}]"
        predictors = list(pred_combo)

        train_prep = self._prepare(self.train_data, predictors, target,
                                   self.min_train_samples)
        test_prep = self._prepare(self.test_data, predictors, target,
                                  self.min_test_samples)

        if train_prep is None or test_prep is None:
            if debug:
                print(f"[holdout]   skip {label}: insufficient "
                      f"{'train' if train_prep is None else 'test'} data")
            self._cache[cache_key] = None
            return None

        X_train, y_train = train_prep
        X_test, y_test = test_prep

        try:
            model, prep = self._fit(selector, key, model_type, X_train, y_train)
            y_pred = ModelFactory.predict(model, prep, X_test, model_type)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            if debug:
                print(f"[holdout]   skip {label}: fit failed ({str(exc)[:60]})")
            self._cache[cache_key] = None
            return None

        try:
            n_params = ModelFactory.get_n_parameters(
                model_type, X_train, fitted_model=model)
        except Exception:
            n_params = X_train.shape[1]

        reference = float(self.reference_data[target].dropna().mean())
        metrics = RegressionMetrics(
            np.asarray(y_test), np.asarray(y_pred), reference=reference
        ).calculate_predictive_metrics(n_params)

        # A single split has no spread; zero keeps error bars off.
        for name in ('r2', 'adj_r2', 'rrmse', 'rmse', 'mae', 'mape'):
            metrics[f'std_{name}'] = 0.0
        metrics['n_train'] = len(y_train)
        metrics['evaluation'] = 'holdout'

        if debug:
            print(f"[holdout]   {label}: test NRMSE={metrics['rrmse']:.3f}% "
                  f"(n_train={len(y_train)}, n_test={len(y_test)}, "
                  f"ref mean={reference:.4g})")

        self._cache[cache_key] = metrics
        return metrics

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _prepare(self, dataset: pd.DataFrame, predictors: list[str],
                 target: str, min_samples: int):
        """
        Return clean ``(X, y)`` arrays or ``None``.

        :param dataset: Partition to draw from.
        :type dataset: pandas.DataFrame
        :param predictors: Predictor columns.
        :type predictors: list[str]
        :param target: Target column.
        :type target: str
        :param min_samples: Minimum complete rows required.
        :type min_samples: int
        :rtype: tuple[numpy.ndarray, numpy.ndarray] | None
        """
        if self.prepare_fn is not None:
            return self.prepare_fn(dataset, predictors, target, min_samples)

        cols = predictors + [target]
        if not all(c in dataset.columns for c in cols):
            return None
        subset = dataset[cols].dropna()
        if len(subset) < min_samples:
            return None
        return subset[predictors].values, subset[target].values

    def _fit(self, selector: Any, key: tuple, model_type: str,
             X_train: np.ndarray, y_train: np.ndarray):
        """
        Fit one model on the training partition, using the same policy as the
        domain analyzer.

        Structure-discovering models (MFP, polynomial-lasso) are refitted with
        the structure found by the selector, so the test score evaluates the
        model that was actually selected. The stored descriptor is deep-copied
        before refitting so the selector's full-data coefficients are left
        untouched.

        :returns: ``(model, preprocessor)``
        :rtype: tuple
        """
        use_descriptor = (self.use_descriptor_fn(model_type)
                          if self.use_descriptor_fn is not None
                          else selector._is_structure_discovering_model(model_type))

        descriptor = selector.model_descriptors.get(key, {}).get(model_type)

        if use_descriptor and descriptor is not None:
            return copy.deepcopy(descriptor).refit(
                X=X_train, y=y_train, force_structure=True)

        if model_type.endswith('-piecewise'):
            return ModelFactory.create_and_fit(
                model_type, X_train, y_train,
                bins=selector.piecewise_bins,
                binning_variable=key[0][0])

        return ModelFactory.create_and_fit(model_type, X_train, y_train)