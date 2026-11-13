# File: small_dataset_evaluation.py

"""
Small Dataset Evaluation Strategies
===================================
Multiple evaluation strategies for reliable performance estimation,
especially on small datasets.

Strategies:
- Standard k-fold CV
- Repeated k-fold CV
- Bootstrap .632+
"""

import numpy as np
from typing import Dict, List, Optional
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor

from knowledge_layer.Metrics.regression_metrics_class import RegressionMetrics


class SmallDatasetEvaluator:
    """
    Evaluation strategy provider for regression models.

    Provides multiple evaluation methods optimized for different dataset sizes.
    Use via composition in analysis classes.

    **Works with both parametric models and Random Forest.**

    Methods
    -------
    standard_cv : Standard k-fold cross-validation
    repeated_cv : Repeated k-fold with different random splits
    bootstrap_632_plus : Bootstrap .632+ with bias correction

    Examples
    --------
    >>> evaluator = SmallDatasetEvaluator(random_state=42)
    >>>
    >>> # For parametric model
    >>> results = evaluator.standard_cv(X, y, model_type='linear', n_folds=10)
    >>>
    >>> # For Random Forest with custom params
    >>> rf_params = {'n_estimators': 100, 'max_depth': 10}
    >>> results = evaluator.bootstrap_632_plus(
    ...     X, y, 'random_forest',
    ...     n_bootstraps=100,
    ...     model_params=rf_params
    ... )
    """

    def __init__(self,
                 random_state: int = 42,
                 verbose: bool = True,
                 default_model_params: Optional[Dict[str, Dict]] = None):
        """
        Initialize evaluator.

        Parameters
        ----------
        random_state : int
            Random seed for reproducibility
        verbose : bool
            Print progress information
        default_model_params : dict, optional
            Default parameters for each model type.
            Format: {'random_forest': {...}, 'linear': {...}}
        """
        self.random_state = random_state
        self.verbose = verbose
        self.default_model_params = default_model_params or {}

    # ----------------------------------------------------------------------------
    # NEW HELPER — add to SmallDatasetEvaluator
    # ----------------------------------------------------------------------------

    def _resolve_n_params(self,
                          model_type: str,
                          X: np.ndarray,
                          params: Dict) -> int:
        """
        Determine the parameter count used by n_params-dependent metrics.

        Random Forests have no classical parameter count, and ensemble
        averaging means complexity does not scale with ``n_estimators``.
        The predictor count is used instead, which keeps Random Forest
        adjusted metrics on the same footing as the parametric models
        (whose branch already uses the actual feature count) and guarantees
        the adjusted-R² denominator ``(n - p - 1)`` stays positive for any
        dataset large enough to evaluate.

        This also removes the ``max_depth=None`` failure mode: sklearn uses
        ``None`` to mean "grow to full depth", and ``dict.get`` returns a
        stored ``None`` rather than the fallback, so any arithmetic on
        ``params.get('max_depth', 10)`` breaks when the key is present.

        :param model_type: Model identifier, e.g. ``'random_forest'``,
            ``'linear'``, ``'mfp'``.
        :type model_type: str
        :param X: Predictor matrix; only its shape is used.
        :type X: np.ndarray
        :param params: Merged model parameters (defaults overridden by
            caller-supplied ``model_params``). Unused for Random Forest but
            kept in the signature for future model types.
        :type params: dict
        :returns: Parameter count for adjusted metrics.
        :rtype: int
        """
        if model_type == 'random_forest':
            # Predictor count — consistent with the parametric branch and
            # always < n_samples for any evaluable dataset.
            return int(X.shape[1]) if X.ndim > 1 else 1

        from knowledge_layer.parametric_regression.parametric_model_factory import ModelFactory
        X_sample = X[:20] if len(X) > 20 else X
        return ModelFactory.get_n_parameters(model_type, X_sample)

    # ========================================================================
    # STANDARD CV
    # ========================================================================

    def standard_cv(self,
                    X: np.ndarray,
                    y: np.ndarray,
                    model_type: str,
                    n_folds: int = 10,
                    model_params: dict = None,
                    reference_scale: float | None = None,
                    min_coverage: float = 0.8) -> Optional[Dict]:
        """
        Single-partition k-fold cross-validation, scored on pooled out-of-fold
        predictions.

        Each observation is held out exactly once, so the k folds together yield
        one prediction per observation. Metrics are computed once on that pooled
        vector rather than per fold, because R-squared normalises by the sum of
        squares of the set it is computed on: on a fold of a handful of samples
        that denominator is dominated by which observations happened to be grouped
        together rather than by model quality.

        No dispersion is returned. A single partition gives a single estimate, and
        the fold-to-fold standard deviation reported previously measured the
        unevenness of the shuffle, not uncertainty in the performance estimate.

        :param min_coverage: Minimum fraction of observations that must receive a
            prediction for the result to be returned. A partition in which several
            folds failed to fit would otherwise be scored on a subset selected by
            which fits were well conditioned, which is not a random subset.
        :returns: Metric dictionary, or ``None`` if coverage was insufficient.
        """
        n_samples = len(X)

        params = self.default_model_params.get(model_type, {}).copy()
        if model_params:
            params.update(model_params)

        effective_folds = n_samples if n_samples < n_folds else n_folds

        if self.verbose:
            print(f"      Running {effective_folds}-fold CV (pooled)...")

        kf = KFold(n_splits=effective_folds, shuffle=True,
                   random_state=self.random_state)

        oof_pred = np.full(n_samples, np.nan)
        n_failed = 0

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]

            try:
                if model_type == 'random_forest':
                    model = RandomForestRegressor(**params)
                    model.fit(X_train, y_train)
                    oof_pred[test_idx] = model.predict(X_test)

                else:
                    from knowledge_layer.parametric_regression.parametric_model_factory import ModelFactory

                    model, preprocessor = ModelFactory.create_and_fit(
                        model_type, X_train, y_train
                    )
                    oof_pred[test_idx] = ModelFactory.predict(
                        model, preprocessor, X_test, model_type
                    )

            except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                n_failed += 1
                if self.verbose:
                    print(f"        Fold {fold_idx + 1} failed: {str(e)[:120]}")
                continue

        predicted = ~np.isnan(oof_pred)
        coverage = predicted.sum() / n_samples

        if coverage < min_coverage:
            if self.verbose:
                print(f"        → discarded: {coverage:.0%} coverage "
                      f"({n_failed} folds failed)")
            return None

        metric_calc = RegressionMetrics(y[predicted], oof_pred[predicted],
                                        reference=reference_scale)
        metrics = metric_calc.calculate_cv_metrics()

        metrics['n_samples'] = n_samples
        metrics['coverage'] = coverage
        metrics['n_evaluations'] = 1
        metrics['method'] = f'{effective_folds}-fold-cv-pooled'

        if self.verbose:
            print(f"        → {effective_folds - n_failed}/{effective_folds} "
                  f"folds completed, {coverage:.0%} coverage")

        return metrics

    # ========================================================================
    # REPEATED CV
    # ========================================================================

    def repeated_cv(self,
                    X: np.ndarray,
                    y: np.ndarray,
                    model_type: str,
                    n_folds: int = 5,
                    n_repeats: int = 10,
                    model_params: dict = None,
                    reference_scale: float | None = None,
                    min_coverage: float = 0.8) -> Optional[Dict]:
        """
        Repeated k-fold cross-validation, scored on pooled out-of-fold predictions.

        One metric set is produced per repeat, computed on the full out-of-fold
        vector of that partition. The dispersion returned is therefore the spread
        across partitions: it quantifies how much the estimate depends on which
        observations were grouped together, not how much it would change under a
        different sample of vessels.

        :param min_coverage: Minimum fraction of observations that must receive a
            prediction for a repeat to be retained.
        :returns: Metrics aggregated over surviving repeats, or ``None``.
        """
        n_samples = len(X)

        params = self.default_model_params.get(model_type, {}).copy()
        if model_params:
            params.update(model_params)

        if n_samples < n_folds:
            effective_folds = max(2, n_samples // 2)
        else:
            effective_folds = n_folds

        if self.verbose:
            print(f"      Running {effective_folds}-fold CV × {n_repeats} repeats "
                  f"({n_repeats} pooled evaluations)...")

        repeat_metrics = []

        for repeat_idx in range(n_repeats):
            kf = KFold(n_splits=effective_folds, shuffle=True,
                       random_state=self.random_state + repeat_idx)

            oof_pred = np.full(n_samples, np.nan)

            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train = y[train_idx]

                try:
                    if model_type == 'random_forest':
                        model = RandomForestRegressor(**params)
                        model.fit(X_train, y_train)
                        oof_pred[test_idx] = model.predict(X_test)

                    else:
                        from knowledge_layer.parametric_regression.parametric_model_factory import ModelFactory

                        model, preprocessor = ModelFactory.create_and_fit(
                            model_type, X_train, y_train
                        )
                        oof_pred[test_idx] = ModelFactory.predict(
                            model, preprocessor, X_test, model_type
                        )

                except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                    if self.verbose:
                        print(f"        Repeat {repeat_idx + 1}, fold {fold_idx + 1} "
                              f"failed: {str(e)[:120]}")
                    continue

            predicted = ~np.isnan(oof_pred)
            coverage = predicted.sum() / n_samples

            if coverage < min_coverage:
                if self.verbose:
                    print(f"        Repeat {repeat_idx + 1} discarded "
                          f"({coverage:.0%} coverage)")
                continue

            metric_calc = RegressionMetrics(y[predicted], oof_pred[predicted],
                                            reference=reference_scale)
            metrics = metric_calc.calculate_cv_metrics()
            metrics['coverage'] = coverage
            repeat_metrics.append(metrics)

        if not repeat_metrics:
            return None

        avg_metrics = self._average_metrics(repeat_metrics)
        avg_metrics['n_samples'] = n_samples
        avg_metrics['n_evaluations'] = len(repeat_metrics)
        avg_metrics['method'] = f'{effective_folds}x{n_repeats}-repeated-cv-pooled'

        if self.verbose:
            print(f"        → {len(repeat_metrics)}/{n_repeats} repeats completed")

        return avg_metrics

    # ----------------------------------------------------------------------------
    # UPDATED: bootstrap_632_plus
    # ----------------------------------------------------------------------------

    def bootstrap_632_plus(self,
                           X: np.ndarray,
                           y: np.ndarray,
                           model_type: str,
                           n_bootstraps: int = 100,
                           model_params: dict = None,
                            reference_scale:float|None=None) -> dict:
        """
        Bootstrap .632+ evaluation with bias correction.

        Works for both parametric and Random Forest models.

        Bootstrap training error is optimistically biased (samples repeat)
        while out-of-bag error is pessimistically biased (test sets are
        small). The .632+ estimator blends them with a weight driven by the
        relative overfitting rate:

        ``err_632+ = w · err_oob + (1 − w) · err_train`` where
        ``w = 0.632 / (1 − 0.368 · R)``.

        :param X: Predictor data.
        :type X: np.ndarray
        :param y: Target data.
        :type y: np.ndarray
        :param model_type: Model type (``'linear'``, ``'random_forest'``,
            ``'mfp'``, ...).
        :type model_type: str
        :param n_bootstraps: Number of bootstrap resamples.
        :type n_bootstraps: int
        :param model_params: Model-specific parameter overrides. May contain
            ``None`` values (e.g. sklearn's ``max_depth=None``); these are
            handled safely.
        :type model_params: dict, optional
        :returns: Metrics with .632+ correction, or ``None`` if no bootstrap
            sample produced enough OOB observations.
        :rtype: dict or None
        """
        n_samples = len(X)

        if self.verbose:
            print(f"      Running Bootstrap .632+ ({n_bootstraps} samples)...")

        # Merge parameters (caller overrides defaults)
        params = self.default_model_params.get(model_type, {}).copy()
        if model_params:
            params.update(model_params)

        # ✅ FIX: None-safe parameter count on a consistent footing with the
        #    parametric branch. Previously:
        #        params.get('max_depth', 10) * params.get('n_estimators', 100)
        #    which raised TypeError for max_depth=None and could exceed
        #    n_samples, corrupting adjusted metrics.
        n_params = self._resolve_n_params(model_type, X, params)

        # ✅ Seeded resampling → reproducible bootstrap results across reruns
        rng = np.random.RandomState(self.random_state)

        oob_metrics_list = []
        train_metrics_list = []

        for i in range(n_bootstraps):
            bootstrap_idx = rng.choice(n_samples, size=n_samples, replace=True)
            # ✅ O(n log n) instead of O(n²) list-membership scan
            oob_idx = np.setdiff1d(np.arange(n_samples), bootstrap_idx,
                                   assume_unique=False)

            if len(oob_idx) < 5:  # Need minimum OOB samples
                continue

            X_boot = X[bootstrap_idx]
            y_boot = y[bootstrap_idx]
            X_oob = X[oob_idx]
            y_oob = y[oob_idx]

            try:
                if model_type == 'random_forest':
                    model = RandomForestRegressor(**params)
                    model.fit(X_boot, y_boot)
                    y_train_pred = model.predict(X_boot)
                    y_oob_pred = model.predict(X_oob)

                else:
                    from knowledge_layer.parametric_regression.parametric_model_factory import ModelFactory

                    model, preprocessor = ModelFactory.create_and_fit(
                        model_type, X_boot, y_boot
                    )
                    y_train_pred = ModelFactory.predict(model, preprocessor, X_boot, model_type)
                    y_oob_pred = ModelFactory.predict(model, preprocessor, X_oob, model_type)

                train_calc = RegressionMetrics(y_boot, y_train_pred, reference=reference_scale)
                train_metrics_list.append(train_calc.calculate_predictive_metrics(n_params))

                oob_calc = RegressionMetrics(y_oob, y_oob_pred, reference=reference_scale)
                oob_metrics_list.append(oob_calc.calculate_predictive_metrics(n_params))

            except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                if self.verbose and i < 5:
                    print(f"        Bootstrap {i + 1} failed: {str(e)[:50]}")
                continue

        if not oob_metrics_list:
            return None

        result = self._calculate_632_plus(train_metrics_list, oob_metrics_list, X, y, n_params)

        if self.verbose:
            print(f"        → {len(oob_metrics_list)}/{n_bootstraps} bootstrap samples completed")

        return result

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _calculate_632_plus(self,
                            train_metrics: List[Dict],
                            oob_metrics: List[Dict],
                            X: np.ndarray,
                            y: np.ndarray,
                            n_params: int) -> Dict:
        """
        Calculate .632+ bootstrap correction.

        **Concept:**
        Bootstrap samples are optimistically biased (training error too low).
        OOB samples are pessimistically biased (test error too high).
        .632+ finds optimal weight between them using relative overfitting rate.

        Parameters
        ----------
        train_metrics : list of dict
            Metrics on bootstrap training sets
        oob_metrics : list of dict
            Metrics on OOB test sets
        X : np.ndarray
            Full predictor data
        y : np.ndarray
            Full target data
        n_params : int
            Number of model parameters

        Returns
        -------
        dict
            .632+ corrected metrics with decomposition
        """
        # Average training and OOB metrics
        train_avg = self._average_metrics(train_metrics)
        oob_avg = self._average_metrics(oob_metrics)

        # Calculate no-information error rate (gamma)
        # This is the error when X and y are independent
        y_perm = np.random.permutation(y)
        no_info_calc = RegressionMetrics(y, y_perm)
        no_info_metrics = no_info_calc.calculate_all_metrics(n_params)
        gamma_mse = no_info_metrics['rmse'] ** 2

        # Calculate relative overfitting rate (R)
        oob_mse = oob_avg['rmse'] ** 2
        train_mse = train_avg['rmse'] ** 2

        if gamma_mse > train_mse:
            R = (oob_mse - train_mse) / (gamma_mse - train_mse)
            R = np.clip(R, 0, 1)  # Keep in [0,1]
        else:
            R = 0.0

        # Calculate .632+ weight
        w = 0.632 / (1 - 0.368 * R)

        # Calculate .632+ corrected estimates for all metrics
        corrected = {}

        for metric in ['r2', 'adj_r2', 'rrmse', 'mape', 'mae', 'rmse']:
            train_val = train_avg.get(metric, np.nan)
            oob_val = oob_avg.get(metric, np.nan)

            if not np.isnan(train_val) and not np.isnan(oob_val):
                # .632+ correction
                # For R² (higher is better), formula is same
                # For error metrics (lower is better), formula is same
                corrected[f'{metric}_632plus'] = w * oob_val + (1 - w) * train_val

                # Store components for transparency
                corrected[f'{metric}_train'] = train_val
                corrected[f'{metric}_oob'] = oob_val
                corrected[f'{metric}_std'] = oob_avg.get(f'{metric}_std', 0)
            else:
                corrected[f'{metric}_632plus'] = np.nan
                corrected[f'{metric}_train'] = np.nan
                corrected[f'{metric}_oob'] = np.nan

        # Store diagnostic information
        corrected['relative_overfitting'] = R
        corrected['bootstrap_weight'] = w
        corrected['n_evaluations'] = len(oob_metrics)
        corrected['gamma_rmse'] = np.sqrt(gamma_mse)  # No-information RMSE
        corrected['n_samples'] = len(X)
        corrected['method'] = f'bootstrap-632plus-{len(oob_metrics)}x'

        return corrected

    def _average_metrics(self, metrics_list: List[Dict]) -> Dict:
        """
        Average metric dictionaries across repeats or bootstrap replicates.

        Keys are taken from the dictionaries themselves rather than from a fixed
        list, so the aggregation follows whatever metric set the caller produced.
        Non-numeric and bookkeeping entries are skipped.

        Standard deviations use ``ddof=1``: the values being aggregated are a
        sample of possible partitions, not the population of them.
        """
        if not metrics_list:
            return {}

        skip = {'n_samples', 'n_features', 'n_evaluations', 'method', 'coverage'}
        keys = [k for k in metrics_list[0]
                if k not in skip and isinstance(metrics_list[0][k], (int, float, np.floating))]

        avg_metrics = {}

        for key in keys:
            values = [m[key] for m in metrics_list
                      if key in m and not np.isnan(m[key])]

            if values:
                avg_metrics[key] = float(np.mean(values))
                avg_metrics[f'{key}_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            else:
                avg_metrics[key] = np.nan
                avg_metrics[f'{key}_std'] = np.nan

        if 'coverage' in metrics_list[0]:
            avg_metrics['mean_coverage'] = float(
                np.mean([m['coverage'] for m in metrics_list])
            )

        return avg_metrics