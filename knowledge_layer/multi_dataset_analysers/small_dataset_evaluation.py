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

    # ========================================================================
    # STANDARD CV
    # ========================================================================

    def standard_cv(self,
                    X: np.ndarray,
                    y: np.ndarray,
                    model_type: str,
                    n_folds: int = 10,
                    model_params: Optional[Dict] = None) -> Optional[Dict]:  # ✅ ADDED
        """
        Standard k-fold cross-validation.

        Single pass through data with k folds.

        Parameters
        ----------
        X : np.ndarray
            Predictor variables
        y : np.ndarray
            Target variable
        model_type : str
            Model type ('linear', 'random_forest', 'mfp', etc.)
        n_folds : int, default=10
            Number of folds
        model_params : dict, optional  # ✅ NEW
            Model-specific parameters to override defaults

        Returns
        -------
        dict
            Performance metrics with std dev across folds
        """
        n_samples = len(X)

        # ✅ Merge parameters
        params = self.default_model_params.get(model_type, {}).copy()
        if model_params:
            params.update(model_params)

        # Adjust folds if necessary
        if n_samples < n_folds:
            effective_folds = n_samples  # LOO
        else:
            effective_folds = n_folds

        if self.verbose:
            print(f"      Running {effective_folds}-fold CV...")

        kf = KFold(n_splits=effective_folds, shuffle=True, random_state=self.random_state)

        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            try:
                # ✅ Model-specific fitting with parameters
                if model_type == 'random_forest':
                    model = RandomForestRegressor(**params)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    n_features = params.get('max_depth', 10)  # Proxy for complexity

                else:
                    # Parametric models
                    from knowledge_layer.parametric_regression.parametric_model_factory import ModelFactory

                    model, preprocessor = ModelFactory.create_and_fit(
                        model_type, X_train, y_train
                    )
                    y_pred = ModelFactory.predict(model, preprocessor, X_test, model_type)
                    n_features = ModelFactory.get_actual_n_features(model_type, X_train)

                # Calculate metrics
                metric_calc = RegressionMetrics(y_test, y_pred)
                metrics = metric_calc.calculate_all_metrics(n_features)

                fold_metrics.append(metrics)

            except Exception as e:
                if self.verbose:
                    print(f"        Fold {fold_idx + 1} failed: {str(e)[:50]}")
                continue

        if not fold_metrics:
            return None

        # Average metrics
        avg_metrics = self._average_metrics(fold_metrics)
        avg_metrics['n_samples'] = n_samples
        avg_metrics['n_evaluations'] = len(fold_metrics)
        avg_metrics['method'] = f'{effective_folds}-fold-cv'

        if self.verbose:
            print(f"        → {len(fold_metrics)}/{effective_folds} folds completed")

        return avg_metrics

    # ========================================================================
    # REPEATED CV
    # ========================================================================

    def repeated_cv(self,
                    X: np.ndarray,
                    y: np.ndarray,
                    model_type: str,
                    n_folds: int = 5,
                    n_repeats: int = 10,
                    model_params: Optional[Dict] = None) -> Optional[Dict]:  # ✅ ADDED
        """
        Repeated k-fold cross-validation.

        Performs k-fold CV multiple times with different random splits
        to obtain more stable performance estimates.

        Parameters
        ----------
        X : np.ndarray
            Predictor variables
        y : np.ndarray
            Target variable
        model_type : str
            Model type ('linear', 'random_forest', 'mfp', etc.)
        n_folds : int, default=5
            Number of folds per repetition
        n_repeats : int, default=10
            Number of repetitions
        model_params : dict, optional  # ✅ NEW
            Model-specific parameters to override defaults

        Returns
        -------
        dict
            Performance metrics averaged across all folds×repeats

        Notes
        -----
        - Total evaluations = n_folds × n_repeats
        - More stable than single CV run
        - Each sample appears in test set multiple times
        """
        n_samples = len(X)

        # ✅ Merge parameters
        params = self.default_model_params.get(model_type, {}).copy()
        if model_params:
            params.update(model_params)

        # Adjust folds if necessary
        if n_samples < n_folds:
            effective_folds = max(2, n_samples // 2)
        else:
            effective_folds = n_folds

        if self.verbose:
            print(f"      Running {effective_folds}-fold CV × {n_repeats} repeats "
                  f"({effective_folds * n_repeats} evaluations)...")

        all_metrics = []

        for repeat_idx in range(n_repeats):
            kf = KFold(
                n_splits=effective_folds,
                shuffle=True,
                random_state=self.random_state + repeat_idx
            )

            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                try:
                    # ✅ Model-specific fitting with parameters
                    if model_type == 'random_forest':
                        model = RandomForestRegressor(**params)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        n_features = params.get('max_depth', 10)

                    else:
                        # Parametric models
                        from knowledge_layer.parametric_regression.parametric_model_factory import ModelFactory

                        model, preprocessor = ModelFactory.create_and_fit(
                            model_type, X_train, y_train
                        )
                        y_pred = ModelFactory.predict(model, preprocessor, X_test, model_type)
                        n_features = ModelFactory.get_actual_n_features(model_type, X_train)

                    # Calculate metrics
                    metric_calc = RegressionMetrics(y_test, y_pred)
                    metrics = metric_calc.calculate_all_metrics(n_features)

                    all_metrics.append(metrics)

                except Exception as e:
                    if self.verbose:
                        print(f"        Repeat {repeat_idx + 1}, fold {fold_idx + 1} failed: "
                              f"{str(e)[:50]}")
                    continue

        if not all_metrics:
            return None

        # Average metrics
        avg_metrics = self._average_metrics(all_metrics)
        avg_metrics['n_samples'] = n_samples
        avg_metrics['n_evaluations'] = len(all_metrics)
        avg_metrics['method'] = f'{effective_folds}-fold-cv-{n_repeats}x'

        if self.verbose:
            print(f"        → {len(all_metrics)}/{effective_folds * n_repeats} evaluations completed")

        return avg_metrics

    # ========================================================================
    # BOOTSTRAP .632+
    # ========================================================================

    def bootstrap_632_plus(self,
                           X: np.ndarray,
                           y: np.ndarray,
                           model_type: str,
                           n_bootstraps: int = 100,
                           model_params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Bootstrap .632+ evaluation with bias correction.

        **Works for both parametric and Random Forest models.**

        Parameters
        ----------
        X : np.ndarray
            Predictor data
        y : np.ndarray
            Target data
        model_type : str
            Model type ('linear', 'random_forest', 'mfp', etc.)
        n_bootstraps : int, default=100
            Number of bootstrap samples
        model_params : dict, optional
            Model-specific parameters

        Returns
        -------
        dict or None
            Metrics with .632+ correction

        Notes
        -----
        **Bootstrap .632+ Concept:**

        Bootstrap training error is optimistically biased (too low) because
        samples are repeated. OOB error is pessimistically biased (too high)
        because test sets are small. The .632+ estimator finds optimal weight
        between them based on relative overfitting rate.

        Formula: err_632+ = w × err_oob + (1-w) × err_train
        where w = 0.632 / (1 - 0.368 × R)
        and R is relative overfitting rate
        """
        n_samples = len(X)

        if self.verbose:
            print(f"      Running Bootstrap .632+ ({n_bootstraps} samples)...")

        # ✅ Merge parameters
        params = self.default_model_params.get(model_type, {}).copy()
        if model_params:
            params.update(model_params)

        # Get n_parameters for metrics calculation
        if model_type == 'random_forest':
            # RF doesn't have traditional parameter count
            # Use a proxy based on complexity
            n_params = params.get('max_depth', 10) * params.get('n_estimators', 100)
        else:
            # For parametric models, get actual parameter count
            from knowledge_layer.parametric_regression.parametric_model_factory import ModelFactory
            # Use a small sample to estimate parameter count
            if len(X) > 20:
                X_sample = X[:20]
            else:
                X_sample = X
            n_params = ModelFactory.get_n_parameters(model_type, X_sample)

        oob_metrics_list = []
        train_metrics_list = []

        for i in range(n_bootstraps):
            # Bootstrap sample
            bootstrap_idx = np.random.choice(n_samples, size=n_samples, replace=True)
            oob_idx = np.array([j for j in range(n_samples) if j not in bootstrap_idx])

            if len(oob_idx) < 5:  # Need minimum OOB samples
                continue

            X_boot = X[bootstrap_idx]
            y_boot = y[bootstrap_idx]
            X_oob = X[oob_idx]
            y_oob = y[oob_idx]

            try:
                # ✅ Model-specific fitting with parameters
                if model_type == 'random_forest':
                    model = RandomForestRegressor(**params)
                    model.fit(X_boot, y_boot)

                    # Training predictions
                    y_train_pred = model.predict(X_boot)
                    # OOB predictions
                    y_oob_pred = model.predict(X_oob)

                else:
                    # Parametric models
                    from knowledge_layer.parametric_regression.parametric_model_factory import ModelFactory

                    model, preprocessor = ModelFactory.create_and_fit(
                        model_type, X_boot, y_boot
                    )

                    # Training predictions
                    y_train_pred = ModelFactory.predict(model, preprocessor, X_boot, model_type)
                    # OOB predictions
                    y_oob_pred = ModelFactory.predict(model, preprocessor, X_oob, model_type)

                # Calculate metrics
                # Training metrics
                train_calc = RegressionMetrics(y_boot, y_train_pred)
                train_metrics = train_calc.calculate_predictive_metrics(n_params)
                train_metrics_list.append(train_metrics)

                # OOB metrics
                oob_calc = RegressionMetrics(y_oob, y_oob_pred)
                oob_metrics = oob_calc.calculate_predictive_metrics(n_params)
                oob_metrics_list.append(oob_metrics)

            except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                if self.verbose and i < 5:  # Only print first few errors
                    print(f"        Bootstrap {i + 1} failed: {str(e)[:50]}")
                continue

        if not oob_metrics_list:
            return None

        # Calculate .632+ correction
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
        """Average metrics across samples."""
        avg_metrics = {}

        metric_names = ['r2', 'adj_r2', 'rrmse', 'mape', 'mae', 'rmse']

        for metric in metric_names:
            values = [m[metric] for m in metrics_list
                      if not np.isnan(m.get(metric, np.nan))]

            if values:
                avg_metrics[metric] = np.mean(values)
                avg_metrics[f'{metric}_std'] = np.std(values)
            else:
                avg_metrics[metric] = np.nan
                avg_metrics[f'{metric}_std'] = np.nan

        return avg_metrics