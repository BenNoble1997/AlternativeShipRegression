"""
Cross-Dataset Performance Analyzer
===================================
Analyzes how regression models transfer between conventional, matched,
and alternative fuel ship datasets.

Creates 3x3 performance matrices showing model performance when trained
on one dataset and tested on another.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

from config import LoggerMixin
from knowledge_layer.fleet_data_class.matched_dataset_creation import MatchedDataset
from knowledge_layer.parametric_regression.parametric_regression_model_factory_class import ModelFactory
from knowledge_layer.Metrics.regression_metrics_class import RegressionMetrics
from knowledge_layer.model_selectors.parametric_regresssion_selector_class import ParametricModelSelector


class CrossDatasetAnalyzer(LoggerMixin):
    """
    Analyzes model transferability across conventional, matched, and alternative datasets.

    **Key Difference from ParametricDomainAnalyzer:**
    - DomainAnalyzer: Trains ONCE on conventional, tests on 4 datasets
    - CrossDatasetAnalyzer: Trains on EACH of 3 datasets, tests on ALL 3

    This reveals:
    - Dataset-specific overfitting
    - Distribution shift effects
    - Fuel type transferability in both directions

    Workflow:
    ---------
    1. Get best model+predictors for each target from selector
    2. For each target:
       a. Get appropriate matched dataset for those predictors
       b. Train model on Conv → test on Conv, Matched, Alt
       c. Train model on Matched → test on Conv, Matched, Alt
       d. Train model on Alt → test on Conv, Matched, Alt
    3. Create 3x3 heatmap of performance
    """

    def __init__(self,
                 selector,
                 conventional_data: pd.DataFrame,
                 alternative_data: pd.DataFrame,
                 metric: str = 'rrmse',
                 verbose: bool = True):
        """
        Initialize Cross-Dataset Analyzer.

        **SIMPLIFIED:** Just pass the selector! Everything else is extracted automatically.

        Parameters
        ----------
        selector : ParametricModelSelector
            Fitted selector with results. If not fitted, will run analyze_all_combinations()
            automatically with default parameters.
        conventional_data : pd.DataFrame
            Full conventional ship dataset
        alternative_data : pd.DataFrame
            Alternative fuel ship dataset
        metric : str, default='rrmse'
            Metric to display in heatmaps: 'rrmse', 'r2', 'mape', 'rmse'
        verbose : bool
            Whether to print progress

        Examples
        --------
        >>> # Option 1: Pass fitted selector
        >>> selector.analyze_all_combinations(max_predictors=2)
        >>> analyzer = CrossDatasetAnalyzer(
        ...     selector=selector,
        ...     conventional_data=conv_data,
        ...     alternative_data=alt_data
        ... )

        >>> # Option 2: Pass unfitted selector (will auto-fit)
        >>> selector = ParametricModelSelector(...)
        >>> analyzer = CrossDatasetAnalyzer(
        ...     selector=selector,
        ...     conventional_data=conv_data,
        ...     alternative_data=alt_data
        ... )
        """
        self.selector = selector
        self.conventional = conventional_data.copy()
        self.alternative = alternative_data.copy()
        self.metric = metric
        self.verbose = verbose

        # Results storage
        self.results = {}  # {target: 3x3 performance matrix}
        self.best_combinations = {}  # {target: (predictors, model_type)}
        self.matched_datasets = {}  # Will be populated automatically

        # Validate inputs
        self._validate_inputs()

        # ✅ Extract or generate selector results
        self._ensure_selector_has_results()

        # ✅ Extract what we need from selector
        self.model_recommendations = self.selector.model_recommendations
        self.detailed_results = self.selector.detailed_results

        # ✅ AUTO-CREATE matched datasets (same as ParametricDomainAnalyzer)
        self._create_matched_datasets()

    def _validate_inputs(self):
        """Validate that inputs are properly formatted."""
        # Check selector type


        if not isinstance(self.selector, ParametricModelSelector):
            raise TypeError(
                f"selector must be a ParametricModelSelector instance, "
                f"got {type(self.selector).__name__}"
            )

        # Check metric is valid
        valid_metrics = ['r2', 'rrmse', 'rmse', 'mae', 'mape']
        if self.metric not in valid_metrics:
            raise ValueError(f"metric must be one of {valid_metrics}, got '{self.metric}'")

    def _ensure_selector_has_results(self):
        """
        Ensure selector has been fitted. If not, run analyze_all_combinations().

        This allows users to pass either:
        1. A fitted selector (already ran analyze_all_combinations)
        2. An unfitted selector (we'll fit it for them)
        """
        if not self.selector.model_recommendations:
            if self.verbose:
                print("\n" + "=" * 80)
                print("⚠️  Selector has no results - running analyze_all_combinations()")
                print("=" * 80)
                print("Tip: For custom analysis, run selector.analyze_all_combinations()")
                print("     with your desired parameters before creating the analyzer.\n")

            # Run with default parameters
            self.selector.analyze_all_combinations(
                max_predictors=2,
                cv_method='standard',
                n_folds=10,
                display_results=self.verbose
            )

            if not self.selector.model_recommendations:
                raise ValueError(
                    "Selector analysis produced no results! "
                    "Check your data and selector configuration."
                )

        if self.verbose:
            n_combos = len(self.selector.model_recommendations)
            targets = set(tgt for (_, tgt) in self.selector.model_recommendations.keys())
            print(f"\n✓ Using selector results: {n_combos} combinations across {len(targets)} targets")

    def _create_matched_datasets(self):
        """
        Create matched datasets for all predictor combinations.

        **Reuses logic from ParametricDomainAnalyzer._create_matched_sets()**
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("Creating matched datasets for all predictor combinations...")
            print("=" * 80)

        # Extract all unique predictor combinations from recommendations
        all_predictor_combos = set()
        for (pred_combo, target) in self.model_recommendations.keys():
            all_predictor_combos.add(pred_combo)

        if self.verbose:
            print(f"Found {len(all_predictor_combos)} unique predictor combinations")

        # Create matched datasets using MatchedDataset utility
        self.matched_datasets = MatchedDataset.multiple_matched_datasets(
            alternative_fleet=self.alternative,
            conventional_fleet=self.conventional,
            logger=self.logger,
            predictor_groups=list(all_predictor_combos),
            n_matches_per_ship=10,
            std_reference='conventional'
        )

        if self.verbose:
            print(f"✓ Created {len(self.matched_datasets)} matched datasets")
            print("=" * 80 + "\n")

    # ========================================================================
    # MAIN ANALYSIS METHOD
    # ========================================================================

    def analyze_all_targets(self,
                            targets: Optional[List[str]] = None,
                            selection_metric: str = 'adj_r2') -> pd.DataFrame:
        """
        Perform cross-dataset analysis for all targets.

        For each target:
        1. Get best predictor combination (by selection_metric)
        2. Get matched dataset for those predictors
        3. Train on each dataset (Conv, Matched, Alt)
        4. Test on all three datasets
        5. Store 3x3 performance matrix

        Parameters
        ----------
        targets : List[str], optional
            Specific targets to analyze. If None, analyzes all targets found
            in model_recommendations.
        selection_metric : str, default='adj_r2'
            Metric used to select "best" combination from selector results.
            Options: 'adj_r2', 'rrmse', 'bic', 'aic'

        Returns
        -------
        pd.DataFrame
            Summary results with one row per target showing best combination
            and all 9 cross-dataset performance values
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("CROSS-DATASET PERFORMANCE ANALYSIS")
            print("=" * 80)
            print(f"Performance metric: {self.metric.upper()}")
            print(f"Selection metric: {selection_metric}")

        # Get unique targets from recommendations
        all_targets = set(tgt for (_, tgt) in self.model_recommendations.keys())

        if targets is None:
            targets = sorted(all_targets)
        else:
            # Validate requested targets exist
            missing = set(targets) - all_targets
            if missing:
                raise ValueError(f"Targets not found in recommendations: {missing}")

        if self.verbose:
            print(f"Analyzing {len(targets)} targets\n")

        # Analyze each target
        summary_data = []

        for target in targets:
            if self.verbose:
                print(f"\n{'─' * 80}")
                print(f"Target: {target}")
                print(f"{'─' * 80}")

            result = self._analyze_single_target(target, selection_metric)

            if result is not None:
                summary_data.append(result)

        if not summary_data:
            if self.verbose:
                print("\n⚠️  No valid results generated!")
            return pd.DataFrame()

        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_data)

        if self.verbose:
            print("\n" + "=" * 80)
            print("ANALYSIS COMPLETE")
            print("=" * 80)
            print(f"✓ {len(summary_df)} targets analyzed")
            print(f"✓ Results stored in self.results")

        return summary_df

    def _analyze_single_target(self,
                               target: str,
                               selection_metric: str) -> Optional[Dict]:
        """
        Analyze a single target dimension.

        Returns
        -------
        dict or None
            Summary dict with target, predictors, model_type, and all 9
            cross-dataset performance values, or None if analysis failed
        """
        # Step 1: Get best combination for this target
        best_combo = self._get_best_combination(target, selection_metric)

        if best_combo is None:
            if self.verbose:
                print(f"  ⚠️  No valid combination found for {target}")
            return None

        predictors, model_type = best_combo
        self.best_combinations[target] = best_combo

        if self.verbose:
            print(f"  Best combination: {' + '.join(predictors)} → {model_type}")

        # Step 2: Get matched dataset for these predictors
        predictor_key = tuple(sorted(predictors))

        if predictor_key not in self.matched_datasets:
            if self.verbose:
                print(f"  ⚠️  No matched dataset found for predictors: {predictors}")
                print(f"     Available keys: {list(self.matched_datasets.keys())[:3]}...")
            return None

        matched_data = self.matched_datasets[predictor_key]

        if self.verbose:
            print(f"  Dataset sizes:")
            print(f"    Conventional: {len(self.conventional)}")
            print(f"    Matched:      {len(matched_data)}")
            print(f"    Alternative:  {len(self.alternative)}")

        # Step 3: Prepare clean data from all three datasets
        datasets = self._prepare_three_datasets(predictors, target, matched_data)

        if datasets is None:
            if self.verbose:
                print(f"  ⚠️  Data preparation failed")
            return None

        # Step 4: Perform 3x3 cross-validation
        performance_matrix = self._create_performance_matrix(
            datasets, model_type, predictors, target
        )

        if performance_matrix is None:
            return None

        # Step 5: Store results
        self.results[target] = {
            'predictors': predictors,
            'model_type': model_type,
            'performance_matrix': performance_matrix
        }

        # Step 6: Create summary dict
        summary = {
            'target': target,
            'predictors': ' + '.join(predictors),
            'n_predictors': len(predictors),
            'model_type': model_type,
        }

        # Add all 9 matrix values with descriptive names
        dataset_names = ['conv', 'matched', 'alt']
        for train_dataset in dataset_names:
            for test_dataset in dataset_names:
                key = f'{train_dataset}_train_{test_dataset}_test'
                value = performance_matrix.loc[test_dataset, train_dataset]
                summary[key] = value

        return summary

    # ========================================================================
    # COMBINATION SELECTION
    # ========================================================================

    def _get_best_combination(self,
                              target: str,
                              selection_metric: str) -> Optional[Tuple[Tuple[str, ...], str]]:
        """
        Get best predictor combination for a target from selector results.

        Parameters
        ----------
        target : str
            Target dimension
        selection_metric : str
            Metric to optimize ('adj_r2', 'rrmse', 'bic', 'aic')

        Returns
        -------
        tuple or None
            (predictors_tuple, model_type) or None if no valid combination
        """
        # Get all combinations for this target from selector results
        candidates = []

        for (pred_combo, tgt), model_type in self.model_recommendations.items():
            if tgt != target:
                continue

            # Get metrics from selector results
            key = (pred_combo, tgt)
            if key not in self.detailed_results:
                continue

            model_results = self.detailed_results[key]
            if model_type not in model_results:
                continue

            metrics = model_results[model_type]

            # Skip if metric not available
            if selection_metric not in metrics:
                continue

            candidates.append({
                'predictors': pred_combo,
                'model_type': model_type,
                'metric_value': metrics[selection_metric]
            })

        if not candidates:
            return None

        # Select best based on metric
        if selection_metric in ['adj_r2', 'r2']:
            # Higher is better
            best = max(candidates, key=lambda x: x['metric_value'])
        else:
            # Lower is better (rrmse, rmse, aic, bic, etc.)
            best = min(candidates, key=lambda x: x['metric_value'])

        return (best['predictors'], best['model_type'])

    # ========================================================================
    # DATA PREPARATION
    # ========================================================================

    def _prepare_three_datasets(self,
                                predictors: Tuple[str, ...],
                                target: str,
                                matched_data: pd.DataFrame) -> Optional[Dict]:
        """
        Prepare clean data from all three datasets.

        Parameters
        ----------
        predictors : tuple of str
            Predictor variable names
        target : str
            Target variable name
        matched_data : pd.DataFrame
            Matched conventional dataset for these predictors

        Returns
        -------
        dict or None
            {
                'conv': {'X': array, 'y': array},
                'matched': {'X': array, 'y': array},
                'alt': {'X': array, 'y': array}
            }
            or None if insufficient data
        """
        datasets = {}

        for dataset_name, data in [
            ('conv', self.conventional),
            ('matched', matched_data),
            ('alt', self.alternative)
        ]:
            # Extract features and target
            X = data[list(predictors)].values
            y = data[target].values

            # Clean data (remove NaN and inf)
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y) |
                     np.isinf(X).any(axis=1) | np.isinf(y))

            X_clean = X[mask]
            y_clean = y[mask]

            # Check minimum samples
            if len(X_clean) < 3:
                if self.verbose:
                    print(f"  ⚠️  Insufficient {dataset_name} data: {len(X_clean)} samples")
                return None

            datasets[dataset_name] = {'X': X_clean, 'y': y_clean}

            if self.verbose:
                print(f"    {dataset_name.capitalize():12s}: {len(X_clean):4d} clean samples")

        return datasets

    # ========================================================================
    # PERFORMANCE MATRIX CREATION
    # ========================================================================

    def _create_performance_matrix(self,
                                   datasets: Dict,
                                   model_type: str,
                                   predictors: Tuple[str, ...],
                                   target: str) -> Optional[pd.DataFrame]:
        """
        Create 3x3 performance matrix.

        Trains on each dataset (conv, matched, alt) and tests on all three.

        Parameters
        ----------
        datasets : dict
            Clean data from _prepare_three_datasets
        model_type : str
            Model type to use
        predictors : tuple
            Predictor names (for logging)
        target : str
            Target name (for logging)

        Returns
        -------
        pd.DataFrame or None
            3x3 matrix with index/columns = ['conv', 'matched', 'alt']
            Values = performance metric
        """
        dataset_names = ['conv', 'matched', 'alt']

        # Initialize matrix
        matrix = pd.DataFrame(
            index=dataset_names,
            columns=dataset_names,
            dtype=float
        )

        if self.verbose:
            print(f"\n  Training and testing:")

        # Train on each dataset
        for train_name in dataset_names:
            X_train = datasets[train_name]['X']
            y_train = datasets[train_name]['y']

            try:
                # Fit model
                model, preprocessor = ModelFactory.create_and_fit(
                    model_type, X_train, y_train
                )

                # Test on all datasets
                for test_name in dataset_names:
                    X_test = datasets[test_name]['X']
                    y_test = datasets[test_name]['y']

                    # Predict
                    y_pred = ModelFactory.predict(
                        model, preprocessor, X_test, model_type
                    )

                    # Calculate metric
                    metric_calc = RegressionMetrics(y_test, y_pred)

                    if self.metric == 'r2':
                        value = metric_calc.calculate_r2()
                    elif self.metric == 'rrmse':
                        value = metric_calc.calculate_rrmse()
                    elif self.metric == 'rmse':
                        value = metric_calc.calculate_rmse()
                    elif self.metric == 'mae':
                        value = metric_calc.calculate_mae()
                    elif self.metric == 'mape':
                        value = metric_calc.calculate_mape()
                    else:
                        value = np.nan

                    matrix.loc[test_name, train_name] = value

                    if self.verbose:
                        arrow = "→"
                        print(f"    {train_name:8s} {arrow} {test_name:8s}: "
                              f"{self.metric.upper()} = {value:.4f}")

            except Exception as e:
                if self.verbose:
                    print(f"  ⚠️  Training on {train_name} failed: {str(e)[:60]}")
                # Fill column with NaN
                matrix[train_name] = np.nan
                continue

        # Check if any valid results
        if matrix.isna().all().all():
            if self.verbose:
                print(f"  ⚠️  All training/testing combinations failed!")
            return None

        return matrix

    # ========================================================================
    # VISUALIZATION (keeping the previous implementation)
    # ========================================================================

    def plot_heatmaps(self,
                      targets: Optional[List[str]] = None,
                      figsize_per_plot: Tuple[float, float] = (6, 5),
                      cmap: str = 'RdYlGn_r',
                      save_path: Optional[str] = None,
                      annotation_format: str = '.2f',
                      use_global_scale: bool = False):  # ✅ NEW PARAMETER
        """
        Plot heatmaps for all analyzed targets.

        Parameters
        ----------
        targets : List[str], optional
            Specific targets to plot. If None, plots all analyzed targets.
        figsize_per_plot : tuple
            Size for each individual heatmap
        cmap : str
            Colormap name. Default 'RdYlGn_r' (reversed red-yellow-green)
            makes red=bad, green=good, which works for RRMSE.
            For R², use 'RdYlGn' (not reversed).
        save_path : str, optional
            Path to save figure
        annotation_format : str
            Format string for cell annotations (e.g., '.2f', '.3f', '.1f')
        use_global_scale : bool, default=False
            If True, uses same color scale across all heatmaps.
            If False, each heatmap gets its own scale (recommended).

        Notes
        -----
        **Color Scale Strategy:**
        - `use_global_scale=False` (default): Each target's heatmap uses its own
          min/max values. This highlights relative differences within each target,
          making it easier to see which training→testing combinations work best
          for that specific dimension.
        - `use_global_scale=True`: All heatmaps share the same color scale based
          on global min/max across all targets. This makes it easier to compare
          absolute performance across different targets, but can mask patterns
          within individual targets if one target has particularly bad performance.
        """
        if not self.results:
            print("No results to plot. Run analyze_all_targets() first.")
            return

        # Determine which targets to plot
        if targets is None:
            targets = sorted(self.results.keys())
        else:
            # Validate targets exist
            missing = set(targets) - set(self.results.keys())
            if missing:
                print(f"Warning: Targets not found in results: {missing}")
                targets = [t for t in targets if t in self.results]

        if not targets:
            print("No valid targets to plot!")
            return

        # Calculate grid layout
        n_targets = len(targets)
        n_cols = min(3, n_targets)
        n_rows = int(np.ceil(n_targets / n_cols))

        # Create figure
        fig_width = figsize_per_plot[0] * n_cols
        fig_height = figsize_per_plot[1] * n_rows

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(fig_width, fig_height))

        # Handle single plot case
        if n_targets == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten()

        # Adjust colormap based on metric
        if self.metric == 'r2':
            # Higher is better → use normal RdYlGn
            if cmap == 'RdYlGn_r':
                cmap = 'RdYlGn'

        # ✅ MODIFIED: Determine color scale strategy
        if use_global_scale:
            # OLD BEHAVIOR: Global vmin/vmax
            all_values = []
            for target in targets:
                matrix = self.results[target]['performance_matrix']
                all_values.extend(matrix.values.flatten())

            all_values = [v for v in all_values if not np.isnan(v)]

            if all_values:
                vmin_global = np.min(all_values)
                vmax_global = np.max(all_values)
            else:
                vmin_global, vmax_global = 0, 1

            if self.verbose:
                print(f"\nUsing global color scale: {vmin_global:.2f} to {vmax_global:.2f}")
        else:
            vmin_global, vmax_global = None, None  # Will be calculated per-target
            if self.verbose:
                print("\nUsing per-target color scales for better contrast")

        # Plot each target
        for idx, target in enumerate(targets):
            ax = axes[idx]

            result = self.results[target]
            matrix = result['performance_matrix']
            predictors = result['predictors']
            model_type = result['model_type']

            # ✅ NEW: Calculate per-target vmin/vmax if not using global scale
            if use_global_scale:
                vmin, vmax = vmin_global, vmax_global
            else:
                # Calculate min/max for this specific target
                target_values = matrix.values.flatten()
                target_values = target_values[~np.isnan(target_values)]

                if len(target_values) > 0:
                    vmin = np.min(target_values)
                    vmax = np.max(target_values)

                    if self.verbose:
                        print(f"  {target}: scale [{vmin:.2f}, {vmax:.2f}]")
                else:
                    vmin, vmax = 0, 1

            # Create heatmap
            sns.heatmap(
                matrix,
                annot=True,
                fmt=annotation_format,
                cmap=cmap,
                vmin=vmin,  # ✅ Now per-target by default
                vmax=vmax,  # ✅ Now per-target by default
                cbar=True,
                square=True,
                linewidths=1,
                linecolor='gray',
                ax=ax
            )

            # Formatting
            ax.set_xlabel('Training Dataset', fontsize=10, fontweight='bold')
            ax.set_ylabel('Testing Dataset', fontsize=10, fontweight='bold')

            # Title with target and combination
            title = f"{target}\n{' + '.join(predictors)} ({model_type})"
            ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

            # Rotate labels
            ax.set_xticklabels(['Conv', 'Matched', 'Alt'], rotation=0)
            ax.set_yticklabels(['Conv', 'Matched', 'Alt'], rotation=0)

        # Hide extra subplots
        for idx in range(n_targets, len(axes)):
            axes[idx].set_visible(False)

        # Super title
        metric_name = self.metric.upper()
        if self.metric == 'r2':
            metric_name = 'R²'
        elif self.metric == 'rrmse':
            metric_name = 'RRMSE (%)'

        # ✅ Updated title to indicate scale type
        scale_type = "Global Scale" if use_global_scale else "Per-Target Scales"
        fig.suptitle(
            f'Cross-Dataset Model Performance ({metric_name}) - {scale_type}',
            fontsize=14,
            fontweight='bold',
            y=0.995
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Heatmaps saved to {save_path}")

        plt.show()

    def plot_single_heatmap(self,
                            target: str,
                            figsize: Tuple[float, float] = (7, 6),
                            cmap: str = 'RdYlGn_r',
                            save_path: Optional[str] = None,
                            annotation_format: str = '.3f',
                            show_diagonal: bool = True):
        """
        Plot detailed heatmap for a single target with annotations.

        Parameters
        ----------
        target : str
            Target dimension to plot
        figsize : tuple
            Figure size
        cmap : str
            Colormap
        save_path : str, optional
            Path to save figure
        annotation_format : str
            Format for cell values
        show_diagonal : bool
            If True, highlights diagonal cells (train=test)
        """
        if target not in self.results:
            print(f"Target '{target}' not found in results!")
            print(f"Available targets: {list(self.results.keys())}")
            return

        result = self.results[target]
        matrix = result['performance_matrix']
        predictors = result['predictors']
        model_type = result['model_type']

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Adjust colormap for metric type
        if self.metric == 'r2' and cmap == 'RdYlGn_r':
            cmap = 'RdYlGn'

        # Create heatmap
        sns.heatmap(
            matrix,
            annot=True,
            fmt=annotation_format,
            cmap=cmap,
            cbar_kws={'label': self._get_metric_label()},
            square=True,
            linewidths=2,
            linecolor='white',
            ax=ax
        )

        # Highlight diagonal if requested
        if show_diagonal:
            for i in range(3):
                # Add rectangle around diagonal cells
                ax.add_patch(plt.Rectangle((i, i), 1, 1,
                                           fill=False,
                                           edgecolor='blue',
                                           lw=3))

        # Labels
        ax.set_xlabel('Training Dataset', fontsize=12, fontweight='bold')
        ax.set_ylabel('Testing Dataset', fontsize=12, fontweight='bold')

        ax.set_xticklabels(['Conventional', 'Matched', 'Alternative'],
                           rotation=45, ha='right')
        ax.set_yticklabels(['Conventional', 'Matched', 'Alternative'],
                           rotation=0)

        # Title
        title = (f"{target}\n"
                 f"Predictors: {' + '.join(predictors)}\n"
                 f"Model: {model_type}")
        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)

        # Add interpretation text
        interp_text = self._generate_interpretation(matrix, target)
        fig.text(0.5, -0.05, interp_text,
                 ha='center', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Heatmap saved to {save_path}")

        plt.show()

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _get_metric_label(self) -> str:
        """Get formatted label for metric."""
        labels = {
            'r2': 'R²',
            'rrmse': 'RRMSE (%)',
            'rmse': 'RMSE',
            'mae': 'MAE',
            'mape': 'MAPE (%)'
        }
        return labels.get(self.metric, self.metric.upper())

    def _generate_interpretation(self,
                                 matrix: pd.DataFrame,
                                 target: str) -> str:
        """
        Generate interpretation text for heatmap.

        Highlights key transferability insights.
        """
        # Diagonal = same dataset (train and test)
        diag_values = [matrix.loc[d, d] for d in ['conv', 'matched', 'alt']]

        # Off-diagonal = transfer
        conv_to_alt = matrix.loc['alt', 'conv']
        alt_to_conv = matrix.loc['conv', 'alt']

        matched_to_alt = matrix.loc['alt', 'matched']
        alt_to_matched = matrix.loc['matched', 'alt']

        if self.metric == 'r2':
            # Higher is better
            best_transfer = min(conv_to_alt, alt_to_conv)
            interpretation = (
                f"Best in-domain: {max(diag_values):.3f} | "
                f"Conv↔Alt transfer: {best_transfer:.3f}"
            )
        else:
            # Lower is better (RRMSE, etc.)
            best_transfer = max(conv_to_alt, alt_to_conv)
            interpretation = (
                f"Best in-domain: {min(diag_values):.2f}% | "
                f"Worst Conv↔Alt transfer: {best_transfer:.2f}%"
            )

        return interpretation

    # ========================================================================
    # EXPORT METHODS
    # ========================================================================

    def export_results(self,
                       filepath: str = 'cross_dataset_results.xlsx'):
        """
        Export all results to Excel with separate sheets.

        Parameters
        ----------
        filepath : str
            Output file path
        """
        if not self.results:
            print("No results to export!")
            return

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Sheet 1: Summary
            summary_data = []
            for target, result in self.results.items():
                matrix = result['performance_matrix']

                row = {
                    'Target': target,
                    'Predictors': ' + '.join(result['predictors']),
                    'Model': result['model_type'],
                }

                # Add all 9 matrix values
                for train in ['conv', 'matched', 'alt']:
                    for test in ['conv', 'matched', 'alt']:
                        col_name = f'{train.capitalize()}→{test.capitalize()}'
                        row[col_name] = matrix.loc[test, train]

                summary_data.append(row)

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

            # Sheets 2-N: Individual matrices
            for target, result in self.results.items():
                matrix = result['performance_matrix']

                # Add metadata rows
                metadata = pd.DataFrame({
                    'Info': ['Predictors', 'Model', ''],
                    'Value': [
                        ' + '.join(result['predictors']),
                        result['model_type'],
                        ''
                    ]
                })

                # Safe sheet name (Excel limit: 31 chars, no special chars)
                #

            sheet_name = target[:28].replace('/', '')
            # Write metadata then matrix
            startrow = 0
            metadata.to_excel(writer, sheet_name=sheet_name,
                              index=False, startrow=startrow)

            startrow = len(metadata) + 2
            matrix.to_excel(writer, sheet_name=sheet_name,
                            startrow=startrow)

            print(f"✓ Results exported to {filepath}")
            print(f"  - Summary sheet with all targets")
            print(f"  - {len(self.results)} individual target sheets")