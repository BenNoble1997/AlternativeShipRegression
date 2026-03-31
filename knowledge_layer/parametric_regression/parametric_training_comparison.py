"""
Training Domain Comparator (REFACTORED)
========================================
Analyzes how regression models change when trained on different datasets.

Compatible with refactored PolynomialDomainAnalyzer and PolynomialModelSelector.

Key Changes from Original:
- Uses selector's combination_manager to generate valid combinations
- Uses RegressionMetrics class for metric calculation
- Correctly references ModelFactory methods
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from itertools import combinations

from knowledge_layer.multi_dataset_analysers.domain_shift.parametric_regression_analyser_class import ParametricDomainAnalyzer
from knowledge_layer.model_selectors.parametric_regresssion_selector_class import ParametricModelSelector
from knowledge_layer.parametric_regression.parametric_regression_model_factory_class import ModelFactory
from knowledge_layer.Metrics.regression_metrics_class import RegressionMetrics


class TrainingDomainComparator:
    """
    Compares regression models trained on different datasets.

    Experimental Design:
    -------------------
    For each predictor combination:
    1. Train model on conventional data → evaluate on all datasets
    2. Train model on matched data → evaluate on all datasets
    3. Train model on alternative data → evaluate on all datasets
    4. Compare: recommendations, coefficients, performance
    """

    def __init__(self,
                 domain_analyzer: ParametricDomainAnalyzer,
                 model_selector: ParametricModelSelector = None):
        """
        Parameters
        ----------
        domain_analyzer : ParametricDomainAnalyzer
            Existing analyzer with prepared datasets
        model_selector : PolynomialModelSelector, optional
            Model selector for getting recommendations and combination generation.
            If None, will use simple combination generation.
        """
        # Store references (composition, not inheritance)
        self.analyzer = domain_analyzer
        self.selector = model_selector

        # Quick access to datasets from analyzer
        self.datasets = {
            'conventional_full': domain_analyzer.conventional_full,
            'conventional_matched': domain_analyzer.conventional_matched,
            'alternative': domain_analyzer.alternative
        }

        # Store configuration
        self.variable_dimensions = domain_analyzer.variable_dimensions
        self.target_dimensions = domain_analyzer.target_dimensions
        self.test_size = domain_analyzer.test_size
        self.random_state = domain_analyzer.random_state

        # Results storage
        self.comparison_results = pd.DataFrame()

        print(f"\nTraining Domain Comparator initialized")
        print(f"  Available training sources: {list(self.datasets.keys())}")
        print(f"  Targets to analyze: {len(self.target_dimensions)}")
        if self.selector:
            print(f"  Using selector's combination manager for valid combinations")
        else:
            print(f"  No selector provided - will use simple combination generation")

    # ========================================================================
    # COMBINATION GENERATION (REFACTORED)
    # ========================================================================

    def _get_valid_combinations(self,
                                target: str,
                                max_predictors: int) -> List[Tuple[str, ...]]:
        """
        Get valid predictor combinations for a target.

        Uses selector's combination_manager if available, otherwise
        generates combinations with simple target exclusion.

        Parameters
        ----------
        target : str
            Target dimension
        max_predictors : int
            Maximum number of predictors

        Returns
        -------
        List[Tuple[str, ...]]
            Valid predictor combinations
        """
        if self.selector:
            # Use selector's combination manager (respects all rules)
            return self.selector.combination_manager.generate_combinations(
                target=target,
                min_predictors=1,
                max_predictors=max_predictors
            )
        else:
            # Fallback: simple combination generation
            available_predictors = [d for d in self.variable_dimensions if d != target]

            all_combos = []
            for n_preds in range(1, min(max_predictors + 1, len(available_predictors) + 1)):
                all_combos.extend(combinations(available_predictors, n_preds))

            return all_combos

    # ========================================================================
    # MAIN ANALYSIS METHOD (REFACTORED)
    # ========================================================================

    def compare_training_domains(self,
                                max_predictors: int = 2,
                                training_sources: List[str] = None,
                                test_all_models: bool = False,
                                verbose: bool = True):
        """
        Train models on different datasets and compare results.

        Parameters
        ----------
        max_predictors : int
            Maximum number of predictors to test
        training_sources : List[str], optional
            Which datasets to train on. Default: all three
        test_all_models : bool
            If True, tests all model types on each training source.
            If False, uses model_selector recommendations (if available).
        verbose : bool
            Print progress

        Returns
        -------
        pd.DataFrame
            Comparison results
        """
        if training_sources is None:
            training_sources = ['conventional_full', 'conventional_matched', 'alternative']

        print(f"\n{'='*140}")
        print(f"TRAINING DOMAIN COMPARISON ANALYSIS")
        print(f"  Training sources: {training_sources}")
        print(f"  Max predictors: {max_predictors}")
        print(f"  Test all models: {test_all_models}")
        print(f"{'='*140}")

        all_results = []

        for target in self.target_dimensions:
            if verbose:
                print(f"\nAnalyzing target: {target}")

            # Get valid combinations using refactored method
            pred_combos = self._get_valid_combinations(target, max_predictors)

            if not pred_combos:
                if verbose:
                    print(f"  No valid combinations for {target}")
                continue

            for pred_combo in pred_combos:
                if verbose:
                    print(f"  Testing: {' + '.join(pred_combo)} → {target}")

                combo_results = self._compare_single_combination(
                    predictors=list(pred_combo),
                    target=target,
                    training_sources=training_sources,
                    test_all_models=test_all_models
                )

                all_results.extend(combo_results)

        # Convert to DataFrame
        if all_results:
            self.comparison_results = pd.DataFrame(all_results)

            if verbose:
                self._print_summary()

            return self.comparison_results
        else:
            print("No valid results generated!")
            return pd.DataFrame()

    # ========================================================================
    # COMBINATION-LEVEL COMPARISON (REFACTORED)
    # ========================================================================

    def _compare_single_combination(self,
                                    predictors: List[str],
                                    target: str,
                                    training_sources: List[str],
                                    test_all_models: bool) -> List[Dict]:
        """
        Compare one predictor combination across all training sources.
        """
        results = []

        # Determine which models to test
        if test_all_models:
            model_types = self._get_all_applicable_models(predictors, training_sources)
        else:
            # Use selector's recommendation if available
            key = (tuple(predictors), target)
            if self.selector and key in self.selector.model_recommendations:
                model_types = [self.selector.model_recommendations[key]]
            else:
                # Fallback to linear
                model_types = ['linear']

        # Train on each source
        for source_name in training_sources:
            for model_type in model_types:
                result = self._train_and_evaluate_single(
                    predictors=predictors,
                    target=target,
                    training_source=source_name,
                    model_type=model_type
                )

                if result is not None:
                    results.append(result)

        return results

    def _train_and_evaluate_single(self,
                                   predictors: List[str],
                                   target: str,
                                   training_source: str,
                                   model_type: str) -> Optional[Dict]:
        """
        Train model on one source, evaluate on all datasets.

        Returns
        -------
        dict with metrics for each evaluation dataset
        """
        # Get training data
        train_data = self.datasets[training_source]
        X_train = train_data[predictors].values
        y_train = train_data[target].values

        # Clean training data
        train_mask = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train) |
                      np.isinf(X_train).any(axis=1) | np.isinf(y_train))
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]

        if len(X_train) < 5:
            return None

        # Fit model using ModelFactory
        try:
            model, preprocessor = ModelFactory.create_and_fit(model_type, X_train, y_train)
        except Exception as e:
            print(f"    Error fitting {model_type} on {training_source}: {e}")
            return None

        # Extract coefficients (FIXED method name)
        coefficients = self._extract_coefficients(model, preprocessor, predictors, model_type)

        # Evaluate on all datasets
        evaluation_metrics = {}

        # Get evaluation datasets using analyzer's data preparation
        eval_datasets = self._prepare_evaluation_datasets(predictors, target)

        for eval_name, (X_eval, y_eval) in eval_datasets.items():
            try:
                y_pred = ModelFactory.predict(model, preprocessor, X_eval, model_type)

                # Calculate metrics using RegressionMetrics class (FIXED)
                actual_n_features = ModelFactory.get_actual_n_features(model_type, X_eval)
                metric_calculator = RegressionMetrics(y_eval, y_pred)
                metrics = metric_calculator.calculate_all_metrics(actual_n_features)

                # Add sample count
                metrics['n_samples'] = len(y_eval)

                evaluation_metrics[eval_name] = metrics

            except Exception as e:
                print(f"    Error evaluating on {eval_name}: {e}")
                continue

        # Build result dictionary
        result = {
            'target': target,
            'predictors': tuple(predictors),
            'predictor_str': ' + '.join(predictors),
            'n_predictors': len(predictors),
            'training_source': training_source,
            'model_type': model_type,
            'coefficients': coefficients,
            'n_train_samples': len(X_train)
        }

        # Add metrics for each evaluation dataset
        for eval_name, metrics in evaluation_metrics.items():
            for metric_name, value in metrics.items():
                result[f'{eval_name}_{metric_name}'] = value

        return result

    # ========================================================================
    # HELPER METHODS (REFACTORED)
    # ========================================================================

    def _prepare_evaluation_datasets(self,
                                     predictors: List[str],
                                     target: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Prepare evaluation datasets using analyzer's logic.

        Returns
        -------
        dict : {dataset_name: (X, y)}
        """
        # Reuse analyzer's data preparation method
        data_prep = self.analyzer._prepare_fourway_data(predictors, target)

        if data_prep is None:
            return {}

        return {
            'conv_train': (data_prep['X_train'], data_prep['y_train']),
            'conv_test': (data_prep['X_test'], data_prep['y_test']),
            'conv_matched': (data_prep['X_matched'], data_prep['y_matched']),
            'alternative': (data_prep['X_alternative'], data_prep['y_alternative'])
        }

    def _extract_coefficients(self, model, preprocessor, predictors, model_type):
        """Extract model coefficients in interpretable format."""
        try:
            # FIXED: Use correct method name from ModelFactory
            coeffs = ModelFactory.get_model_coefficients(
                model, preprocessor, predictors, model_type
            )
            return coeffs
        except Exception as e:
            return None

    def _get_all_applicable_models(self,
                                   predictors: List[str],
                                   training_sources: List[str]) -> List[str]:
        """
        Get all model types applicable across all training sources.

        Only returns models that are applicable to ALL training sources
        to ensure fair comparison.
        """
        # Get sample data from each training source to check applicability
        applicable_per_source = []

        for source_name in training_sources:
            source_data = self.datasets[source_name]
            X_sample = source_data[predictors].dropna().values[:100]  # Sample for speed
            y_sample = source_data[self.target_dimensions[0]].dropna().values[:100]

            if len(X_sample) < 5:
                continue

            # Get applicable models for this source
            applicable = ModelFactory.get_applicable_models(
                X_sample, y_sample,
                exclude_models=None,
                clean_first=True
            )
            applicable_per_source.append(set(applicable))

        if not applicable_per_source:
            return ['linear']  # Fallback

        # Return intersection (models applicable to ALL sources)
        common_models = set.intersection(*applicable_per_source)

        return list(common_models) if common_models else ['linear']

    # ========================================================================
    # SUMMARY AND DISPLAY
    # ========================================================================

    def _print_summary(self):
        """Print summary of comparison results."""
        if self.comparison_results.empty:
            return

        print(f"\n{'='*140}")
        print("SUMMARY: Training Domain Comparison")
        print(f"{'='*140}")

        # Count results by training source
        by_source = self.comparison_results.groupby('training_source').size()
        print(f"\nResults by training source:")
        for source, count in by_source.items():
            print(f"  {source}: {count} combinations")

        # Identify recommendation changes
        print(f"\nModel Type Distribution:")
        by_model = self.comparison_results.groupby(['training_source', 'model_type']).size()
        for (source, model), count in by_model.items():
            print(f"  {source} - {model}: {count}")

    # ========================================================================
    # VISUALIZATION METHODS (Keep existing implementations)
    # ========================================================================

    def plot_performance_heatmap(self,
                                target: str = None,
                                predictors: List[str] = None,
                                metric: str = 'r2',
                                use_best: bool = True,
                                save_path: str = None):
        """
        Create heatmap showing performance of each training source
        on each evaluation dataset.
        """
        if self.comparison_results.empty:
            print("No results to plot")
            return

        # Filter data
        filtered_df = self.comparison_results.copy()

        if target is None and predictors is None:
            print("Error: Must specify either target or predictors")
            return

        if target is not None:
            filtered_df = filtered_df[filtered_df['target'] == target]
            if filtered_df.empty:
                print(f"No results for target '{target}'")
                return

        if predictors is not None:
            filtered_df = filtered_df[filtered_df['predictors'] == tuple(predictors)]
            if filtered_df.empty:
                print(f"No results for predictors {predictors}")
                return

        # Setup
        eval_datasets = ['conv_train', 'conv_test', 'conv_matched', 'alternative']
        training_sources = filtered_df['training_source'].unique()

        if len(training_sources) == 0:
            print("No training sources in filtered data")
            return

        heatmap_data = np.zeros((len(training_sources), len(eval_datasets)))
        best_combinations = {}

        # For each training source, find best model
        for i, source in enumerate(training_sources):
            source_df = filtered_df[filtered_df['training_source'] == source]

            if use_best and predictors is None:
                primary_eval_col = f'conv_test_{metric}'

                if primary_eval_col in source_df.columns:
                    if metric == 'r2':
                        best_idx = source_df[primary_eval_col].idxmax()
                    else:
                        best_idx = source_df[primary_eval_col].idxmin()

                    if pd.notna(best_idx):
                        best_row = source_df.loc[best_idx]
                        best_combo_name = best_row['predictor_str']
                        best_model_type = best_row['model_type']
                        best_combinations[source] = f"{best_combo_name} ({best_model_type})"

                        for j, eval_ds in enumerate(eval_datasets):
                            col_name = f'{eval_ds}_{metric}'
                            if col_name in best_row:
                                heatmap_data[i, j] = best_row[col_name]
                    else:
                        best_combinations[source] = "N/A"
                else:
                    best_combinations[source] = "N/A"
            else:
                for j, eval_ds in enumerate(eval_datasets):
                    col_name = f'{eval_ds}_{metric}'
                    if col_name in source_df.columns:
                        values = source_df[col_name].dropna()
                        if len(values) > 0:
                            heatmap_data[i, j] = values.mean()

                if predictors is not None:
                    pred_str = ' + '.join(predictors)
                    model_type = source_df.iloc[0]['model_type'] if len(source_df) > 0 else 'N/A'
                    best_combinations[source] = f"{pred_str} ({model_type})"
                else:
                    best_combinations[source] = "Average"

        # Create Y-axis labels
        y_labels = []
        for source in training_sources:
            combo = best_combinations.get(source, "N/A")
            if len(combo) > 40:
                combo = combo[:37] + "..."
            y_labels.append(f"{source}\n[{combo}]")

        # Create title
        if target and predictors:
            title = f"{metric.upper()} Performance: {' + '.join(predictors)} → {target}"
        elif target:
            title = f"{metric.upper()} Performance for {target}"
        else:
            title = f"{metric.upper()} Performance Comparison"

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(12, 8))

        if metric == 'r2':
            vmin, vmax = 0, 1
            cmap = 'RdYlGn'
        elif metric in ['rrmse', 'mape']:
            vmin = 0
            vmax = heatmap_data.max() * 1.1 if heatmap_data.max() > 0 else 100
            cmap = 'RdYlGn_r'
        else:
            vmin, vmax = heatmap_data.min(), heatmap_data.max()
            cmap = 'RdYlGn'

        im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)

        ax.set_xticks(range(len(eval_datasets)))
        ax.set_yticks(range(len(training_sources)))
        ax.set_xticklabels(eval_datasets, rotation=45, ha='right', fontsize=11)
        ax.set_yticklabels(y_labels, fontsize=9)

        # Add text annotations
        for i in range(len(training_sources)):
            for j in range(len(eval_datasets)):
                value = heatmap_data[i, j]
                text_color = 'white' if (metric in ['rrmse', 'mape'] and value > vmax * 0.5) or (metric == 'r2' and value < 0.5) else 'black'
                ax.text(j, i, f'{value:.3f}', ha="center", va="center",
                       color=text_color, fontsize=11, fontweight='bold')

        ax.set_title(title, fontweight='bold', fontsize=13, pad=20)
        ax.set_xlabel('Evaluation Dataset', fontweight='bold', fontsize=12)
        ax.set_ylabel('Training Source [Best Combination]', fontweight='bold', fontsize=12)

        cbar = plt.colorbar(im, ax=ax, label=metric.upper())
        cbar.ax.tick_params(labelsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Heatmap saved to {save_path}")

        plt.show()

    def plot_coefficient_comparison(self,
                                    target: str,
                                    predictors: List[str],
                                    model_type: str = None,
                                    save_path: str = None):
        """
        Plot coefficient comparison across training domains.

        Shows how the relationship (coefficients) changes when trained
        on different datasets.
        """
        # Filter results
        df = self.comparison_results[
            (self.comparison_results['target'] == target) &
            (self.comparison_results['predictors'] == tuple(predictors))
            ]

        if model_type:
            df = df[df['model_type'] == model_type]

        if df.empty:
            print(f"No results for {predictors} → {target}")
            return

        # Extract coefficients
        fig, ax = plt.subplots(figsize=(10, 6))

        sources = df['training_source'].values
        coeffs_list = df['coefficients'].values

        # Plot coefficient comparison (implementation depends on coefficient structure)
        # This is a placeholder - actual implementation depends on how coefficients are stored

        ax.set_title(f'Coefficient Comparison: {" + ".join(predictors)} → {target}')
        ax.set_xlabel('Training Source')
        ax.set_ylabel('Coefficient Value')
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_prediction_lines(self,
                              target: str,
                              predictor: str,
                              model_type: str = None,
                              n_points: int = 200,
                              save_path: str = None):
        """
        Plot regression lines from different training sources on same plot.

        Only works for single-predictor models.
        Shows how the fitted relationship changes across training domains.
        """
        # Filter to single predictor
        df = self.comparison_results[
            (self.comparison_results['target'] == target) &
            (self.comparison_results['predictor_str'] == predictor)
            ]

        if model_type:
            df = df[df['model_type'] == model_type]

        if df.empty:
            print(f"No results for {predictor} → {target}")
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        # Get data range
        all_X = []
        for source_name in df['training_source'].unique():
            source_data = self.datasets[source_name]
            X = source_data[predictor].dropna().values
            all_X.append(X)

        X_min = np.min([X.min() for X in all_X])
        X_max = np.max([X.max() for X in all_X])
        X_line = np.linspace(X_min, X_max, n_points).reshape(-1, 1)

        # Plot each training source's regression line
        colors = {'conventional_full': '#2ecc71',
                  'conventional_matched': '#e67e22',
                  'alternative': '#e74c3c'}

        for idx, row in df.iterrows():
            source = row['training_source']
            model_type_used = row['model_type']

            # Get trained model (would need to retrain or store models)
            # This is a simplified version

            source_data = self.datasets[source]
            X_train = source_data[[predictor]].dropna().values
            y_train = source_data[target].dropna().values

            # Align X and y
            mask = np.isin(source_data.index,
                           source_data[[predictor, target]].dropna().index)
            X_train = X_train[:len(y_train)]

            try:
                model, preprocessor = ModelFactory.create_and_fit(
                    model_type_used, X_train, y_train
                )
                y_line = ModelFactory.predict(model, preprocessor, X_line, model_type_used)

                # Plot line
                label = f"{source} (R²={row['conv_test_r2']:.3f})"
                ax.plot(X_line, y_line, '-', linewidth=2.5,
                        color=colors.get(source, 'gray'),
                        label=label, alpha=0.8)

            except Exception as e:
                print(f"Error plotting {source}: {e}")
                continue

        # Add scatter points from all datasets
        for source_name, color in colors.items():
            data = self.datasets[source_name]
            X_scatter = data[predictor].values
            y_scatter = data[target].values
            mask = ~(np.isnan(X_scatter) | np.isnan(y_scatter))

            ax.scatter(X_scatter[mask], y_scatter[mask],
                       alpha=0.3, s=20, color=color,
                       label=f'{source_name} data')

        ax.set_xlabel(predictor, fontweight='bold', fontsize=12)
        ax.set_ylabel(target, fontweight='bold', fontsize=12)
        ax.set_title(f'Regression Line Comparison: {predictor} → {target}\n'
                     f'Trained on Different Datasets',
                     fontweight='bold', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    # ========================================================================
    # EXPORT
    # ========================================================================

    def export_results(self, filename: str = 'training_domain_comparison.csv'):
        """Export comparison results to CSV."""
        if self.comparison_results.empty:
            print("No results to export")
            return

        export_df = self.comparison_results.copy()

        if 'coefficients' in export_df.columns:
            export_df['coefficients'] = export_df['coefficients'].astype(str)

        export_df.to_csv(filename, index=False)
        print(f"✓ Results exported to {filename}")