import pandas as pd
import numpy as np


class FormulaPrinter:


    @staticmethod
    def print_formulas_for_target(model_recommendations:dict,
                                  model_descriptors:dict,
                                  detailed_results:dict,
                                  target: str,
                                  max_predictors: int = 2,
                                  notation: str = 'math',
                                  sort_by: str = 'adj_r2',
                                  top_n: int|None = None) -> None:
        """
        Print formulas for all models for a specific target.

        **Displays readable mathematical formulas with actual coefficients.**

        Parameters
        ----------
        target : str
            Target variable to show formulas for
        max_predictors : int, default=2
            Maximum number of predictors to include
        notation : str, default='math'
            Formula notation style:
            - 'math': LOA = 12.400 + 0.523·√DWT - 0.321·(1/GT)
            - 'python': LOA = 12.400 + 0.523 * sqrt(DWT) - 0.321 * (1/GT)
            - 'latex': LOA = 12.400 + 0.523 \\sqrt{DWT} - 0.321 \\frac{1}{GT}
        sort_by : str, default='adj_r2'
            Metric to sort by ('adj_r2', 'rrmse', 'aic', 'bic')
        top_n : int, optional
            Only show top N models. If None, shows all.

        Examples
        --------
        >>> # Show all formulas for LOA
        >>> selector.print_formulas_for_target('LOA (M)')

        >>> # Show top 3 by RRMSE in Python notation
        >>> selector.print_formulas_for_target('LOA (M)',
        ...                                    sort_by='rrmse',
        ...                                    top_n=3,
        ...                                    notation='python')
        """
        if not model_descriptors:
            print("⚠️  No model descriptors available. Formulas require CV or model fitting.")
            print("   Run analyze_all_combinations() first.")
            return

        # Collect all models for this target
        models_for_target = []

        for (predictors, tgt), model_type in model_recommendations.items():
            if tgt != target:
                continue

            if len(predictors) > max_predictors:
                continue

            key = (predictors, tgt)

            # Check if we have descriptor
            if key not in model_descriptors:
                continue

            if model_type not in model_descriptors[key]:
                continue

            descriptor = model_descriptors[key][model_type]

            # Get metrics for sorting
            if key in detailed_results and model_type in detailed_results[key]:
                metrics = detailed_results[key][model_type]
            else:
                continue

            models_for_target.append({
                'predictors': predictors,
                'model_type': model_type,
                'descriptor': descriptor,
                'metrics': metrics
            })

        if not models_for_target:
            print(f"\n⚠️  No formulas available for target '{target}'")
            return

        # Sort models
        reverse = sort_by in ['adj_r2', 'r2']  # Higher is better for R²
        models_for_target.sort(
            key=lambda x: x['metrics'].get(sort_by, 0 if reverse else float('inf')),
            reverse=reverse
        )

        # Limit to top_n
        if top_n is not None:
            models_for_target = models_for_target[:top_n]

        # Print header
        print("\n" + "=" * 100)
        print(f"FORMULAS FOR TARGET: {target}")
        print("=" * 100)
        print(f"Notation: {notation.upper()}")
        print(f"Sorted by: {sort_by.upper()} ({'higher is better' if reverse else 'lower is better'})")
        if top_n:
            print(f"Showing: Top {top_n} models")
        print("=" * 100)

        # Print each model
        for idx, model_info in enumerate(models_for_target, 1):
            predictors = model_info['predictors']
            model_type = model_info['model_type']
            descriptor = model_info['descriptor']
            metrics = model_info['metrics']

            # Get formula
            try:
                formula = descriptor.get_formula_string(
                    include_coefficients=True,
                    notation=notation,
                    only_active=True
                )
            except Exception as e:
                formula = f"[Error generating formula: {str(e)[:50]}]"

            # Print model info
            print(f"\n{idx}. {' + '.join(predictors)} → {model_type}")
            print("-" * 100)
            print(f"Formula: {formula}")
            print(f"Metrics: Adj R² = {metrics.get('adj_r2', 0):.4f}, "
                  f"RRMSE = {metrics.get('rrmse', 0):.2f}%, "
                  f"AIC = {metrics.get('aic', 0):.1f}")

            # Show parameter count for verification
            n_params = descriptor.get_effective_parameters()
            print(f"Parameters: {n_params}")

            # Extra info for special models
            if model_type == 'mfp':
                if descriptor.transformations:
                    n_active = sum(1 for t in descriptor.transformations.values() if t[0] != 'excluded')
                    n_total = len(descriptor.transformations)
                    print(f"MFP: {n_active}/{n_total} variables active")

            elif 'lasso' in model_type:
                if descriptor.active_features:
                    n_active = len(descriptor.active_features)
                    n_total = len(descriptor.coefficients) if descriptor.coefficients else 0
                    print(f"Lasso: {n_active}/{n_total} features selected (α={metrics.get('selected_alpha', 0):.4f})")

    @staticmethod
    def print_all_formulas(target_dimensions:list[str],
                           model_reccomendations,
                           model_descriptors,
                           detailed_results,
                            max_predictors: int = 2,
                           notation: str = 'math',
                           sort_by: str = 'adj_r2',
                           top_n_per_target: int|None = None,
                           save_to_file: str|None = None) -> None:
        """
        Print formulas for all targets.

        Parameters
        ----------
        max_predictors : int, default=2
            Maximum number of predictors to include
        notation : str, default='math'
            Formula notation style
        sort_by : str, default='adj_r2'
            Metric to sort by
        top_n_per_target : int, optional
            Only show top N models per target
        save_to_file : str, optional
            If provided, save output to this file instead of printing


        """
        if save_to_file:
            import sys
            from io import StringIO

            # Redirect stdout to string
            old_stdout = sys.stdout
            sys.stdout = StringIO()

        try:
            for target in target_dimensions:
                FormulaPrinter.print_formulas_for_target(
                    model_recommendations=model_reccomendations,
                    model_descriptors=model_descriptors,
                    detailed_results=detailed_results,
                    target=target,
                    max_predictors=max_predictors,
                    notation=notation,
                    sort_by=sort_by,
                    top_n=top_n_per_target
                )

            if save_to_file:
                # Get captured output
                output = sys.stdout.getvalue()
                sys.stdout = old_stdout

                # Write to file
                with open(save_to_file, 'w', encoding='utf-8') as f:
                    f.write(output)

                print(f"✅ Formulas saved to {save_to_file}")

        finally:
            if save_to_file:
                sys.stdout = old_stdout

    @staticmethod
    def print_best_formulas(target_dimensions:list[str],
                            model_recommendations:dict,
                            detailed_results:dict,
                            model_descriptors: dict[str,],
                            selection_metric:str='r2_adj',
                            notation: str = 'math',
                            include_metrics: bool = True) -> None:
        """
        Print formulas for the best model for each target.

        Shows only the recommended model for each target variable.

        Parameters
        ----------
        notation : str, default='math'
            Formula notation style
        include_metrics : bool, default=True
            Whether to show performance metrics

        Examples
        --------
        >>> # Show best formulas
        >>> selector.print_best_formulas()

        >>> # LaTeX notation without metrics
        >>> selector.print_best_formulas(
        ...     notation='latex',
        ...     include_metrics=False
        ... )
        """
        if not model_descriptors:
            print("⚠️  No model descriptors available.")
            return

        print("\n" + "=" * 100)
        print("BEST FORMULAS (RECOMMENDED MODELS)")
        print("=" * 100)
        print(f"Notation: {notation.upper()}")
        print(f"Selection metric: {selection_metric.upper()}")
        print("=" * 100)

        for target in target_dimensions:
            # Find best model for this target
            best_found = False

            for (predictors, tgt), model_type in model_recommendations.items():
                if tgt != target:
                    continue

                key = (predictors, tgt)

                # Check if we have descriptor
                if key not in model_descriptors:
                    continue

                if model_type not in model_descriptors[key]:
                    continue

                descriptor = model_descriptors[key][model_type]

                # Get metrics
                if key in detailed_results and model_type in detailed_results[key]:
                    metrics = detailed_results[key][model_type]
                else:
                    continue

                # This is the best model for this target
                print(f"\n{target}")
                print("-" * 100)
                print(f"Model: {model_type} ({' + '.join(predictors)})")

                try:
                    formula = descriptor.get_formula_string(
                        include_coefficients=True,
                        notation=notation,
                        only_active=True
                    )
                    print(f"Formula: {formula}")
                except Exception as e:
                    print(f"Formula: [Error: {str(e)[:50]}]")

                if include_metrics:
                    print(f"Performance: Adj R² = {metrics.get('adj_r2', 0):.4f}, "
                          f"RRMSE = {metrics.get('rrmse', 0):.2f}%, "
                          f"AIC = {metrics.get('aic', 0):.1f}")

                best_found = True
                break  # Only one best per target

            if not best_found:
                print(f"\n{target}")
                print("-" * 100)
                print("⚠️  No formula available")

        print("\n" + "=" * 100)

    @staticmethod
    def export_formulas_to_csv(model_descriptors:dict,
                               model_recommendations:dict,
                               detailed_results:dict,
                               filename: str = 'model_formulas.csv',
                               max_predictors: int = 2,
                               notation: str = 'math',
                               verbose=False) -> pd.DataFrame:
        """
        Export formulas to CSV file.

        Creates a spreadsheet with all formulas and their metrics.

        Parameters
        ----------
        filename : str, default='model_formulas.csv'
            Output filename
        max_predictors : int, default=2
            Maximum number of predictors to include
        notation : str, default='math'
            Formula notation style

        Returns
        -------
        pd.DataFrame
            DataFrame with all formulas and metrics

        Examples
        --------
        >>> df = selector.export_formulas_to_csv('formulas.csv')
        >>> print(df.head())
        """
        if not model_descriptors:
            print("⚠️  No model descriptors available.")
            return pd.DataFrame()

        export_data = []

        for (predictors, target), model_type in model_recommendations.items():
            if len(predictors) > max_predictors:
                continue

            key = (predictors, target)

            # Check if we have descriptor
            if key not in model_descriptors:
                continue

            if model_type not in model_descriptors[key]:
                continue

            descriptor = model_descriptors[key][model_type]

            # Get metrics
            if key in detailed_results and model_type in detailed_results[key]:
                metrics = detailed_results[key][model_type]
            else:
                continue

            # Get formula
            try:
                formula = descriptor.get_formula_string(
                    include_coefficients=True,
                    notation=notation,
                    only_active=True
                )
            except Exception as e:
                formula = f"[Error: {str(e)[:50]}]"

            # Build row
            row = {
                'target': target,
                'predictors': ' + '.join(predictors),
                'n_predictors': len(predictors),
                'model_type': model_type,
                'formula': formula,
                'n_parameters': descriptor.get_effective_parameters(),
                'adj_r2': metrics.get('adj_r2', np.nan),
                'rrmse': metrics.get('rrmse', np.nan),
                'rmse': metrics.get('rmse', np.nan),
                'aic': metrics.get('aic', np.nan),
                'bic': metrics.get('bic', np.nan),
                'heteroscedasticity': metrics.get('heteroscedasticity', 'unknown'),
                'n_samples': metrics.get('n_samples', np.nan)
            }

            # Add coefficients as separate columns
            if descriptor.coefficients:
                for coef_name, coef_value in descriptor.coefficients.items():
                    row[f'coef_{coef_name}'] = coef_value

            if descriptor.intercept is not None:
                row['intercept'] = descriptor.intercept

            export_data.append(row)

        # Create DataFrame
        df = pd.DataFrame(export_data)

        # Save to CSV
        df.to_csv(filename, index=False)

        if verbose:
            print(f"✅ Formulas exported to {filename}")
            print(f"   {len(df)} formulas exported")

        return df

    @staticmethod
    def print_formula_comparison(target: str,
                                 detailed_results: dict,
                                 model_descriptors: dict[str,],
                                 predictor_combo: tuple[str, ...],
                                 notation: str = 'math') -> None:
        """
        Compare formulas for different model types for a specific combination.

        Shows how different models (linear, polynomial, MFP, etc.) represent
        the same predictor-target relationship.

        Parameters
        ----------
        target : str
            Target variable
        predictor_combo : tuple of str
            Predictor combination, e.g., ('DWT', 'GT')
        notation : str, default='math'
            Formula notation style

        Examples
        --------
        >>> # Compare all models for DWT → LOA
        >>> selector.print_formula_comparison('LOA (M)', ('DWT',))

        >>> # Compare models for DWT + GT → LOA
        >>> selector.print_formula_comparison('LOA (M)', ('DWT', 'GT'))
        """
        key = (tuple(predictor_combo), target)

        if key not in detailed_results:
            print(f"⚠️  No results found for {' + '.join(predictor_combo)} → {target}")
            return

        if key not in model_descriptors:
            print(f"⚠️  No formulas available for {' + '.join(predictor_combo)} → {target}")
            return

        print("\n" + "=" * 100)
        print(f"FORMULA COMPARISON: {' + '.join(predictor_combo)} → {target}")
        print("=" * 100)

        # Get all models for this combination
        model_results = detailed_results[key]
        model_descriptors = model_descriptors[key]

        # Sort by performance
        sorted_models = sorted(
            model_results.items(),
            key=lambda x: x[1].get('adj_r2', 0),
            reverse=True
        )

        for idx, (model_type, metrics) in enumerate(sorted_models, 1):
            if model_type not in model_descriptors:
                continue

            descriptor = model_descriptors[model_type]

            print(f"\n{idx}. {model_type}")
            print("-" * 100)

            try:
                formula = descriptor.get_formula_string(
                    include_coefficients=True,
                    notation=notation,
                    only_active=True
                )
                print(f"Formula: {formula}")
            except Exception as e:
                print(f"Formula: [Error: {str(e)[:50]}]")

            print(f"Adj R² = {metrics.get('adj_r2', 0):.4f}, "
                  f"RRMSE = {metrics.get('rrmse', 0):.2f}%, "
                  f"Parameters = {descriptor.get_effective_parameters()}")

        print("\n" + "=" * 100)