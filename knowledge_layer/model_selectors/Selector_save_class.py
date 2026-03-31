"""
Results Exporter with ModelDescriptor Support
==============================================

Enhanced export/import system supporting fitted model storage.

Key Updates:
- Handles ModelDescriptor objects for ParametricModelSelector
- Preserves fitted models and preprocessors in pickle format
- Gracefully handles non-pickle formats (stores metrics only)
- Maintains backward compatibility with RandomForestSelector

Export formats:
- Pickle: Full storage including fitted models (RECOMMENDED)
- Excel/CSV/JSON: Metrics and metadata only (models not included)
"""

import pickle
import gzip
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
import warnings


class ResultsExporter:
    """
    Unified exporter for regression selector results with model descriptor support.

    **Concept: Pickle vs Non-Pickle Formats**

    Pickle Format:
    - Stores EVERYTHING: fitted models, preprocessors, all Python objects
    - Best for: Reloading selectors for predictions or continued analysis
    - File size: Larger (contains actual model objects)

    Non-Pickle Formats (Excel/CSV/JSON):
    - Stores METRICS ONLY: performance stats, formulas as text
    - Best for: Sharing results, creating reports, visualization
    - File size: Smaller (just numbers and text)
    - Limitation: Cannot recreate fitted models for predictions

    Examples
    --------
    >>> # Full save (with fitted models)
    >>> exporter = ResultsExporter(selector)
    >>> exporter.save('models.pkl')  # Can reload and make predictions
    >>>
    >>> # Metrics-only save (for sharing)
    >>> exporter.save('results.xlsx')  # Human-readable, but no models
    """

    SUPPORTED_SELECTORS = [
        'ParametricModelSelector',
        'RandomForestSelector'
    ]

    def __init__(self, selector, skip_validation: bool = False):
        """
        Initialize exporter with a selector instance.

        Parameters
        ----------
        selector : ParametricModelSelector or RandomForestSelector
            Fitted selector with results to export
        skip_validation : bool
            If True, skip checking for results (used when loading)
        """
        self.selector = selector
        self.selector_type = type(selector).__name__

        # Validate selector type
        if self.selector_type not in self.SUPPORTED_SELECTORS:
            warnings.warn(
                f"Selector type '{self.selector_type}' not officially supported. "
                f"Export may not work correctly. Supported types: {self.SUPPORTED_SELECTORS}"
            )

        if not skip_validation:
            self._validate_has_results()

        self.verbose = getattr(selector, 'verbose', True)

    def _validate_has_results(self):
        """Check that selector has results to export."""
        if self.selector_type == 'ParametricModelSelector':
            if not hasattr(self.selector, 'model_recommendations') or \
                    not self.selector.model_recommendations:
                raise ValueError(
                    "No results to export. Run analyze_all_combinations() first."
                )
        elif self.selector_type == 'RandomForestSelector':
            if not hasattr(self.selector, 'results') or \
                    not self.selector.results:
                raise ValueError(
                    "No results to export. Run analyze_all_combinations() first."
                )

    # ========================================================================
    # MAIN EXPORT/IMPORT METHODS
    # ========================================================================

    def save(self,
             filepath: str,
             format: str = 'auto',
             include_metadata: bool = True,
             compress: bool = False) -> None:
        """
        Save selector results.

        **Important for ParametricModelSelector:**
        - Pickle format: Saves fitted models (can reload for predictions)
        - Other formats: Save metrics only (models not included)

        Parameters
        ----------
        filepath : str
            Path to save results (extension added automatically if missing)
        format : str
            Format: 'pickle', 'csv', 'excel', 'json', or 'auto' (detect from extension)
        include_metadata : bool
            Whether to save configuration metadata
        compress : bool
            Whether to compress pickle files (saves space but slower)

        Examples
        --------
        >>> # RECOMMENDED: Pickle for full functionality
        >>> exporter.save('models.pkl')  # Can reload and predict
        >>>
        >>> # Alternative: Excel for sharing
        >>> exporter.save('results.xlsx')  # Metrics only
        """
        filepath = Path(filepath)

        # Auto-detect format from extension
        if format == 'auto':
            ext = filepath.suffix.lower()
            format_map = {
                '.pkl': 'pickle',
                '.pickle': 'pickle',
                '.csv': 'csv',
                '.xlsx': 'excel',
                '.xls': 'excel',
                '.json': 'json'
            }
            format = format_map.get(ext, 'pickle')

            # Add extension if missing
            if not filepath.suffix:
                filepath = filepath.with_suffix('.pkl')

        # Warn if using non-pickle format with ParametricModelSelector
        if self.selector_type == 'ParametricModelSelector' and format != 'pickle':
            if self.verbose:
                print(f"\n⚠️  Warning: Saving as {format.upper()} will NOT include fitted models.")
                print(f"   For full functionality (predictions, model objects), use pickle format.")
                print(f"   This export will contain metrics and formulas only.\n")

        # Route to appropriate save method
        if format == 'pickle':
            self._save_as_pickle(filepath, compress, include_metadata)
        elif format == 'csv':
            self._save_as_csv(filepath, include_metadata)
        elif format == 'excel':
            self._save_as_excel(filepath, include_metadata)
        elif format == 'json':
            self._save_as_json(filepath, include_metadata)
        else:
            raise ValueError(
                f"Unknown format: {format}. "
                f"Use 'pickle', 'csv', 'excel', or 'json'"
            )

        if self.verbose:
            print(f"✅ Results saved to {filepath}")

    def load(self,
             filepath: str,
             format: str = 'auto') -> None:
        """
        Load previously saved results into the selector.

        **Important:**
        - Only pickle format contains fitted models
        - Other formats restore metrics only (cannot make predictions)

        Parameters
        ----------
        filepath : str
            Path to saved results
        format : str
            Format: 'pickle', 'csv', 'excel', 'json', or 'auto' (detect from extension)

        Examples
        --------
        >>> exporter = ResultsExporter(selector)
        >>> exporter.load('models.pkl')  # Full restore
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Auto-detect format from extension
        if format == 'auto':
            ext = filepath.suffix.lower()
            format_map = {
                '.pkl': 'pickle',
                '.pickle': 'pickle',
                '.csv': 'csv',
                '.xlsx': 'excel',
                '.xls': 'excel',
                '.json': 'json',
                '.gz': 'pickle'
            }
            format = format_map.get(ext, 'pickle')

        # Warn if loading non-pickle format
        if self.selector_type == 'ParametricModelSelector' and format != 'pickle':
            if self.verbose:
                print(f"\n⚠️  Warning: Loading from {format.upper()} format.")
                print(f"   Fitted models and model descriptors are NOT available.")
                print(f"   For full functionality, load from pickle format.\n")

        # Route to appropriate load method
        if format == 'pickle':
            self._load_from_pickle(filepath)
        elif format == 'csv':
            self._load_from_csv(filepath)
        elif format == 'excel':
            self._load_from_excel(filepath)
        elif format == 'json':
            self._load_from_json(filepath)
        else:
            raise ValueError(f"Unknown format: {format}")

        if self.verbose:
            n_results = self._count_results()
            print(f"✅ Loaded {n_results} results from {filepath}")

    def quick_save(self,
                   name: str = 'results',
                   output_dir: str = 'results') -> None:
        """
        Quick save with both pickle and Excel.

        Creates:
        - {output_dir}/{name}_{timestamp}.pkl (for reloading with models)
        - {output_dir}/{name}_{timestamp}.xlsx (for inspection, metrics only)

        Parameters
        ----------
        name : str
            Base name for files
        output_dir : str
            Directory to save in (created if doesn't exist)

        Examples
        --------
        >>> exporter.quick_save('ship_models_v1')
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_dir = Path(output_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        # Save pickle (compressed) - includes models
        pkl_path = base_dir / f"{name}_{timestamp}.pkl"
        self.save(pkl_path, format='pickle', compress=True)

        # Save Excel - metrics only
        xlsx_path = base_dir / f"{name}_{timestamp}.xlsx"
        self.save(xlsx_path, format='excel')

        if self.verbose:
            print(f"\n✅ Quick save complete:")
            print(f"   Pickle: {pkl_path} (with fitted models)")
            print(f"   Excel:  {xlsx_path} (metrics only)")

    # ========================================================================
    # PICKLE METHODS (Full storage including models)
    # ========================================================================

    def _save_as_pickle(self,
                        filepath: Path,
                        compress: bool,
                        include_metadata: bool) -> None:
        """
        Save results as pickle file with full model storage.

        **This is the ONLY format that preserves fitted models.**
        """
        if filepath.suffix not in ['.pkl', '.pickle', '.gz']:
            filepath = filepath.with_suffix('.pkl')

        # Prepare data bundle
        data = {
            'selector_type': self.selector_type,
            'results': self._extract_results(),
            'version': '2.0',  # Incremented for ModelDescriptor support
            'saved_at': datetime.now().isoformat()
        }

        if include_metadata:
            data['metadata'] = self._extract_metadata()

        # ✅ NEW: Include model descriptors for ParametricModelSelector
        if self.selector_type == 'ParametricModelSelector':
            if hasattr(self.selector, 'model_descriptors'):
                data['model_descriptors'] = self.selector.model_descriptors
                if self.verbose:
                    n_descriptors = sum(len(models) for models in self.selector.model_descriptors.values())
                    print(f"   📦 Saving {n_descriptors} model descriptors with fitted models")

        # Save with optional compression
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if compress:
            save_path = str(filepath) + '.gz' if not str(filepath).endswith('.gz') else filepath
            with gzip.open(save_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_from_pickle(self, filepath: Path) -> None:
        """Load results from pickle file including fitted models."""
        # Handle compressed files
        if filepath.suffix == '.gz' or str(filepath).endswith('.pkl.gz'):
            with gzip.open(filepath, 'rb') as f:
                data = pickle.load(f)
        else:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

        # Check version
        version = data.get('version', '1.0')
        if self.verbose and version == '1.0':
            print("   ℹ️  Loading version 1.0 format (pre-ModelDescriptor)")

        # Validate selector type match
        saved_type = data.get('selector_type')
        if saved_type != self.selector_type:
            warnings.warn(
                f"Loaded results are from {saved_type}, "
                f"but current selector is {self.selector_type}. "
                f"This may cause issues."
            )

        # Load results into selector
        self._inject_results(data['results'])

        # ✅ NEW: Load model descriptors if present
        if self.selector_type == 'ParametricModelSelector':
            if 'model_descriptors' in data:
                self.selector.model_descriptors = data['model_descriptors']
                if self.verbose:
                    n_descriptors = sum(len(models) for models in self.selector.model_descriptors.values())
                    print(f"   ✅ Loaded {n_descriptors} model descriptors with fitted models")
            elif version == '1.0':
                if self.verbose:
                    print("   ⚠️  Old format: model_descriptors not available")
                    print("      Metrics are loaded, but you cannot:")
                    print("      - Print formulas")
                    print("      - Make predictions with fitted models")
                    print("      - Export model parameters")

        # Optional: Check metadata compatibility
        if 'metadata' in data and self.verbose:
            self._check_metadata_compatibility(data['metadata'])

    # ========================================================================
    # CSV METHODS (Metrics only - no models)
    # ========================================================================

    def _save_as_csv(self, filepath: Path, include_metadata: bool) -> None:
        """
        Save results as CSV files (metrics only, no fitted models).

        Creates:
        - {name}_results.csv: Main results
        - {name}_formulas.txt: Model formulas (if available)
        - {name}_metadata.json: Configuration info (if include_metadata=True)
        """
        if filepath.suffix == '.csv':
            base_path = filepath.with_suffix('')
        else:
            base_path = filepath

        base_path.parent.mkdir(parents=True, exist_ok=True)

        # Save main results
        results_df = self._results_to_dataframe()
        results_path = Path(str(base_path) + '_results.csv')
        results_df.to_csv(results_path, index=False)

        # ✅ NEW: Save formulas as text if available
        if self.selector_type == 'ParametricModelSelector':
            if hasattr(self.selector, 'model_descriptors') and self.selector.model_descriptors:
                formulas_path = Path(str(base_path) + '_formulas.txt')
                self._save_formulas_as_text(formulas_path)
                if self.verbose:
                    print(f"   📄 Formulas: {formulas_path}")

        # Save metadata
        if include_metadata:
            meta_path = Path(str(base_path) + '_metadata.json')
            with open(meta_path, 'w') as f:
                json.dump(self._extract_metadata(), f, indent=2, default=str)

        if self.verbose:
            print(f"   📄 Results: {results_path}")
            if include_metadata:
                print(f"   📄 Metadata: {meta_path}")

    def _save_formulas_as_text(self, filepath: Path) -> None:
        """Save model formulas as readable text file."""
        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MODEL FORMULAS\n")
            f.write("=" * 80 + "\n\n")
            f.write("Note: These are text representations only.\n")
            f.write("Fitted model objects are not included in this export.\n")
            f.write("To use models for predictions, load from pickle format.\n\n")
            f.write("=" * 80 + "\n\n")

            for (predictors, target), models in self.selector.model_descriptors.items():
                f.write(f"Target: {target}\n")
                f.write(f"Predictors: {' + '.join(predictors)}\n")
                f.write("-" * 80 + "\n")

                for model_type, descriptor in models.items():
                    try:
                        formula = descriptor.get_formula(notation='math')
                        f.write(f"\n{model_type}:\n")
                        f.write(f"{formula}\n")
                    except Exception as e:
                        f.write(f"\n{model_type}: [Formula unavailable: {e}]\n")

                f.write("\n" + "=" * 80 + "\n\n")

    def _load_from_csv(self, filepath: Path) -> None:
        """
        Load results from CSV files (metrics only).

        **Warning:** Does not restore fitted models.
        """
        if filepath.suffix == '.csv':
            base_path = filepath.with_suffix('')
        else:
            base_path = filepath

        results_path = Path(str(base_path) + '_results.csv')

        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")

        # Load results
        results_df = pd.read_csv(results_path)
        results = self._dataframe_to_results(results_df)
        self._inject_results(results)

    # ========================================================================
    # EXCEL METHODS (Metrics only with nice formatting)
    # ========================================================================

    def _save_as_excel(self, filepath: Path, include_metadata: bool) -> None:
        """
        Save results as Excel file (metrics only, no fitted models).

        Creates multiple sheets for organization.
        """
        if filepath.suffix not in ['.xlsx', '.xls']:
            filepath = filepath.with_suffix('.xlsx')

        filepath.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Sheet 1: Main results (all combinations)
            results_df = self._results_to_dataframe()
            results_df.to_excel(writer, sheet_name='All_Results', index=False)

            # Sheet 2: Best combinations summary
            best_df = self._best_combinations_to_dataframe()
            best_df.to_excel(writer, sheet_name='Best_Combinations', index=False)

            # Sheet 3: Formulas (if available)
            if self.selector_type == 'ParametricModelSelector':
                if hasattr(self.selector, 'model_descriptors') and self.selector.model_descriptors:
                    formulas_df = self._formulas_to_dataframe()
                    formulas_df.to_excel(writer, sheet_name='Formulas', index=False)

            # Sheet 4: Metadata
            if include_metadata:
                meta_df = self._metadata_to_dataframe()
                meta_df.to_excel(writer, sheet_name='Metadata', index=False)

            # Sheet 5: Per-target summary (if multiple targets)
            if len(self.selector.target_dimensions) > 1:
                target_summary_df = self._target_summary_to_dataframe()
                target_summary_df.to_excel(writer, sheet_name='Target_Summary', index=False)

        n_sheets = 3 + include_metadata + (1 if len(self.selector.target_dimensions) > 1 else 0)
        if self.verbose:
            print(f"   📊 Excel file with {n_sheets} sheets (metrics only)")

    def _formulas_to_dataframe(self) -> pd.DataFrame:
        """Convert model formulas to DataFrame for Excel export."""
        rows = []

        for (predictors, target), models in self.selector.model_descriptors.items():
            for model_type, descriptor in models.items():
                try:
                    formula_math = descriptor.get_formula(notation='math')
                    formula_latex = descriptor.get_formula(notation='latex')

                    row = {
                        'target': target,
                        'predictors': ' + '.join(predictors),
                        'model_type': model_type,
                        'formula_math': formula_math,
                        'formula_latex': formula_latex,
                        'n_parameters': descriptor.get_effective_parameters()
                    }

                    # Add coefficient info if available
                    if hasattr(descriptor, 'coefficients') and descriptor.coefficients:
                        for var, coef in descriptor.coefficients.items():
                            row[f'coef_{var}'] = coef

                    rows.append(row)

                except Exception as e:
                    rows.append({
                        'target': target,
                        'predictors': ' + '.join(predictors),
                        'model_type': model_type,
                        'formula_math': f'[Error: {e}]',
                        'formula_latex': '',
                        'n_parameters': np.nan
                    })

        return pd.DataFrame(rows)

    def _load_from_excel(self, filepath: Path) -> None:
        """Load results from Excel file (metrics only)."""
        # Load main results sheet
        results_df = pd.read_excel(filepath, sheet_name='All_Results')
        results = self._dataframe_to_results(results_df)
        self._inject_results(results)

    # ========================================================================
    # JSON METHODS (Metrics only, portable)
    # ========================================================================

    def _save_as_json(self, filepath: Path, include_metadata: bool) -> None:
        """Save results as JSON file (metrics only, no fitted models)."""
        if filepath.suffix != '.json':
            filepath = filepath.with_suffix('.json')

        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert to JSON-serializable format
        data = {
            'selector_type': self.selector_type,
            'results': self._results_to_dict(),
            'version': '2.0',
            'saved_at': datetime.now().isoformat()
        }

        if include_metadata:
            data['metadata'] = self._extract_metadata()

        # ✅ NEW: Add formulas if available
        if self.selector_type == 'ParametricModelSelector':
            if hasattr(self.selector, 'model_descriptors') and self.selector.model_descriptors:
                data['formulas'] = self._formulas_to_dict()

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=self._json_default)

    def _formulas_to_dict(self) -> Dict:
        """Convert formulas to JSON-serializable dict."""
        formulas = {}

        for (predictors, target), models in self.selector.model_descriptors.items():
            key = f"{'+'.join(predictors)}→{target}"
            formulas[key] = {}

            for model_type, descriptor in models.items():
                try:
                    formulas[key][model_type] = {
                        'formula_math': descriptor.get_formula(notation='math'),
                        'formula_latex': descriptor.get_formula(notation='latex'),
                        'n_parameters': descriptor.get_effective_parameters()
                    }

                    # Add coefficients if available
                    if hasattr(descriptor, 'coefficients') and descriptor.coefficients:
                        formulas[key][model_type]['coefficients'] = descriptor.coefficients

                except Exception as e:
                    formulas[key][model_type] = {'error': str(e)}

        return formulas

    def _load_from_json(self, filepath: Path) -> None:
        """Load results from JSON file (metrics only)."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Validate selector type
        saved_type = data.get('selector_type')
        if saved_type != self.selector_type:
            warnings.warn(
                f"Loaded results are from {saved_type}, "
                f"but current selector is {self.selector_type}"
            )

        results = self._dict_to_results(data['results'])
        self._inject_results(results)

    @staticmethod
    def _json_default(obj):
        """JSON serializer for objects not serializable by default."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        return str(obj)

    # ========================================================================
    # ADAPTER METHODS: Extract/Inject data from/to selectors
    # ========================================================================

    def _extract_results(self) -> Dict:
        """Extract results from selector in a unified format."""
        if self.selector_type == 'ParametricModelSelector':
            return {
                'model_recommendations': self.selector.model_recommendations,
                'detailed_results': self.selector.detailed_results
            }

        elif self.selector_type == 'RandomForestSelector':
            return {
                'results': self.selector.results,
                'feature_importance': getattr(self.selector, 'feature_importance', {}),
                'preprocessors': getattr(self.selector, 'preprocessors', {})
            }

        else:
            return {
                'results': getattr(self.selector, 'results', {}),
                'model_recommendations': getattr(self.selector, 'model_recommendations', {})
            }

    def _inject_results(self, results: Dict) -> None:
        """Inject loaded results back into selector."""
        if self.selector_type == 'ParametricModelSelector':
            self.selector.model_recommendations = results.get('model_recommendations', {})
            self.selector.detailed_results = results.get('detailed_results', {})

        elif self.selector_type == 'RandomForestSelector':
            self.selector.results = results.get('results', {})
            self.selector.feature_importance = results.get('feature_importance', {})
            self.selector.preprocessors = results.get('preprocessors', {})

        else:
            if 'results' in results:
                self.selector.results = results['results']
            if 'model_recommendations' in results:
                self.selector.model_recommendations = results['model_recommendations']

    def _extract_metadata(self) -> Dict[str, Any]:
        """Extract configuration metadata from selector."""
        base_metadata = {
            'selector_type': self.selector_type,
            'n_targets': len(self.selector.target_dimensions),
            'n_variables': len(self.selector.variable_dimensions),
            'target_dimensions': self.selector.target_dimensions,
            'variable_dimensions': self.selector.variable_dimensions,
            'fixed_dimensions': getattr(self.selector, 'fixed_dimensions', []),
        }

        if self.selector_type == 'ParametricModelSelector':
            base_metadata.update({
                'selection_metric': self.selector.selection_metric,
                'selection_strategy': self.selector.model_selector.selection_strategy,
                'assumptions': self.selector.model_selector.assumptions,
                'vif_threshold': self.selector.model_selector.vif_threshold,
                'include_regularized': self.selector.include_regularized,
                'use_cv_regularization': self.selector.use_cv_regularization,
                'enable_piecewise': self.selector.enable_piecewise,
                'fit_mfp': self.selector.fit_mfp,
                'n_combinations_tested': len(self.selector.model_recommendations),
                'has_model_descriptors': hasattr(self.selector, 'model_descriptors') and \
                                        bool(self.selector.model_descriptors)
            })

        elif self.selector_type == 'RandomForestSelector':
            base_metadata.update({
                'selection_metric': self.selector.selection_metric,
                'rf_params': self.selector.rf_params,
                'n_combinations_tested': sum(len(combos) for combos in self.selector.results.values()),
            })

        return base_metadata

    def _count_results(self) -> int:
        """Count number of results loaded."""
        if self.selector_type == 'ParametricModelSelector':
            return len(self.selector.model_recommendations)
        elif self.selector_type == 'RandomForestSelector':
            return sum(len(combos) for combos in self.selector.results.values())
        return 0

    # ========================================================================
    # CONVERSION TO DATAFRAME (continues in next section...)
    # ========================================================================

    def _results_to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame (adapts to selector type)."""
        if self.selector_type == 'ParametricModelSelector':
            return self._parametric_results_to_dataframe()
        elif self.selector_type == 'RandomForestSelector':
            return self._rf_results_to_dataframe()
        else:
            raise NotImplementedError(
                f"DataFrame conversion not implemented for {self.selector_type}"
            )

    def _parametric_results_to_dataframe(self) -> pd.DataFrame:
        """Convert ParametricModelSelector results to DataFrame."""
        rows = []

        for (predictors, target), models in self.selector.detailed_results.items():
            for model_name, metrics in models.items():
                row = {
                    'target': target,
                    'predictors': ' + '.join(predictors),
                    'n_predictors': len(predictors),
                    'model_type': model_name,
                    'recommended': (self.selector.model_recommendations.get((predictors, target)) == model_name),
                    **metrics
                }
                rows.append(row)

        return pd.DataFrame(rows)

    def _rf_results_to_dataframe(self) -> pd.DataFrame:
        """Convert RandomForestSelector results to DataFrame."""
        rows = []

        for target, combos in self.selector.results.items():
            for combo_name, data in combos.items():
                metrics = data['metrics']

                row = {
                    'target': target,
                    'predictors': combo_name,
                    'n_predictors': len(data['predictors']),
                    'combination_type': data.get('combination_type', 'other'),
                    'feature_engineered': data.get('feature_engineered', False),
                    'r2': metrics['r2'],
                    'adj_r2': metrics['adj_r2'],
                    'rrmse': metrics['rrmse'],
                    'mape': metrics['mape'],
                    'rmse': metrics['rmse'],
                    'mae': metrics['mae'],
                    'n_samples': metrics['n_samples'],
                    'max_vif': metrics.get('max_vif', 1.0),
                }

                # Add VIF values
                vif_values = metrics.get('vif_values', {})
                for feat, vif in vif_values.items():
                    row[f'vif_{feat}'] = vif

                # Add feature importance
                feature_imp = metrics.get('feature_importance', {})
                for feat, imp in feature_imp.items():
                    row[f'importance_{feat}'] = imp

                rows.append(row)

        return pd.DataFrame(rows)

    def _best_combinations_to_dataframe(self) -> pd.DataFrame:
        """Create summary of best combinations (adapts to selector type)."""
        if self.selector_type == 'ParametricModelSelector':
            return self._parametric_best_to_dataframe()
        elif self.selector_type == 'RandomForestSelector':
            return self._rf_best_to_dataframe()
        else:
            return pd.DataFrame()

    def _parametric_best_to_dataframe(self) -> pd.DataFrame:
        """Best combinations for ParametricModelSelector."""
        rows = []

        for target in self.selector.target_dimensions:
            # Get all combinations for this target
            candidates = [
                (preds, model, self.selector.detailed_results[(preds, tgt)][model])
                for (preds, tgt), model in self.selector.model_recommendations.items()
                if tgt == target
            ]

            if not candidates:
                continue

            # Sort by selection metric
            metric = self.selector.selection_metric
            candidates.sort(
                key=lambda x: x[2][metric],
                reverse=(metric in ['r2', 'adj_r2'])
            )

            best_preds, best_model, best_metrics = candidates[0]

            rows.append({
                'target': target,
                'predictors': ' + '.join(best_preds),
                'n_predictors': len(best_preds),
                'model_type': best_model,
                'r2': best_metrics['r2'],
                'adj_r2': best_metrics['adj_r2'],
                'rrmse': best_metrics['rrmse'],
                'mape': best_metrics['mape'],
                'aic': best_metrics.get('aic', np.nan),
                'bic': best_metrics.get('bic', np.nan),
                'max_vif': best_metrics.get('max_vif', np.nan),
                'heteroscedasticity': best_metrics.get('heteroscedasticity', 'unknown')
            })

        return pd.DataFrame(rows)

    def _rf_best_to_dataframe(self) -> pd.DataFrame:
        """Best combinations for RandomForestSelector."""
        best_combos = self.selector.get_best_combinations()

        rows = []
        for target, (combo_name, data) in best_combos.items():
            metrics = data['metrics']

            rows.append({
                'target': target,
                'predictors': combo_name,
                'n_predictors': len(data['predictors']),
                'combination_type': data.get('combination_type', 'other'),
                'feature_engineered': data.get('feature_engineered', False),
                'r2': metrics['r2'],
                'adj_r2': metrics['adj_r2'],
                'rrmse': metrics['rrmse'],
                'mape': metrics['mape'],
                'max_vif': metrics.get('max_vif', 1.0),
            })

        return pd.DataFrame(rows)

    def _target_summary_to_dataframe(self) -> pd.DataFrame:
        """Create per-target summary statistics."""
        rows = []

        for target in self.selector.target_dimensions:
            if self.selector_type == 'ParametricModelSelector':
                n_combos = len([
                    k for k in self.selector.model_recommendations.keys()
                    if k[1] == target
                ])

                target_results = [
                    self.selector.detailed_results[k][v]
                    for k, v in self.selector.model_recommendations.items()
                    if k[1] == target
                ]

            elif self.selector_type == 'RandomForestSelector':
                if target not in self.selector.results:
                    continue

                n_combos = len(self.selector.results[target])
                target_results = [
                    data['metrics']
                    for data in self.selector.results[target].values()
                ]

            else:
                continue

            if not target_results:
                continue

            # Calculate summary stats
            r2_values = [m['r2'] for m in target_results]
            rrmse_values = [m['rrmse'] for m in target_results]

            rows.append({
                'target': target,
                'n_combinations_tested': n_combos,
                'best_r2': max(r2_values),
                'median_r2': np.median(r2_values),
                'worst_r2': min(r2_values),
                'best_rrmse': min(rrmse_values),
                'median_rrmse': np.median(rrmse_values),
                'worst_rrmse': max(rrmse_values),
            })

        return pd.DataFrame(rows)

    def _metadata_to_dataframe(self) -> pd.DataFrame:
        """Convert metadata to DataFrame for Excel."""
        meta = self._extract_metadata()
        rows = []

        for key, value in meta.items():
            if isinstance(value, list):
                value = ', '.join(str(v) for v in value)
            elif isinstance(value, dict):
                value = json.dumps(value, default=str)
            rows.append({'parameter': key, 'value': str(value)})

        return pd.DataFrame(rows)

    # ========================================================================
    # CONVERSION FROM DATAFRAME (for loading)
    # ========================================================================

    def _dataframe_to_results(self, df: pd.DataFrame) -> Dict:
        """Convert DataFrame back to results dict (adapts to selector type)."""
        if self.selector_type == 'ParametricModelSelector':
            return self._dataframe_to_parametric_results(df)
        elif self.selector_type == 'RandomForestSelector':
            return self._dataframe_to_rf_results(df)
        else:
            raise NotImplementedError(
                f"DataFrame conversion not implemented for {self.selector_type}"
            )

    def _dataframe_to_parametric_results(self, df: pd.DataFrame) -> Dict:
        """Convert DataFrame to ParametricModelSelector format."""
        model_recommendations = {}
        detailed_results = {}

        for _, row in df.iterrows():
            predictors = tuple(row['predictors'].split(' + '))
            target = row['target']
            model_name = row['model_type']

            key = (predictors, target)

            # Reconstruct metrics dict
            metrics = row.drop(['target', 'predictors', 'n_predictors', 'model_type', 'recommended']).to_dict()

            # Build detailed_results
            if key not in detailed_results:
                detailed_results[key] = {}
            detailed_results[key][model_name] = metrics

            # Build recommendations (only if this was the recommended model)
            if row.get('recommended', False):
                model_recommendations[key] = model_name

        return {
            'model_recommendations': model_recommendations,
            'detailed_results': detailed_results
        }

    def _dataframe_to_rf_results(self, df: pd.DataFrame) -> Dict:
        """Convert DataFrame to RandomForestSelector format."""
        results = {}

        for _, row in df.iterrows():
            target = row['target']
            combo_name = row['predictors']

            # Extract predictor names
            if ' (Feat Eng)' in combo_name:
                base_predictors = combo_name.replace(' (Feat Eng)', '')
                predictors = tuple(base_predictors.split(' + '))
                feature_engineered = True
            else:
                predictors = tuple(combo_name.split(' + '))
                feature_engineered = False

            # Build metrics dict
            metrics = {}
            for col in df.columns:
                if col in ['target', 'predictors', 'n_predictors', 'combination_type', 'feature_engineered']:
                    continue
                if col.startswith('vif_') or col.startswith('importance_'):
                    continue
                metrics[col] = row[col]

            # Add VIF values
            vif_values = {}
            for col in df.columns:
                if col.startswith('vif_'):
                    feat = col.replace('vif_', '')
                    vif_values[feat] = row[col]
            if vif_values:
                metrics['vif_values'] = vif_values

            # Add feature importance
            feature_imp = {}
            for col in df.columns:
                if col.startswith('importance_'):
                    feat = col.replace('importance_', '')
                    feature_imp[feat] = row[col]
            if feature_imp:
                metrics['feature_importance'] = feature_imp

            # Build result entry
            if target not in results:
                results[target] = {}

            results[target][combo_name] = {
                'predictors': predictors,
                'metrics': metrics,
                'feature_engineered': feature_engineered,
                'combination_type': row.get('combination_type', 'other')
            }

        return {'results': results}

    # ========================================================================
    # CONVERSION TO/FROM DICT (for JSON)
    # ========================================================================

    def _results_to_dict(self) -> Dict:
        """Convert results to JSON-serializable dict."""
        if self.selector_type == 'ParametricModelSelector':
            return {
                'model_recommendations': {
                    f"{'+'.join(preds)}→{target}": model
                    for (preds, target), model in self.selector.model_recommendations.items()
                },
                'detailed_results': {
                    f"{'+'.join(preds)}→{target}": models
                    for (preds, target), models in self.selector.detailed_results.items()
                }
            }

        elif self.selector_type == 'RandomForestSelector':
            return {
                'results': self.selector.results,
                'feature_importance': self.selector.feature_importance
            }

        return {}

    def _dict_to_results(self, data: Dict) -> Dict:
        """Convert dict back to results format."""
        if self.selector_type == 'ParametricModelSelector':
            model_recommendations = {}
            detailed_results = {}

            # Convert model_recommendations
            for key, model in data.get('model_recommendations', {}).items():
                preds_str, target = key.split('→')
                predictors = tuple(preds_str.split('+'))
                model_recommendations[(predictors, target)] = model

            # Convert detailed_results
            for key, models in data.get('detailed_results', {}).items():
                preds_str, target = key.split('→')
                predictors = tuple(preds_str.split('+'))
                detailed_results[(predictors, target)] = models

            return {
                'model_recommendations': model_recommendations,
                'detailed_results': detailed_results
            }

        elif self.selector_type == 'RandomForestSelector':
            return data

        return {}

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def _check_metadata_compatibility(self, saved_metadata: Dict) -> None:
        """Warn if loaded results have different configuration."""
        current = self._extract_metadata()

        important_keys = [
            'selection_metric', 'n_targets', 'n_variables',
            'target_dimensions', 'variable_dimensions'
        ]

        if self.selector_type == 'ParametricModelSelector':
            important_keys.extend([
                'selection_strategy', 'assumptions', 'vif_threshold'
            ])
        elif self.selector_type == 'RandomForestSelector':
            important_keys.extend(['rf_params'])

        mismatches = []
        for key in important_keys:
            if key in saved_metadata and key in current:
                if saved_metadata[key] != current[key]:
                    mismatches.append(
                        f"{key}: saved={saved_metadata[key]}, current={current[key]}"
                    )

        if mismatches:
            print("\n⚠️  Warning: Loaded results have different configuration:")
            for msg in mismatches:
                print(f"   • {msg}")
            print()

    # ========================================================================
    # CONVENIENCE FUNCTIONS
    # ========================================================================

    def save_selector_results(selector,
                              filepath: str,
                              format: str = 'auto',
                              **kwargs) -> None:
        """
        Convenience function to save selector results.

        Parameters
        ----------
        selector : ParametricModelSelector or RandomForestSelector
            Fitted selector with results
        filepath : str
            Path to save results
        format : str
            Format: 'pickle', 'csv', 'excel', 'json', or 'auto'
        **kwargs
            Additional arguments passed to ResultsExporter.save()

        Examples
        --------
        >>> save_selector_results(selector, 'results/models.pkl')
        >>> save_selector_results(selector, 'results/models.xlsx', format='excel')
        """
        exporter = ResultsExporter(selector)
        exporter.save(filepath, format=format, **kwargs)

    def load_selector_results(selector,
                              filepath: str,
                              format: str = 'auto') -> None:
        """
        Convenience function to load selector results.

        Parameters
        ----------
        selector : ParametricModelSelector or RandomForestSelector
            Selector instance to load results into
        filepath : str
            Path to saved results
        format : str
            Format: 'pickle', 'csv', 'excel', 'json', or 'auto'

        Examples
        --------
        >>> selector = ParametricModelSelector(training_data, ...)
        >>> load_selector_results(selector, 'results/models.pkl')
        """
        exporter = ResultsExporter(selector, skip_validation=True)
        exporter.load(filepath, format=format)

    def quick_save_selector(selector,
                            name: str = 'results',
                            output_dir: str = 'results') -> None:
        """
        Quick save with both pickle and Excel.

        Examples
        --------
        >>> quick_save_selector(selector, 'ship_models_v1')
        """
        exporter = ResultsExporter(selector)
        exporter.quick_save(name, output_dir)