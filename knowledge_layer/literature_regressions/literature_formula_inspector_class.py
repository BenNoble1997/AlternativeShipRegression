import numpy as np
import pandas as pd
from typing import List, Optional, Union
from pathlib import Path
import random

from knowledge_layer.literature_regressions.literature_formulas_class import LiteratureFormula
from knowledge_layer.literature_regressions.literature_formula_libraries_class import LiteratureFormulaLibrary


# ============================================================================
# LITERATURE FORMULA INSPECTOR
# ============================================================================

class LiteratureFormulaInspector:
    """
    Debugging and inspection tool for literature formulas.

    Provides detailed breakdowns of predictions, especially useful for
    multi-linear regression where you want to see each term's contribution.

    Features:
    - Select formulas by type, target, predictor, or name
    - Sample random ships from dataset
    - Show term-by-term breakdown for multi-linear formulas
    - Export inspection results to CSV
    - Compare predictions across multiple formulas
    """

    def __init__(self,
                 data: pd.DataFrame,
                 formulae: List[LiteratureFormula],
                 seed: int = 42):
        """
        Initialize inspector with data and formulas.

        Parameters
        ----------
        data : pd.DataFrame
            Ship data (typically conventional dataset)
        formulae : List[LiteratureFormula]
            List of formulas to inspect
        seed : int
            Random seed for reproducible ship selection
        """
        self.data = data
        self.formulae = formulae
        self.seed = seed

        # Set random seed
        random.seed(seed)
        np.random.seed(seed)

        print(f"✓ Inspector initialized with {len(data)} ships and {len(formulae)} formulae")

    @classmethod
    def from_file(cls,
                  data: pd.DataFrame,
                  formula_file: str,
                  reference: str = "Literature",
                  seed: int = 42,
                  sheet_name: str = None,
                  shiptype=None):
        """
        Create inspector by loading formulas from file.

        Parameters
        ----------
        data : pd.DataFrame
            Ship data to inspect formulas against
        formula_file : str
            Path to Excel/CSV file containing formulas
        reference : str, default="Literature"
            Default reference for formulas (used if not specified in file)
        seed : int, default=42
            Random seed for reproducible ship sampling
        sheet_name : str, optional
            Specific Excel sheet name to read from.
            If not provided, uses first sheet (index 0).
            Takes precedence over shiptype parameter.
        shiptype : ShipType, optional
            Ship type enum for selecting Excel sheet by name.
            Uses shiptype.name as sheet name.
            Only used if sheet_name is not provided.

        Returns
        -------
        LiteratureFormulaInspector
            Inspector instance with loaded formulas

        Examples
        --------
        # Specify sheet by name
        >>> inspector = LiteratureFormulaInspector.from_file(
        ...     data=df,
        ...     formula_file='formulas.xlsx',
        ...     sheet_name='Container_Ship'
        ... )

        # Use ShipType enum
        >>> inspector = LiteratureFormulaInspector.from_file(
        ...     data=df,
        ...     formula_file='formulas.xlsx',
        ...     shiptype=ShipType.CONTAINER_SHIP
        ... )

        # Default (first sheet)
        >>> inspector = LiteratureFormulaInspector.from_file(
        ...     data=df,
        ...     formula_file='formulas.xlsx'
        ... )
        """
        formulae = LiteratureFormulaLibrary.from_file(
            file_path=formula_file,
            reference=reference,
            sheet_name=sheet_name,
            shiptype=shiptype
        )

        return cls(data, formulae, seed)

    def list_formulae(self,
                      formula_type: Optional[str] = None,
                      target: Optional[str] = None,
                      predictor: Optional[str] = None) -> pd.DataFrame:
        """
        List available formulas with optional filtering.

        Parameters
        ----------
        formula_type : str, optional
            Filter by formula type (e.g., 'multi_linear', 'power')
        target : str, optional
            Filter by target variable
        predictor : str, optional
            Filter by predictor (must be in predictor list)

        Returns
        -------
        pd.DataFrame
            Summary of matching formulas
        """
        matching = self.formulae

        # Apply filters
        if formula_type:
            matching = [f for f in matching if f.formula_type == formula_type.lower()]

        if target:
            matching = [f for f in matching if f.target == target]

        if predictor:
            matching = [f for f in matching if predictor in f.predictors]

        # Build summary
        summary = []
        for f in matching:
            summary.append({
                'Name': f.name,
                'Type': f.formula_type,
                'Predictors': ', '.join(f.predictors),
                'Target': f.target,
                'Reference': f.reference,
                'N_Predictors': len(f.predictors)
            })

        df = pd.DataFrame(summary)

        print(f"\n{'=' * 80}")
        print(f"FORMULA SUMMARY ({len(matching)} matching)")
        print(f"{'=' * 80}")
        if len(df) > 0:
            print(df.to_string(index=False))
        else:
            print("No matching formulas found")

        return df

    def get_formula(self,
                    name: Optional[str] = None,
                    formula_type: Optional[str] = None,
                    target: Optional[str] = None) -> Optional[LiteratureFormula]:
        """
        Get a single formula by name or criteria.

        Parameters
        ----------
        name : str, optional
            Exact formula name
        formula_type : str, optional
            Formula type filter
        target : str, optional
            Target variable filter

        Returns
        -------
        LiteratureFormula or None
        """
        if name:
            for f in self.formulae:
                if f.name == name:
                    return f
            print(f"⚠️  Formula '{name}' not found")
            return None

        # Filter by criteria
        matching = self.formulae
        if formula_type:
            matching = [f for f in matching if f.formula_type == formula_type.lower()]
        if target:
            matching = [f for f in matching if f.target == target]

        if len(matching) == 0:
            print("⚠️  No formulas match the criteria")
            return None
        elif len(matching) > 1:
            print(f"⚠️  Multiple formulas match ({len(matching)}). Specify name or refine criteria.")
            print("Matching formulas:")
            for f in matching:
                print(f"  - {f.name}")
            return None

        return matching[0]

    def sample_ships(self,
                     n: int = 5,
                     required_cols: Optional[List[str]] = None,
                     filter_valid: bool = True) -> pd.DataFrame:
        """
        Sample random ships from dataset.

        Parameters
        ----------
        n : int
            Number of ships to sample
        required_cols : List[str], optional
            Specific columns that must not be NaN.
            If None and filter_valid=True, removes rows with any NaN in any column.
            If provided, only checks these specific columns for NaN.
        filter_valid : bool
            If True, remove ships with NaN values

        Returns
        -------
        pd.DataFrame
            Sample of ships

        Examples
        --------
        # Sample ships that have DWT and LOA filled
        >>> sample = inspector.sample_ships(n=5, required_cols=['DWT', 'LOA (M)'])

        # Sample ships with no NaN in any column
        >>> sample = inspector.sample_ships(n=5)

        # Sample any 5 ships (including those with NaN)
        >>> sample = inspector.sample_ships(n=5, filter_valid=False)
        """
        data = self.data.copy()

        if filter_valid:
            if required_cols is not None:
                # Only check specific columns for NaN
                missing_cols = [col for col in required_cols if col not in data.columns]
                if missing_cols:
                    print(f"⚠️  Columns not in dataset: {missing_cols}")
                    return pd.DataFrame()

                data = data.dropna(subset=required_cols)
            else:
                # Check all columns for NaN (original behavior)
                data = data.dropna()

        if len(data) < n:
            if len(data) == 0:
                print(f"⚠️  No valid ships available")
                return pd.DataFrame()
            else:
                print(f"⚠️  Only {len(data)} valid ships available (requested {n})")
                n = len(data)

        sample = data.sample(n=n, random_state=self.seed)

        print(f"\n✓ Sampled {len(sample)} ships (seed={self.seed})")

        return sample

    def inspect_formula(self,
                        formula: Union[str, LiteratureFormula],
                        n_ships: int = 5,
                        ship_indices: Optional[List[int]] = None,
                        show_intermediate: bool = True,
                        export_path: Optional[str] = None,
                        enforce_domain:bool=True) -> pd.DataFrame:
        """
        Inspect formula predictions on sample ships with detailed breakdown.

        Parameters
        ----------
        formula : str or LiteratureFormula
            Formula name or formula object
        n_ships : int
            Number of ships to inspect (if ship_indices not provided)
        ship_indices : List[int], optional
            Specific ship indices to inspect
        show_intermediate : bool
            Show intermediate calculations (especially for multi_linear)
        export_path : str, optional
            Path to save results as CSV

        enforce_domain : bool, default=True
        If True, only sample ships within formula's domain limits

        Returns
        -------
        pd.DataFrame
            Inspection results
        """
        # Get formula
        if isinstance(formula, str):
            formula = self.get_formula(name=formula)
            if formula is None:
                return pd.DataFrame()

        print(f"\n{'=' * 80}")
        print(f"INSPECTING: {formula.name}")
        print(f"{'=' * 80}")
        print(f"Type: {formula.formula_type}")
        print(f"Predictors: {', '.join(formula.predictors)}")
        print(f"Target: {formula.target}")
        print(f"Reference: {formula.reference}")
        print(f"Formula: {formula.get_formula_text()}")

        domain_info = formula.get_domain_info()
        print(f"Domain: {domain_info}")


        # Get sample ships
        if ship_indices is not None:
            sample = self.data.loc[ship_indices].copy()
            print(f"\n✓ Using specified {len(sample)} ships")
        else:
            # Check if all predictors and target are available
            required_cols = formula.predictors + [formula.target]
            available_data = self.data.dropna(subset=required_cols)

            if len(available_data) == 0:
                print(f"\n⚠️  No ships have all required columns: {required_cols}")
                print(f"   Available columns: {list(self.data.columns)}")
                return pd.DataFrame()

            print(f"\n✓ Found {len(available_data)} ships with all required columns")

            # NEW: Filter by domain if requested
            if enforce_domain:
                before_filter = len(available_data)
                available_data = formula.filter_data_by_domain(available_data)
                after_filter = len(available_data)

                if after_filter < before_filter:
                    print(f"✓ Filtered to {after_filter} ships within domain limits")

                if after_filter == 0:
                    print(f"\n⚠️  No ships within formula's domain limits!")
                    print(f"   Consider setting enforce_domain=False to see all predictions")
                    return pd.DataFrame()

            # SIMPLER FIX: Sample directly from available_data
            n_available = len(available_data)
            if n_available < n_ships:
                print(f"⚠️  Only {len(available_data)} valid ships available (requested {n_ships})")
                n_ships = n_available

            if n_ships == 0:
                print(f"⚠️  No valid ships available")
                return pd.DataFrame()

            sample = available_data.sample(n=n_ships, random_state=self.seed)
            print(f"\n✓ Sampled {len(sample)} ships (seed={self.seed})")

            # Handle case where no ships could be sampled
            if len(sample) == 0:
                return pd.DataFrame()

        # Check for missing columns (should not happen now, but keep as safety)
        missing_predictors = [p for p in formula.predictors if p not in sample.columns]
        if missing_predictors:
            print(f"\n❌ Missing predictor columns: {missing_predictors}")
            return pd.DataFrame()

        if formula.target not in sample.columns:
            print(f"\n⚠️  Target column '{formula.target}' not in data")
            has_actual = False
        else:
            has_actual = True

        # Build inspection dataframe
        results = []

        for idx, row in sample.iterrows():
            # ... [rest of the loop remains the same] ...
            result_row = {'Ship_Index': idx}

            # Add predictor values
            X_values = []
            for pred in formula.predictors:
                value = row[pred]
                result_row[f"Input_{pred}"] = value
                X_values.append(value)

            X = np.array(X_values).reshape(1, -1)

            # Get prediction
            try:
                prediction = formula.predict(X)[0]
                result_row['Predicted'] = prediction
            except Exception as e:
                print(f"\n❌ Error predicting for ship {idx}: {e}")
                result_row['Predicted'] = np.nan
                results.append(result_row)
                continue

            # Add actual value if available
            if has_actual:
                actual = row[formula.target]
                result_row['Actual'] = actual
                result_row['Error'] = actual - prediction
                result_row['Error_%'] = 100 * (actual - prediction) / actual if actual != 0 else np.nan

            # Add intermediate calculations for multi_linear
            if formula.formula_type == 'multi_linear' and show_intermediate:
                if 'terms' in formula.parameters:
                    # Add constant
                    if 'constant' in formula.parameters:
                        result_row['Term_Constant'] = formula.parameters['constant']

                    # Add each term
                    terms = formula.parameters['terms']
                    for i, term_str in enumerate(terms):
                        if i in formula.parameters:
                            coef = formula.parameters[i].get('a', 0.0)

                            # Evaluate term
                            try:
                                term_value = formula._evaluate_term(term_str, X)[0]
                                contribution = coef * term_value

                                result_row[f'Term_{i}_{term_str}'] = term_value
                                result_row[f'Coef_{i}'] = coef
                                result_row[f'Contrib_{i}_{term_str}'] = contribution
                            except Exception as e:
                                print(f"  ⚠️  Error evaluating term '{term_str}': {e}")

            results.append(result_row)

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # ADDED: Check if DataFrame is empty
        if len(df) == 0:
            print(f"\n⚠️  No ships could be processed")
            return df

        # Display results
        print(f"\n{'=' * 80}")
        print(f"INSPECTION RESULTS")
        print(f"{'=' * 80}\n")

        # Basic results (inputs, prediction, actual, error)
        basic_cols = ['Ship_Index'] + [f'Input_{p}' for p in formula.predictors] + ['Predicted']
        if has_actual:
            basic_cols.extend(['Actual', 'Error', 'Error_%'])

        basic_df = df[basic_cols]
        print(basic_df.to_string(index=False))

        # Intermediate calculations for multi_linear
        if formula.formula_type == 'multi_linear' and show_intermediate:
            print(f"\n{'=' * 80}")
            print(f"TERM-BY-TERM BREAKDOWN")
            print(f"{'=' * 80}\n")

            # Constant
            if 'Term_Constant' in df.columns:
                print(f"Constant: {df['Term_Constant'].iloc[0]:.6f}")
                print()

            # Each term
            if 'terms' in formula.parameters:
                terms = formula.parameters['terms']
                for i, term_str in enumerate(terms):
                    if f'Term_{i}_{term_str}' in df.columns:
                        print(f"Term {i}: {term_str}")
                        print(f"  Coefficient: {df[f'Coef_{i}'].iloc[0]:.6e}")

                        term_df = df[['Ship_Index', f'Term_{i}_{term_str}',
                                      f'Contrib_{i}_{term_str}']]
                        term_df.columns = ['Ship', 'Term_Value', 'Contribution']
                        print(term_df.to_string(index=False))
                        print()

        # Summary statistics
        if has_actual and len(df) > 1:
            print(f"\n{'=' * 80}")
            print(f"SUMMARY STATISTICS")
            print(f"{'=' * 80}\n")
            print(f"Mean Absolute Error: {df['Error'].abs().mean():.4f}")
            print(f"RMSE: {np.sqrt((df['Error'] ** 2).mean()):.4f}")
            print(f"Mean Absolute % Error: {df['Error_%'].abs().mean():.2f}%")

            if formula.target in sample.columns:
                actual_mean = sample[formula.target].mean()
                pred_mean = df['Predicted'].mean()
                print(f"\nActual mean: {actual_mean:.4f}")
                print(f"Predicted mean: {pred_mean:.4f}")
                print(f"Bias: {pred_mean - actual_mean:.4f}")

        # Export if requested
        if export_path:
            df.to_csv(export_path, index=False)
            print(f"\n✓ Results exported to {export_path}")

        return df

    def inspect_all(self,
                    formula_type: Optional[str] = None,
                    target: Optional[str] = None,
                    n_ships: int = 5,
                    export_dir: Optional[str] = None) -> dict:
        """
        Inspect all formulas matching criteria.

        Parameters
        ----------
        formula_type : str, optional
            Filter by formula type
        target : str, optional
            Filter by target variable
        n_ships : int
            Number of ships to inspect per formula
        export_dir : str, optional
            Directory to save individual CSV files

        Returns
        -------
        dict
            Dictionary mapping formula names to inspection DataFrames
        """
        # Filter formulas
        matching = self.formulae
        if formula_type:
            matching = [f for f in matching if f.formula_type == formula_type.lower()]
        if target:
            matching = [f for f in matching if f.target == target]

        print(f"\n{'=' * 80}")
        print(f"INSPECTING {len(matching)} FORMULAS")
        print(f"{'=' * 80}")

        results = {}

        for i, formula in enumerate(matching, 1):
            print(f"\n[{i}/{len(matching)}]")

            # Prepare export path
            export_path = None
            if export_dir:
                Path(export_dir).mkdir(parents=True, exist_ok=True)
                export_path = f"{export_dir}/{formula.name}_inspection.csv"

            # Inspect formula
            df = self.inspect_formula(
                formula=formula,
                n_ships=n_ships,
                show_intermediate=True,
                export_path=export_path
            )

            results[formula.name] = df

        print(f"\n{'=' * 80}")
        print(f"COMPLETED INSPECTION OF {len(matching)} FORMULAS")
        print(f"{'=' * 80}")

        return results

    def compare_formulas(self,
                         formulas: List[Union[str, LiteratureFormula]],
                         n_ships: int = 5,
                         ship_indices: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Compare predictions from multiple formulas on same ships.

        Parameters
        ----------
        formulas : List[str or LiteratureFormula]
            List of formula names or formula objects
        n_ships : int
            Number of ships to compare
        ship_indices : List[int], optional
            Specific ships to use

        Returns
        -------
        pd.DataFrame
            Comparison results
        """
        # Resolve formulas
        resolved_formulas = []
        for f in formulas:
            if isinstance(f, str):
                formula = self.get_formula(name=f)
                if formula is None:
                    continue
                resolved_formulas.append(formula)
            else:
                resolved_formulas.append(f)

        if len(resolved_formulas) == 0:
            print("❌ No valid formulas to compare")
            return pd.DataFrame()

        # Check they all have same target
        targets = set(f.target for f in resolved_formulas)
        if len(targets) > 1:
            print(f"⚠️  Warning: Formulas have different targets: {targets}")

        # Get all required predictors
        all_predictors = set()
        for f in resolved_formulas:
            all_predictors.update(f.predictors)

        target = resolved_formulas[0].target
        required_cols = list(all_predictors) + ([target] if target in self.data.columns else [])

        # Get sample
        if ship_indices is not None:
            sample = self.data.loc[ship_indices].copy()
        else:
            available_data = self.data.dropna(subset=required_cols)
            if len(available_data) == 0:
                print(f"❌ No ships have all required columns")
                return pd.DataFrame()
            sample = available_data.sample(n=min(n_ships, len(available_data)),
                                           random_state=self.seed)

        print(f"\n{'=' * 80}")
        print(f"COMPARING {len(resolved_formulas)} FORMULAS")
        print(f"{'=' * 80}")
        for f in resolved_formulas:
            print(f"  - {f.name}")
        print()

        # Build comparison DataFrame
        results = []

        for idx, row in sample.iterrows():
            result_row = {'Ship_Index': idx}

            # Add all predictor values
            for pred in all_predictors:
                if pred in row.index:
                    result_row[f'Input_{pred}'] = row[pred]

            # Add actual target if available
            if target in row.index:
                result_row['Actual'] = row[target]

            # Get predictions from each formula
            for formula in resolved_formulas:
                # Extract predictor values for this formula
                try:
                    X_values = [row[pred] for pred in formula.predictors]
                    X = np.array(X_values).reshape(1, -1)
                    prediction = formula.predict(X)[0]

                    result_row[f'Pred_{formula.name}'] = prediction

                    # Calculate error if actual available
                    if target in row.index:
                        actual = row[target]
                        error = actual - prediction
                        error_pct = 100 * error / actual if actual != 0 else np.nan
                        result_row[f'Error_{formula.name}'] = error
                        result_row[f'Error%_{formula.name}'] = error_pct

                except Exception as e:
                    print(f"  ⚠️  Error with {formula.name} on ship {idx}: {e}")
                    result_row[f'Pred_{formula.name}'] = np.nan

            results.append(result_row)

        df = pd.DataFrame(results)

        # Display results
        print(df.to_string(index=False))

        # Summary statistics
        if target in self.data.columns and len(df) > 1:
            print(f"\n{'=' * 80}")
            print(f"COMPARISON SUMMARY")
            print(f"{'=' * 80}\n")

            for formula in resolved_formulas:
                if f'Error_{formula.name}' in df.columns:
                    mae = df[f'Error_{formula.name}'].abs().mean()
                    rmse = np.sqrt((df[f'Error_{formula.name}'] ** 2).mean())
                    mape = df[f'Error%_{formula.name}'].abs().mean()

                    print(f"{formula.name}:")
                    print(f"  MAE: {mae:.4f}")
                    print(f"  RMSE: {rmse:.4f}")
                    print(f"  MAPE: {mape:.2f}%")
                    print()

        return df


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_inspect(data: pd.DataFrame,
                  formula_file: str,
                  formula_name: str,
                  n_ships: int = 5,
                  reference: str = "Literature"):
    """
    Quick one-liner to inspect a formula.

    Parameters
    ----------
    data : pd.DataFrame
        Ship data
    formula_file : str
        Path to formula file
    formula_name : str
        Name of formula to inspect
    n_ships : int
        Number of ships
    reference : str
        Formula reference

    Returns
    -------
    pd.DataFrame
        Inspection results
    """
    inspector = LiteratureFormulaInspector.from_file(data, formula_file, reference)
    return inspector.inspect_formula(formula_name, n_ships=n_ships)


def quick_compare(data: pd.DataFrame,
                  formula_file: str,
                  formula_names: List[str],
                  n_ships: int = 5,
                  reference: str = "Literature"):
    """
    Quick one-liner to compare formulas.

    Parameters
    ----------
    data : pd.DataFrame
        Ship data
    formula_file : str
        Path to formula file
    formula_names : List[str]
        Names of formulas to compare
    n_ships : int
        Number of ships
    reference : str
        Formula reference

    Returns
    -------
    pd.DataFrame
        Comparison results
    """
    inspector = LiteratureFormulaInspector.from_file(data, formula_file, reference)
    return inspector.compare_formulas(formula_names, n_ships=n_ships)