from pathlib import Path
from typing import List
import pandas as pd
import ast

from knowledge_layer.literature_regressions.literature_formulas_class import LiteratureFormula
from data_processing.file_import_class import DataImporter


# ============================================================================
# LITERATURE FORMULA LIBRARY (UNIFIED PARSING PIPELINE)
# ============================================================================

class LiteratureFormulaLibrary:
    """
    Enhanced library loader with unified parsing pipeline.

    Supports all formula types through intelligent data-driven detection:
    - Single-predictor formulas (power, polynomial, etc.)
    - Multi-predictor formulas (additive_power, multiplicative_power, etc.)
    - Piecewise formulas (piecewise_power, piecewise_linear, etc.)

    The parser adapts to the data structure automatically.
    """

    @staticmethod
    def from_file(file_path: str, reference: str = "Literature",
                  shiptype=None, sheet_name: str = None) -> List[LiteratureFormula]:
        """
        Load formulas from Excel/CSV with unified parsing pipeline.

        Parameters
        ----------
        file_path : str
            Path to Excel or CSV file
        reference : str
            Default reference for formulas
        shiptype : ShipType, optional
            Ship type for selecting Excel sheet
        sheet_name : str, optional
            Explicit sheet name (overrides shiptype)

        Returns
        -------
        List[LiteratureFormula]
            List of parsed formulas
        """

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Load file
        if file_path.suffix in ['.xlsx', '.xls']:
            target_sheet = sheet_name if sheet_name else (shiptype.name if shiptype else 0)
            df = pd.read_excel(file_path, sheet_name=target_sheet)
        elif file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        # Normalize column names
        df.columns = df.columns.str.strip().str.lower()

        # UNIFIED PIPELINE: Parse all rows
        formulae = LiteratureFormulaLibrary._unified_parse(df, reference)

        print(f"\n✓ Successfully loaded {len(formulae)} formulae from {file_path}")
        return formulae

    @staticmethod
    def from_directory(
            directory_path: str,
            shiptype=None,
            reference: str = "Literature",
            file_types: List[str] = None
    ) -> List[LiteratureFormula]:
        """
        Load formulas from all Excel files in a directory.

        This method:
        1. Discovers all Excel files in directory
        2. Loads the appropriate sheet (by shiptype) from each file
        3. Parses each file's formulas
        4. Combines and deduplicates the results

        Parameters
        ----------
        directory_path : str
            Path to directory containing formula files
        shiptype : ShipType, optional
            Ship type for selecting sheets
        reference : str
            Default reference for formulas
        file_types : List[str], optional
            File extensions to search for (default: ['.xlsx', '.xls'])

        Returns
        -------
        List[LiteratureFormula]
            Combined list of unique formulas
        """

        directory_path = Path(directory_path)

        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        # Initialize importer
        importer = DataImporter(verbose=True)

        # Find all Excel files
        if file_types is None:
            file_types = ['.xlsx', '.xls']

        file_paths = importer.file_manager.list_files(
            directory_path,
            file_types=file_types,
            full_path=True
        )

        if not file_paths:
            print(f"  Warning: No Excel files found in {directory_path}")
            return []

        print(f"\n📂 Loading formulas from {len(file_paths)} files in {directory_path}")

        # Collect formulas from all files
        all_formulae = []

        for file_path in file_paths:
            try:
                # Determine sheet name
                sheet_name = shiptype.name if shiptype else 0

                # Try to load this sheet
                try:
                    formulae = LiteratureFormulaLibrary.from_file(
                        file_path=file_path,
                        reference=reference,
                        shiptype=shiptype,
                        sheet_name=sheet_name
                    )
                    all_formulae.extend(formulae)
                    print(f"  ✓ {file_path.name}: {len(formulae)} formulas")

                except ValueError as e:
                    # Sheet might not exist in this file
                    if "Worksheet" in str(e) or "sheet" in str(e).lower():
                        print(f"  ⊘ {file_path.name}: No '{sheet_name}' sheet (skipped)")
                    else:
                        raise

            except Exception as e:
                print(f"  ✗ {file_path.name}: Error - {e}")
                continue

        # Deduplicate
        unique_formulae = LiteratureFormulaLibrary._deduplicate_formulas(all_formulae)

        n_removed = len(all_formulae) - len(unique_formulae)
        if n_removed > 0:
            print(f"\n🔄 Removed {n_removed} duplicate formulas")

        print(f"✅ Total: {len(unique_formulae)} unique formulas loaded")

        return unique_formulae

    @staticmethod
    def _deduplicate_formulas(
            formulae: List[LiteratureFormula]
    ) -> List[LiteratureFormula]:
        """
        Remove duplicate formulas based on identity criteria.

        Two formulas are considered duplicates if they have:
        - Same predictor(s) (order-independent)
        - Same target
        - Same formula type
        - Same reference

        If duplicates found, keeps the first occurrence and warns.

        Parameters
        ----------
        formulae : List[LiteratureFormula]
            List possibly containing duplicates

        Returns
        -------
        List[LiteratureFormula]
            List with duplicates removed
        """
        seen = {}
        unique = []
        duplicates = []

        for formula in formulae:
            # Create identity key (order-independent for predictors)
            pred_key = '|'.join(sorted(formula.predictors))
            key = (
                pred_key,
                formula.target,
                formula.formula_type,
                formula.reference
            )

            if key not in seen:
                seen[key] = formula
                unique.append(formula)
            else:
                # Found a duplicate
                duplicates.append({
                    'key': key,
                    'first': seen[key].name,
                    'duplicate': formula.name
                })

        # Report duplicates
        if duplicates:
            print(f"\n  ⚠️  Found {len(duplicates)} duplicate formulas:")
            for dup in duplicates[:5]:  # Show first 5
                pred_key, target, formula_type, ref = dup['key']
                print(f"    • {pred_key} → {target} ({formula_type}) [{ref}]")
            if len(duplicates) > 5:
                print(f"    ... and {len(duplicates) - 5} more")

        return unique

    @staticmethod
    def _unified_parse(df: pd.DataFrame, reference: str) -> List[LiteratureFormula]:
        """
        Unified parsing pipeline that handles all formula types.

        Strategy:
        1. Parse each row into a standardized dict
        2. Group rows by (predictor, target, formula_type)
        3. For each group, detect if piecewise or single-row
        4. Build appropriate formula object

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with formula data
        reference : str
            Default reference for formulas

        Returns
        -------
        List[LiteratureFormula]
            List of parsed formulas
        """

        # Step 1: Identify predictor column (SIMPLIFIED)
        if 'predictors' in df.columns:
            predictor_col = 'predictors'
        elif 'predictor' in df.columns:
            # Backward compatibility: accept 'predictor' but treat it as 'predictors'
            predictor_col = 'predictor'
            if len(df) > 0:  # Only warn once if data exists
                print("  Note: Column 'predictor' found. Consider renaming to 'predictors' for clarity.")
        else:
            raise ValueError(
                "DataFrame must have a 'predictors' column. "
                "(Legacy 'predictor' column also supported for backward compatibility.)"
            )

        # Step 2: Parse all rows into standardized format
        parsed_rows = []
        for idx, row in df.iterrows():
            try:
                parsed = LiteratureFormulaLibrary._parse_row_to_dict(row, predictor_col, reference)
                if parsed is not None:
                    parsed['_row_index'] = idx  # Track original position
                    parsed_rows.append(parsed)
            except Exception as e:
                print(f"  Warning: Error parsing row {idx + 1}: {e}")
                continue

        if len(parsed_rows) == 0:
            print("  Warning: No valid rows found in file")
            return []

        # Step 3: Group by (predictor_key, target, formula_type)
        groups = {}
        for parsed in parsed_rows:
            key = (parsed['predictor_key'],
                   parsed['target'],
                   parsed['formula_type'],
                   parsed['reference'])
            if key not in groups:
                groups[key] = []
            groups[key].append(parsed)

        # NEW: Report duplicates (same formula from same reference)
        for group_key, group_rows in list(groups.items()):
            if len(group_rows) > 1:
                has_bins = any(r['bin_lower'] is not None for r in group_rows)

                if not has_bins:
                    predictor_key, target, formula_type, ref = group_key  # Unpack 4-tuple
                    print(f"\n  Warning: Found {len(group_rows)} duplicate rows for "
                          f"{predictor_key} → {target} ({formula_type}) [{ref}]")
                    print(f"    Excel row indices: {[r['_row_index'] + 2 for r in group_rows]}")
                    print(f"    Keeping only the first row.")

                    # Keep only first row
                    groups[group_key] = [group_rows[0]]

        # Step 4: Build formulas from groups
        formulae = []
        for group_key, group_rows in groups.items():
            try:
                formula = LiteratureFormulaLibrary._build_formula_from_group(group_rows, reference)
                if formula is not None:
                    formulae.append(formula)
            except Exception as e:
                predictor_key, target, formula_type, ref = group_key
                print(f"  Warning: Error building formula {predictor_key} → {target} ({formula_type}) [{ref}] : {e}")
                continue

        return formulae

    @staticmethod
    def _parse_row_to_dict(row: pd.Series, predictor_col: str, reference: str) -> dict:
        """
        Parse a single row into a standardized dictionary.

        Extracts all parameters regardless of formula type:
        - Simple parameters: a, b, c, d, e
        - Special parameters: min, max, offset, constant
        - Numbered parameters: a1, b1, a2, b2, etc. (for multi-predictor)
        - Bin boundaries: bin_lower, bin_upper (for piecewise)

        Parameters
        ----------
        row : pd.Series
            Single row from DataFrame
        predictor_col : str
            Name of predictor column ('predictor' or 'predictors')
        reference : str
            Default reference

        Returns
        -------
        dict or None
            Standardized formula data, or None if row is invalid
        """

        # Skip empty rows
        if pd.isna(row.get(predictor_col)) or pd.isna(row.get('target')):
            return None

        # Parse predictors
        predictor_val = row[predictor_col]
        try:
            predictors = ast.literal_eval(str(predictor_val))
            if isinstance(predictors, str):
                predictors = [predictors]
            elif not isinstance(predictors, (list, tuple)):
                predictors = [str(predictors)]
            else:
                predictors = list(predictors)
        except:
            predictors = [str(predictor_val).strip()]

        # Clean predictor names
        predictors = [p.strip().strip("'\"[]") for p in predictors]

        # Create key for grouping (deterministic string representation)
        predictor_key = '|'.join(sorted(predictors))

        # Parse target and formula_type
        target = str(row['target']).strip()
        formula_type = str(row['formula_type']).strip().lower()

        # Extract simple parameters (a, b, c, d, e)
        parameters = {}
        simple_param_names = ['a', 'b', 'c', 'd', 'e']

        for param_name in simple_param_names:
            if param_name in row.index:
                value = row[param_name]
                if pd.notna(value) and value != '-' and value != '':
                    if isinstance(value, str):
                        value = value.replace(',', '.')
                    try:
                        parameters[param_name] = float(value)
                    except (ValueError, TypeError):
                        pass

        # Extract special parameters (min, max, offset)
        special_param_names = ['min', 'max', 'offset']
        for param_name in special_param_names:
            if param_name in row.index:
                value = row[param_name]
                if pd.notna(value) and value != '-' and value != '':
                    if isinstance(value, str):
                        value = value.replace(',', '.')
                    try:
                        parameters[param_name] = float(value)
                    except (ValueError, TypeError):
                        pass

        # Extract numbered parameters (a1, b1, a2, b2, etc. for multi-predictor)
        numbered_params = {}
        param_letters = ['a', 'b', 'c', 'd', 'e']

        for col in row.index:
            col_str = str(col).strip().lower()
            for letter in param_letters:
                # Match pattern: letter followed by digits (e.g., 'a1', 'b2', 'a10')
                if col_str.startswith(letter) and len(col_str) > 1:
                    suffix = col_str[len(letter):]
                    if suffix.isdigit():
                        idx = int(suffix) - 1  # Convert to 0-based indexing
                        value = row[col]

                        if pd.notna(value) and value != '-' and value != '':
                            # Handle string values (for functions like 'ln3')
                            if isinstance(value, str):
                                value_clean = value.strip().replace(',', '.')
                                # Check if it's a function notation (contains letters)
                                if any(c.isalpha() for c in value_clean):
                                    # Keep as string (e.g., 'ln3')
                                    if idx not in numbered_params:
                                        numbered_params[idx] = {}
                                    numbered_params[idx][letter] = value_clean
                                else:
                                    # Convert to float
                                    try:
                                        if idx not in numbered_params:
                                            numbered_params[idx] = {}
                                        numbered_params[idx][letter] = float(value_clean)
                                    except (ValueError, TypeError):
                                        pass
                            else:
                                # Numeric value
                                try:
                                    if idx not in numbered_params:
                                        numbered_params[idx] = {}
                                    numbered_params[idx][letter] = float(value)
                                except (ValueError, TypeError):
                                    pass

        # Extract bin boundaries (for piecewise formulas)
        bin_lower = None
        bin_upper = None

        if 'bin_lower' in row.index and pd.notna(row['bin_lower']):
            try:
                bin_lower = float(str(row['bin_lower']).replace(',', '.'))
            except (ValueError, TypeError):
                pass

        if 'bin_upper' in row.index and pd.notna(row['bin_upper']):
            try:
                bin_upper = float(str(row['bin_upper']).replace(',', '.'))
            except (ValueError, TypeError):
                pass

        # Extract constant (for multi-predictor formulas)
        constant = None
        if 'constant' in row.index and pd.notna(row['constant']):
            try:
                constant = float(str(row['constant']).replace(',', '.'))
            except (ValueError, TypeError):
                pass

        # Extract reference
        row_reference = reference
        if 'reference' in row.index and pd.notna(row['reference']):
            row_reference = str(row['reference'])

        # Extract bounds (percentile bounds for applicability range)
        bounds = None
        if 'bound' in row.index and pd.notna(row['bound']):
            try:
                bounds = float(row['bound'])
            except (ValueError, TypeError):
                pass
        elif 'bounds' in row.index and pd.notna(row['bounds']):
            try:
                bounds = float(row['bounds'])
            except (ValueError, TypeError):
                pass

        # Extract terms list (for multi_linear)
        terms = None
        if 'terms' in row.index and pd.notna(row['terms']):
            try:
                # Try to parse as list
                terms = ast.literal_eval(str(row['terms']))
                if isinstance(terms, str):
                    # Single term as string
                    terms = [terms]
                elif not isinstance(terms, list):
                    terms = [str(terms)]
            except:
                # Fallback: treat as comma-separated string
                terms = [t.strip() for t in str(row['terms']).split(',')]

        # NEW: Extract domain_lower and domain_upper (can be float or dict)
        domain_lower = None
        domain_upper = None

        # Parse domain_lower
        if 'domain_lower' in row.index and pd.notna(row['domain_lower']):
            domain_lower = LiteratureFormulaLibrary._parse_domain_value(
                row['domain_lower'],
                predictors
            )
        elif bin_lower is not None:
            # Fallback to bin_lower for piecewise formulas
            domain_lower = bin_lower

        # Parse domain_upper
        if 'domain_upper' in row.index and pd.notna(row['domain_upper']):
            domain_upper = LiteratureFormulaLibrary._parse_domain_value(
                row['domain_upper'],
                predictors
            )
        elif bin_upper is not None:
            # Fallback to bin_upper for piecewise formulas
            domain_upper = bin_upper

        # NEW: Also check 'min' and 'max' columns as alternatives
        if domain_lower is None and 'min' in row.index and pd.notna(row['min']):
            # Check if this is a special parameter or domain limit
            if 'min' not in parameters:  # Not already used as piecewise parameter
                domain_lower = LiteratureFormulaLibrary._parse_domain_value(
                    row['min'],
                    predictors
                )

        if domain_upper is None and 'max' in row.index and pd.notna(row['max']):
            # Check if this is a special parameter or domain limit
            if 'max' not in parameters:  # Not already used as piecewise parameter
                domain_upper = LiteratureFormulaLibrary._parse_domain_value(
                    row['max'],
                    predictors
                )

        return {
            'predictors': predictors,
            'predictor_key': predictor_key,
            'target': target,
            'formula_type': formula_type,
            'parameters': parameters,
            'numbered_params': numbered_params,
            'constant': constant,
            'bin_lower': bin_lower,
            'bin_upper': bin_upper,
            'domain_lower': domain_lower,  # Can be float or dict
            'domain_upper': domain_upper,  # Can be float or dict
            'reference': row_reference,
            'bounds': bounds,
            'terms': terms
        }

    @staticmethod
    def _parse_domain_value(value, predictors: List[str]):
        """
        Parse domain value which can be:
        - Single float: for single-predictor formulas
        - Dict string: for multi-predictor formulas

        Parameters
        ----------
        value : any
            Raw value from Excel (could be float, string, etc.)
        predictors : List[str]
            List of predictor names for validation

        Returns
        -------
        float or dict or None
            Parsed domain value

        Examples
        --------
        Single predictor:
        >>> _parse_domain_value(5000.0, ['DWT'])
        5000.0

        Multi predictor - dict string:
        >>> _parse_domain_value("{'DWT': 5000, 'Vs': 20}", ['DWT', 'Vs'])
        {'DWT': 5000.0, 'Vs': 20.0}

        Multi predictor - single value (applied to all):
        >>> _parse_domain_value(100.0, ['DWT', 'Vs'])
        {'DWT': 100.0, 'Vs': 100.0}
        """
        if pd.isna(value) or value == '-' or value == '':
            return None

        # Try to parse as dictionary first (for multi-predictor)
        if isinstance(value, str):
            value_str = value.strip()

            # Check if it looks like a dict
            if value_str.startswith('{') and value_str.endswith('}'):
                try:
                    parsed_dict = ast.literal_eval(value_str)
                    if isinstance(parsed_dict, dict):
                        # Convert all values to float
                        result = {}
                        for key, val in parsed_dict.items():
                            key_clean = str(key).strip()
                            try:
                                result[key_clean] = float(val)
                            except (ValueError, TypeError):
                                print(f"  Warning: Could not convert '{key}': '{val}' to float")

                        return result
                except (ValueError, SyntaxError) as e:
                    print(f"  Warning: Could not parse dict '{value_str}': {e}")

            # Try to parse as float
            value_str = value_str.replace(',', '.')
            try:
                float_val = float(value_str)
                # Single value for single predictor
                if len(predictors) == 1:
                    return float_val
                # Single value for multiple predictors - apply to all
                else:
                    print(f"  Warning: Single value {float_val} provided for multi-predictor formula {predictors}. "
                      f"Use dict notation like {{'DWT': {float_val}}} to specify which dimension(s).")
                return None
            except (ValueError, TypeError):
                pass

        # Try direct float conversion
        try:
            float_val = float(value)
            if len(predictors) == 1:
                return float_val
            else:
            # CHANGED: Don't auto-apply to all predictors
                print(f"  Warning: Single value {float_val} provided for multi-predictor formula {predictors}. "
                  f"Use dict notation to specify dimensions.")
                return None
        except (ValueError, TypeError):
            pass

        return None

    @staticmethod
    def _build_formula_from_group(group_rows: List[dict], reference: str) -> LiteratureFormula:
        """
        Build a LiteratureFormula from a group of parsed rows.

        Automatically detects:
        - Single-row regular formula
        - Single-row multi-predictor formula
        - Multi-row piecewise formula

        Parameters
        ----------
        group_rows : List[dict]
            List of parsed row dictionaries with same predictor/target/type
        reference : str
            Default reference

        Returns
        -------
        LiteratureFormula
            Constructed formula object
        """

        # Get common attributes from first row
        first_row = group_rows[0]
        predictors = first_row['predictors']
        target = first_row['target']
        formula_type = first_row['formula_type']
        row_reference = first_row['reference']
        bounds = first_row['bounds']

        # Detect if piecewise (multiple rows with bin boundaries)
        has_bins = any(row['bin_lower'] is not None and row['bin_upper'] is not None
                       for row in group_rows)

        if has_bins and len(group_rows) > 1:
            # PIECEWISE FORMULA
            return LiteratureFormulaLibrary._build_piecewise_formula(
                group_rows, predictors, target, formula_type, row_reference, bounds
            )

        elif len(group_rows) == 1:
            # SINGLE-ROW FORMULA
            row = group_rows[0]

            # Determine if multi-predictor or single-predictor
            if len(row['numbered_params']) > 0:
                # MULTI-PREDICTOR FORMULA (has a1, b1, a2, b2, etc.)
                params = row['numbered_params'].copy()

                # Add constant if present
                if row['constant'] is not None:
                    params['constant'] = row['constant']

                if row['terms'] is not None:
                    params['terms'] = row['terms']

                # Add any global parameters (a, b, c without numbers)
                for key, val in row['parameters'].items():
                    if key not in params:
                        params[key] = val

            else:
                # SINGLE-PREDICTOR FORMULA (uses a, b, c, d, e)
                params = row['parameters'].copy()

            # Create formula name
            pred_names = '_'.join([
                p.replace(' ', '_')
                for p in predictors
            ])
            target_clean = target.replace(' ', '_')
            formula_name = f"{pred_names}_to_{target_clean}_{formula_type}"

            # Extract domain limits
            domain_lower = row.get('domain_lower')
            domain_upper = row.get('domain_upper')

            return LiteratureFormula(
                name=formula_name,
                predictors=predictors,
                target=target,
                formula_type=formula_type,
                parameters=params,
                reference=row_reference,
                bounds=bounds,
                domain_lower=domain_lower,
                domain_upper=domain_upper
            )

        else:
            # Multiple rows without bins - ERROR
            raise ValueError(
                f"Multiple rows found for {predictors} → {target} ({formula_type}) but no bin boundaries. "
                f"Either provide bin_lower/bin_upper for piecewise formulas, or ensure only one row per formula."
            )

    @staticmethod
    def _build_piecewise_formula(group_rows: List[dict], predictors: List[str],
                                 target: str, formula_type: str,
                                 reference: str, bounds: float) -> LiteratureFormula:
        """
        Build a piecewise formula from multiple rows with bin boundaries.

        Parameters
        ----------
        group_rows : List[dict]
            List of parsed rows for this piecewise formula
        predictors : List[str]
            Predictor variable names
        target : str
            Target variable name
        formula_type : str
            Formula type
        reference : str
            Formula reference/citation
        bounds : float
            Percentile bounds

        Returns
        -------
        LiteratureFormula
            Piecewise formula object
        """

        # Validate single predictor
        if len(predictors) != 1:
            raise ValueError(
                f"Piecewise formulas only support single predictor. "
                f"Got {len(predictors)} predictors: {predictors}"
            )

        # Auto-convert formula type if needed
        if not formula_type.startswith('piecewise'):
            print(f"  Info: Converting '{formula_type}' to 'piecewise_{formula_type}' "
                  f"for multi-row formula {predictors[0]} → {target}")
            formula_type = f"piecewise_{formula_type}"

        # Sort rows by bin_lower
        sorted_rows = sorted(group_rows, key=lambda r: r['bin_lower'] if r['bin_lower'] is not None else 0)

        # Extract bins and parameters
        bins = []
        bin_parameters = []

        for i, row in enumerate(sorted_rows):
            bin_lower = row['bin_lower']
            bin_upper = row['bin_upper']

            if bin_lower is None or bin_upper is None:
                raise ValueError(
                    f"Row {i + 1} of piecewise formula {predictors[0]} → {target} missing bin boundaries"
                )

            # Add lower bound of first bin
            if len(bins) == 0:
                bins.append(bin_lower)
            else:
                # Validate continuity (allow small gaps due to floating point)
                if abs(bins[-1] - bin_lower) > 0.1:
                    print(f"  Warning: Gap detected in bins for {predictors[0]} → {target}: "
                          f"{bins[-1]:.1f} → {bin_lower:.1f}")

            # Add upper bound
            bins.append(bin_upper)

            # Get parameters for this segment
            segment_params = row['parameters'].copy()

            # Fill in defaults for missing parameters
            for param in ['a', 'b', 'c', 'd', 'e']:
                if param not in segment_params:
                    # Default values: a=0, b=0, c=1, d=0, e=1
                    if param in ['a', 'b', 'd']:
                        segment_params[param] = 0.0
                    elif param in ['c', 'e']:
                        segment_params[param] = 1.0

            bin_parameters.append(segment_params)

        # Validate bins structure
        if len(bins) != len(bin_parameters) + 1:
            raise ValueError(
                f"Inconsistent bin structure for {predictors[0]} → {target}: "
                f"{len(bins)} bin edges but {len(bin_parameters)} parameter sets. "
                f"Should have n_segments + 1 bin edges."
            )

        # Check for non-increasing bins
        for i in range(len(bins) - 1):
            if bins[i] >= bins[i + 1]:
                raise ValueError(
                    f"Non-increasing bins for {predictors[0]} → {target}: "
                    f"bin[{i}]={bins[i]} >= bin[{i + 1}]={bins[i + 1]}"
                )

        # Create formula name
        pred_name = predictors[0].replace(' ', '_')
        target_clean = target.replace(' ', '_')
        formula_name = f"{pred_name}_to_{target_clean}_{formula_type}"

        return LiteratureFormula(
            name=formula_name,
            predictors=predictors,
            target=target,
            formula_type=formula_type,
            parameters={},  # Empty for piecewise - data is in bins/bin_parameters
            reference=reference,
            bounds=bounds,
            bins=bins,
            bin_parameters=bin_parameters,
            domain_lower = bins[0],
            domain_upper = bins[-1],
        )

    @staticmethod
    def create_template_file(output_path: str, format: str = 'xlsx'):
        """
        Create comprehensive template file with examples of all formula types.

        Parameters
        ----------
        output_path : str
            Path for output file
        format : str
            'xlsx' or 'csv'
        """
        template_data = {
            'Predictors': [None, None, "('LOA (M)', 'DEPTH (M)')", None, None, None, None, None],
            'Target': ['LBP (M)', 'BEAM (M)', 'DWT', 'LPP (M)', 'LPP (M)', 'LPP (M)', 'LPP (M)', 'BEAM'],
            'formula_type': [
                'power',
                'cubic_polynomial',
                'additive_power',
                'piecewise_power',
                'piecewise_power',
                'piecewise_power',
                'piecewise_power',
                'power'
            ],
            'bin_lower': [None, None, None, 0, 10000, 25000, 55000, None],
            'bin_upper': [None, None, None, 10000, 25000, 55000, 60000, None],
            'a': [7.603, 0.0, None, 5.582, 5.463, 25.66, 124.18, 11.0],
            'b': [0.0, -0.01, None, 0.329, 0.3285, 0.1813, 0, 0.001],
            'c': [0.300, 0.28, None, None, None, None, 0.00107, 1],
            'd': [None, None, None, 0, 0, 0, 0, 0],
            'e': [None, None, None, 0, 0, 0, 0, 0],
            'a1': [None, None, 1.5, None, None, None, None, None],
            'b1': [None, None, 2.0, None, None, None, None, None],
            'a2': [None, None, 0.5, None, None, None, None, None],
            'b2': [None, None, 1.5, None, None, None, None, None],
            'constant': [None, None, 100, None, None, None, None, None],
            'min': [None, None, None, None, None, None, None, None],
            'max': [None, None, None, None, None, None, None, None],
            'offset': [None, None, None, None, None, None, None, None],
            'bounds': [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            'reference': [
                'Example: Single power law',
                'Example: Cubic polynomial',
                'Example: Multi-predictor',
                'Example: Piecewise (segment 1)',
                'Example: Piecewise (segment 2)',
                'Example: Piecewise (segment 3)',
                'Example: Piecewise (segment 4)',
                'Example: Simple power with all params'
            ]
        }

        df = pd.DataFrame(template_data)

        if format == 'xlsx':
            df.to_excel(output_path, index=False)
        elif format == 'csv':
            df.to_csv(output_path, index=False)

        print(f"✓ Template created: {output_path}")
        print(f"\nTemplate includes:")
        print(f"  - Row 1: Single-predictor power law (DWT → LBP)")
        print(f"  - Row 2: Single-predictor cubic polynomial (LOA → BEAM)")
        print(f"  - Row 3: Multi-predictor additive power ((LOA, DEPTH) → DWT)")
        print(f"  - Rows 4-7: Piecewise power law with 4 segments (DWT → LPP)")
        print(f"  - Row 8: Power law with explicit d, e parameters")
        print(f"\nAll formula types can coexist in the same file!")

    @staticmethod
    def get_default_formulae() -> List[LiteratureFormula]:
        """
        Get hardcoded default formulae (backwards compatibility).

        Returns
        -------
        List[LiteratureFormula]
            List of default formulas
        """
        formulae = []

        # DWT → Displacement
        formulae.append(LiteratureFormula(
            name="DWT_to_Disp_power",
            predictors=['DWT'],
            target='Disp',
            formula_type='power',
            parameters={'a': 2.214420, 'b': 0.0, 'c': 0.943855, 'd': 0.0, 'e': 0.0},
            reference="Literature formula collection"
        ))

        # DWT → LBP
        formulae.append(LiteratureFormula(
            name="DWT_to_LBP_power",
            predictors=['DWT'],
            target='LBP (M)',
            formula_type='power',
            parameters={'a': 7.603010, 'b': 0.0, 'c': 0.300155, 'd': 0.0, 'e': 0.0},
            reference="Literature formula collection"
        ))

        # DWT → Beam
        formulae.append(LiteratureFormula(
            name="DWT_to_Beam_power",
            predictors=['DWT'],
            target='BEAM MLD (M)',
            formula_type='power',
            parameters={'a': 1.055900, 'b': 0.0, 'c': 0.309724, 'd': 0.0, 'e': 0.0},
            reference="Literature formula collection"
        ))

        # DWT → Depth
        formulae.append(LiteratureFormula(
            name="DWT_to_Depth_power",
            predictors=['DWT'],
            target='DEPTH (M)',
            formula_type='power',
            parameters={'a': 0.584268, 'b': 0.0, 'c': 0.310795, 'd': 0.0, 'e': 0.0},
            reference="Literature formula collection"
        ))

        # DWT → Draft
        formulae.append(LiteratureFormula(
            name="DWT_to_Draft_power",
            predictors=['DWT'],
            target='DRAFT (M)',
            formula_type='power',
            parameters={'a': 0.480719, 'b': 0.0, 'c': 0.298295, 'd': 0.0, 'e': 0.0},
            reference="Literature formula collection"
        ))

        # LOA → Beam (cubic polynomial)
        formulae.append(LiteratureFormula(
            name="LOA_to_Beam_cubic",
            predictors=['LOA (M)'],
            target='BEAM MLD (M)',
            formula_type='cubic_polynomial',
            parameters={'a': 0.0, 'b': -0.0, 'c': 0.28, 'd': 4.06, 'e': 0.0},
            reference="Literature formula collection"
        ))

        return formulae

    @staticmethod
    def get_all_formulae(file_path: str = None, reference: str = "Literature") -> List[LiteratureFormula]:
        """
        Get formulae from file or use defaults.

        Parameters
        ----------
        file_path : str, optional
            Path to Excel/CSV file. If None, returns hardcoded defaults.
        reference : str
            Default reference for formulae from file

        Returns
        -------
        List[LiteratureFormula]
            List of formulas
        """
        if file_path is not None:
            return LiteratureFormulaLibrary.from_file(file_path, reference)
        else:
            return LiteratureFormulaLibrary.get_default_formulae()

    @staticmethod
    def get_formulae_for_target(target: str, file_path: str = None) -> List[LiteratureFormula]:
        """
        Get all formulae that predict a specific target.

        Parameters
        ----------
        target : str
            Target dimension name
        file_path : str, optional
            Path to formula file

        Returns
        -------
        List[LiteratureFormula]
            Formulas targeting the specified dimension
        """
        all_formulae = LiteratureFormulaLibrary.get_all_formulae(file_path)
        return [f for f in all_formulae if f.target == target]

    @staticmethod
    def get_formula_by_name(name: str, file_path: str = None) -> LiteratureFormula:
        """
        Get a specific formula by name.

        Parameters
        ----------
        name : str
            Formula name
        file_path : str, optional
            Path to formula file

        Returns
        -------
        LiteratureFormula
            The requested formula

        Raises
        ------
        ValueError
            If formula not found
        """
        all_formulae = LiteratureFormulaLibrary.get_all_formulae(file_path)
        for f in all_formulae:
            if f.name == name:
                return f
        raise ValueError(f"Formula '{name}' not found in library")