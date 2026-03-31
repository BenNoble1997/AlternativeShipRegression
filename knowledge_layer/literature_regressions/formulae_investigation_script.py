from literature_formula_inspector_class import quick_inspect, quick_compare
from literature_formula_inspector_class import LiteratureFormulaInspector
from parameters.shipshape.ShipShapeBasic_class import ShipType
from knowledge_layer.regression_analysis_new import load_fleet_data, getting_alt_fuel_ships
import numpy as np
import pandas as pd

from pathlib import Path


def diagnose_term_evaluation(formula, ship_data: dict):
    """
    Detailed diagnostic showing exactly how each term is evaluated.

    Parameters
    ----------
    formula : LiteratureFormula
        The formula to diagnose
    ship_data : dict
        Dictionary with predictor values, e.g. {'DWT': 7115, 'Vs': 24}
    """

    print("=" * 80)
    print("TERM EVALUATION DIAGNOSTIC")
    print("=" * 80)

    print(f"\nFormula: {formula.name}")
    print(f"Type: {formula.formula_type}")
    print(f"Predictors: {formula.predictors}")

    # Create input array
    X_values = [ship_data[pred] for pred in formula.predictors]
    X = np.array(X_values).reshape(1, -1)

    print(f"\n{'=' * 80}")
    print(f"INPUT VALUES")
    print(f"{'=' * 80}")
    for i, pred in enumerate(formula.predictors):
        print(f"  {pred} = {ship_data[pred]}")
        print(f"  → X[:, {i}] = {X[0, i]}")

    # Get the terms
    if 'terms' not in formula.parameters:
        print("\n⚠️  Formula has no 'terms' parameter - not a multi_linear formula")
        return

    terms = formula.parameters['terms']

    print(f"\n{'=' * 80}")
    print(f"TERMS TO EVALUATE ({len(terms)} terms)")
    print(f"{'=' * 80}")
    for i, term in enumerate(terms):
        print(f"  Term {i}: '{term}'")

    # Evaluate constant
    print(f"\n{'=' * 80}")
    print(f"CONSTANT TERM")
    print(f"{'=' * 80}")
    if 'constant' in formula.parameters:
        print(f"  Value: {formula.parameters['constant']}")
    else:
        print(f"  No constant term")

    # Evaluate each term in detail
    print(f"\n{'=' * 80}")
    print(f"TERM-BY-TERM EVALUATION")
    print(f"{'=' * 80}")

    total = formula.parameters.get('constant', 0.0)
    print(f"\nStarting value (constant): {total}")

    for i, term_str in enumerate(terms):
        print(f"\n{'-' * 80}")
        print(f"TERM {i}: '{term_str}'")
        print(f"{'-' * 80}")

        # Get coefficient
        if i not in formula.parameters:
            print(f"  ⚠️  No coefficient found for term {i}")
            continue

        coef = formula.parameters[i].get('a', 0.0)
        print(f"  Coefficient: {coef}")

        # Parse the term
        print(f"\n  Parsing term: '{term_str}'")

        # Check for different term types
        term_clean = term_str.strip()

        # CASE 1: Simple predictor (e.g., 'DWT', 'Vs')
        if term_clean in formula.predictors:
            pred_idx = formula.predictors.index(term_clean)
            print(f"    → Simple predictor: {term_clean}")
            print(f"    → Maps to X[:, {pred_idx}]")
            print(f"    → Value: {X[0, pred_idx]}")
            term_value = X[0, pred_idx]

        # CASE 2: Multiplication (e.g., 'DWT*Vs')
        elif '*' in term_clean:
            factors = [f.strip() for f in term_clean.split('*')]
            print(f"    → Multiplication detected: {factors}")
            term_value = 1.0

            for factor in factors:
                print(f"    → Evaluating factor: '{factor}'")

                # Check if factor is a simple predictor
                if factor in formula.predictors:
                    pred_idx = formula.predictors.index(factor)
                    factor_value = X[0, pred_idx]
                    print(f"        → Simple predictor: {factor} = {factor_value}")

                # Check if factor is a power (e.g., 'Vs^2')
                elif '^' in factor:
                    base, exp = factor.split('^')
                    base = base.strip()
                    exp = int(exp.strip())

                    if base in formula.predictors:
                        pred_idx = formula.predictors.index(base)
                        base_value = X[0, pred_idx]
                        factor_value = base_value ** exp
                        print(f"        → Power: {base}^{exp} = {base_value}^{exp} = {factor_value}")
                    else:
                        print(f"        ⚠️  Base '{base}' not found in predictors")
                        factor_value = 0.0

                else:
                    print(f"        ⚠️  Unknown factor type: '{factor}'")
                    factor_value = 0.0

                term_value *= factor_value
                print(f"        → Running product: {term_value}")

            print(f"    → Final term value: {term_value}")

        # CASE 3: Power (e.g., 'Vs^2')
        elif '^' in term_clean:
            parts = term_clean.split('^')
            base = parts[0].strip()
            exponent = int(parts[1].strip())

            print(f"    → Power detected: base='{base}', exponent={exponent}")

            if base in formula.predictors:
                pred_idx = formula.predictors.index(base)
                base_value = X[0, pred_idx]
                term_value = base_value ** exponent
                print(f"    → {base}^{exponent} = {base_value}^{exponent} = {term_value}")
            else:
                print(f"    ⚠️  Base '{base}' not in predictors")
                term_value = 0.0

        # CASE 4: Division (e.g., 'DWT/Vs')
        elif '/' in term_clean:
            parts = term_clean.split('/')
            numerator = parts[0].strip()
            denominator = parts[1].strip()

            print(f"    → Division detected: {numerator}/{denominator}")

            num_value = 0.0
            if numerator in formula.predictors:
                pred_idx = formula.predictors.index(numerator)
                num_value = X[0, pred_idx]
                print(f"    → Numerator: {numerator} = {num_value}")

            denom_value = 1.0
            if denominator in formula.predictors:
                pred_idx = formula.predictors.index(denominator)
                denom_value = X[0, pred_idx]
                print(f"    → Denominator: {denominator} = {denom_value}")

            term_value = num_value / denom_value if denom_value != 0 else 0.0
            print(f"    → Result: {num_value}/{denom_value} = {term_value}")

        else:
            print(f"    ⚠️  Unknown term type: '{term_clean}'")
            term_value = 0.0

        # Calculate contribution
        contribution = coef * term_value
        total += contribution

        print(f"\n  Summary:")
        print(f"    Term value: {term_value}")
        print(f"    Coefficient: {coef}")
        print(f"    Contribution: {coef} × {term_value} = {contribution}")
        print(f"    Running total: {total}")

    print(f"\n{'=' * 80}")
    print(f"FINAL RESULT")
    print(f"{'=' * 80}")
    print(f"  Predicted value: {total}")

    # Compare with formula.predict()
    official_prediction = formula.predict(X)[0]
    print(f"\n  Formula.predict() result: {official_prediction}")
    print(f"  Match: {'✓' if abs(total - official_prediction) < 0.01 else '✗ MISMATCH!'}")

    return total


def compare_with_manual_calculation(ship_data: dict):
    """
    Manual calculation to verify term evaluation logic.
    """

    print("\n" + "=" * 80)
    print("MANUAL CALCULATION (for verification)")
    print("=" * 80)

    DWT = ship_data['DWT']
    Vs = ship_data['Vs']

    # Manual calculation based on the formula text
    print(f"\nGiven: DWT={DWT}, Vs={Vs}")
    print(f"\nFormula: y = -8187.3022 + 1.168·DWT + 1074.875·Vs - 0.1498·DWT*Vs")
    print(f"           - 46.1891·Vs² + 0.0064·DWT*Vs² + 0.6583·Vs³ - 0.0000913·DWT*Vs³")

    constant = -8187.3022
    term_0 = 1.168 * DWT
    term_1 = 1074.875 * Vs
    term_2 = -0.1498 * (DWT * Vs)
    term_3 = -46.1891 * (Vs ** 2)
    term_4 = 0.0064 * (DWT * Vs ** 2)
    term_5 = 0.6583 * (Vs ** 3)
    term_6 = -0.0000913 * (DWT * Vs ** 3)

    print(f"\nBreakdown:")
    print(f"  Constant:         {constant:>15.4f}")
    print(f"  1.168 × {DWT:<8} = {term_0:>15.4f}")
    print(f"  1074.875 × {Vs:<5} = {term_1:>15.4f}")
    print(f"  -0.1498 × ({DWT}×{Vs}) = {term_2:>15.4f}")
    print(f"  -46.1891 × {Vs}² = {term_3:>15.4f}")
    print(f"  0.0064 × ({DWT}×{Vs}²) = {term_4:>15.4f}")
    print(f"  0.6583 × {Vs}³ = {term_5:>15.4f}")
    print(f"  -0.0000913 × ({DWT}×{Vs}³) = {term_6:>15.4f}")

    total = constant + term_0 + term_1 + term_2 + term_3 + term_4 + term_5 + term_6

    print(f"\n  {'=' * 60}")
    print(f"  TOTAL: {total:.4f} m")
    print(f"  {'=' * 60}")

    return total


def check_evaluate_term_method(formula, ship_data: dict):
    """
    Test the _evaluate_term method directly on each term.
    """

    print("\n" + "=" * 80)
    print("TESTING _evaluate_term() METHOD")
    print("=" * 80)

    X_values = [ship_data[pred] for pred in formula.predictors]
    X = np.array(X_values).reshape(1, -1)

    print(f"\nInput: X = {X}")
    print(f"Predictors: {formula.predictors}")

    if 'terms' not in formula.parameters:
        print("\n⚠️  No terms found")
        return

    terms = formula.parameters['terms']

    print(f"\n{'=' * 60}")
    for i, term_str in enumerate(terms):
        try:
            result = formula._evaluate_term(term_str, X)
            print(f"Term {i}: '{term_str:<15}' → {result[0]:>15.6f}")
        except Exception as e:
            print(f"Term {i}: '{term_str:<15}' → ERROR: {e}")
    print(f"{'=' * 60}")


# ============================================================================
# MAIN TEST SCRIPT
# ============================================================================

if __name__ == "__main__":

    formula_excel = r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\literature_regressions\literature_formulae.xlsx'

    # # # Bulk carrier filepath
    # filepath=Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\bulk_carrier')
    # ship_type=ShipType.BULK_CARRIER

    # # #Container ship
    # filepath=Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\Container_ship')
    # ship_type=ShipType.CONTAINER_SHIP

    # #Tanker
    # filepath = Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\Tanker')
    # ship_type = ShipType.TANKER

    # RO PAx
    filepath = Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\ROPAX')
    ship_type = ShipType.RO_PAX
    ship_type_str = ship_type.name

    fleet = getting_alt_fuel_ships(filepath=filepath,
                                   ship_type=ship_type)

    conventional_df = fleet.processed_data['pure_conventional']
    conventional_matched_df = fleet.processed_data['matched_dwt3.5_speed']
    alternative_df = fleet.processed_data['pure_alt_fuel']
    dual_fuel_df = fleet.processed_data['dual_fuel']
    dual_fuel_matched_df = fleet.processed_data['matched_dualfuel_dwt3.5']

    inspector = LiteratureFormulaInspector.from_file(
        data=conventional_df,
        formula_file=formula_excel,
        reference=ship_type_str,
        seed=42,
        shiptype=ship_type,
    )

    # List all multi-linear formulas
    inspector.list_formulae()

    # Inspect specific formula
    results = inspector.inspect_formula(
        formula='DWT_Vs_to_LOA_(M)_multi_linear',
        n_ships=5,
        show_intermediate=True,
    )

    # Load data and formula
    conventional_data = conventional_df

    inspector = LiteratureFormulaInspector.from_file(
        data=conventional_data,
        formula_file=formula_excel,
        shiptype=ship_type,
        seed=42
    )

    formula = inspector.get_formula('DWT_Vs_to_LOA_(M)_multi_linear')

    # Test with Ship 204 from your output
    ship_data = {
        'DWT': 7115.0,
        'Vs': 24.0
    }

    print("\n" + "=" * 80)
    print("TESTING SHIP 204")
    print("=" * 80)
    print(f"DWT: {ship_data['DWT']}")
    print(f"Vs: {ship_data['Vs']}")
    print(f"Actual LOA: 186.0 m")

    # Run diagnostics
    result1 = diagnose_term_evaluation(formula, ship_data)

    result2 = compare_with_manual_calculation(ship_data)

    check_evaluate_term_method(formula, ship_data)

    # Compare all results
    print("\n" + "=" * 80)
    print("COMPARISON OF ALL METHODS")
    print("=" * 80)
    print(f"  diagnose_term_evaluation():  {result1:.4f} m")
    print(f"  Manual calculation:          {result2:.4f} m")

    X = np.array([[ship_data['DWT'], ship_data['Vs']]])
    official = formula.predict(X)[0]
    print(f"  formula.predict():           {official:.4f} m")

    print(f"\n  Expected (Actual): 186.0 m")
    print(f"  Error: {abs(186.0 - official):.1f} m ({100 * abs(186.0 - official) / 186.0:.1f}%)")

    if abs(result1 - result2) < 0.01 and abs(result1 - official) < 0.01:
        print("\n  ✓ All methods agree - term evaluation is CORRECT")
        print("  ✗ But prediction is wrong - likely coefficient/scaling issue")
    else:
        print("\n  ✗ Methods disagree - term evaluation has BUG!")

    # Test other term types
    print("\n" + "=" * 80)
    print("TESTING DIFFERENT TERM TYPES")
    print("=" * 80)

    test_cases = [
        ('DWT', 'Simple predictor'),
        ('Vs', 'Simple predictor'),
        ('DWT*Vs', 'Product of two predictors'),
        ('Vs^2', 'Power term'),
        ('DWT*Vs^2', 'Product with power'),
        ('Vs^3', 'Cubic term'),
        ('DWT*Vs^3', 'Product with cubic'),
    ]

    X = np.array([[7115.0, 24.0]])

    print(f"\nInput: DWT=7115, Vs=24")
    print(f"\n{'Term':<20} {'Description':<30} {'Evaluated Value':>20}")
    print("-" * 70)

    for term, description in test_cases:
        try:
            value = formula._evaluate_term(term, X)[0]
            print(f"{term:<20} {description:<30} {value:>20.6f}")
        except Exception as e:
            print(f"{term:<20} {description:<30} ERROR: {e}")