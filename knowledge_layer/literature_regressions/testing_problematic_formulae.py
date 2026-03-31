import numpy as np
import pandas as pd
from literature_formulas_class import LiteratureFormula
from Literature_formula_evaluation_class import LiteratureFormulaEvaluator
from knowledge_layer.regression_analysis_new import getting_alt_fuel_ships
from pathlib import Path
from parameters.shipshape.ShipShapeBasic_class import ShipType


# ============================================================================
# TEST FORMULAS WITH REAL DATA
# ============================================================================

def create_test_formulas():
    """
    Create the two formulas to test:
    1. Draft = 8.177 - 0.419*DWT^(1/3) - 3.949*TEU^(1/4) + 0.0073*ln³(DWT*TEU)
    2. Depth = -1.589 + 0.012*ln³(DWT) + 0.092*TEU - 1.19E-78*DWT^14*TEU
    """

    formulas = []

    # ========================================================================
    # FORMULA 1: DRAFT (with interaction term issue)
    # ========================================================================
    # Note: This formula has ln(DWT*TEU)³ which doesn't fit power_product perfectly
    # We'll create it but it may not work correctly

    draft_formula = LiteratureFormula(
        name="Draft_DWT_TEU_literature",
        predictors=['DWT', 'TEU'],
        target='DRAFT (M)',
        formula_type='power_product',
        parameters={
            # Individual terms
            0: {'a': -0.419, 'b': 0.333},  # -0.419 * DWT^(1/3)
            1: {'a': -3.949, 'b': 0.25},  # -3.949 * TEU^(1/4)

            # Interaction (won't work correctly - see note below)
            2: {'b': 1.0},
            3: {'b': 1.0},
            4: {'a': 0.0073, 'b': 'ln3'},

            'constant': 8.177
        },
        reference="Literature formula - Draft",
        bounds=5.0  # 5th-95th percentile
    )
    formulas.append(draft_formula)

    # ========================================================================
    # FORMULA 2: DEPTH (should work correctly)
    # ========================================================================

    depth_formula = LiteratureFormula(
        name="Depth_DWT_TEU_literature",
        predictors=['DWT', 'TEU'],
        target='DEPTH (M)',
        formula_type='power_product',
        parameters={
            # Individual terms
            0: {'a': 0.012, 'b': 'ln3'},  # 0.012 * (ln(DWT))³
            1: {'a': 0.092, 'b': 0.5},  # 0.092 * TEU

            # Interaction term
            2: {'b': 14.0},  # DWT^14 in interaction
            3: {'b': 1.0},  # TEU^1 in interaction
            4: {'a': -1.19e-78, 'b': 1.0},  # Coefficient

            'constant': -1.589
        },
        reference="Literature formula - Depth",
        bounds=5.0
    )
    formulas.append(depth_formula)

    return formulas


def run_with_evaluator(conventional: pd.DataFrame,
                       conventional_matched: pd.DataFrame,
                       alternative: pd.DataFrame,
                       save_dir: str = None):
    """
    Test formulas using LiteratureFormulaEvaluator.

    Parameters
    ----------
    conventional_data_path : str
        Path to conventional ships CSV
    conventional_matched_path : str
        Path to matched conventional ships CSV
    alternative_data_path : str
        Path to alternative fuel ships CSV
    save_dir : str, optional
        Directory to save plots
    """

    print("=" * 80)
    print("TESTING DRAFT AND DEPTH FORMULAS")
    print("=" * 80)


    print(f"  Conventional: {len(conventional)} ships")
    print(f"  Matched: {len(conventional_matched)} ships")
    print(f"  Alternative: {len(alternative)} ships")

    # Check for required columns
    required_cols = ['DWT', 'TEU', 'DRAFT (M)', 'DEPTH (M)']
    for dataset_name, dataset in [
        ('Conventional', conventional),
        ('Matched', conventional_matched),
        ('Alternative', alternative)
    ]:
        missing = [col for col in required_cols if col not in dataset.columns]
        if missing:
            print(f"\n❌ {dataset_name} missing columns: {missing}")
            print(f"   Available columns: {list(dataset.columns)}")
            return

    # Create test formulas
    print("\nCreating formulas...")
    formulas = create_test_formulas()

    for formula in formulas:
        print(f"\n  {formula.name}:")
        print(f"    Type: {formula.formula_type}")
        print(f"    Predictors: {formula.predictors}")
        print(f"    Target: {formula.target}")
        print(f"    Formula: {formula.get_formula_text()}")

    # Create evaluator
    print("\n" + "=" * 80)
    print("INITIALIZING EVALUATOR")
    print("=" * 80)

    evaluator = LiteratureFormulaEvaluator(
        conventional_data=conventional,
        conventional_matched=conventional_matched,
        alternative_data=alternative,
        formulae=formulas,
        verbose=True
    )

    # Evaluate all formulas
    print("\n" + "=" * 80)
    print("EVALUATING FORMULAS")
    print("=" * 80)

    results = evaluator.evaluate_all_formulae()

    # Interpret results
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    evaluator.interpret_results(
        significance_threshold_r2=0.05,
        significance_threshold_rrmse=10.0
    )

    # Create plots
    if save_dir:
        print("\n" + "=" * 80)
        print("CREATING PLOTS")
        print("=" * 80)

        # Comparison bar plots
        evaluator.plot_comparison(save_path=f"{save_dir}/formula_comparison.png")

        # Scatter plots (predicted vs actual)
        evaluator.plot_comparison_scatter(save_dir=save_dir)

        # Fit plots (for single-predictor only - will skip these formulas)
        # evaluator.plot_formula_fit(save_dir=save_dir)

        # Residual plots
        evaluator.plot_residuals(save_dir=save_dir)

        print(f"\n✓ All plots saved to {save_dir}")

    # Export results
    if save_dir:
        evaluator.export_results(f"{save_dir}/draft_depth_formula_results.csv")

    return evaluator, results


def manual_verification(conventional:pd.DataFrame):
    """
    Manually verify formulas on a few sample ships.
    """
    print("\n" + "=" * 80)
    print("MANUAL VERIFICATION ON SAMPLE SHIPS")
    print("=" * 80)


    # Take first 5 ships as samples
    sample = conventional.head(5)

    print(f"\nTesting on {len(sample)} sample ships:")
    print(
        f"{'Ship':<6} {'DWT':>10} {'TEU':>10} {'Actual Draft':>15} {'Pred Draft':>15} {'Actual Depth':>15} {'Pred Depth':>15}")
    print("-" * 90)

    for idx, row in sample.iterrows():
        dwt = row['DWT']
        teu = row['TEU']
        actual_draft = row['DRAFT (M)']
        actual_depth = row['DEPTH (M)']

        # Manual Draft calculation
        # Draft = 8.177 - 0.419*DWT^(1/3) - 3.949*TEU^(1/4) + 0.0073*ln³(DWT*TEU)
        draft_pred = (8.177
                      - 0.419 * (dwt ** (1 / 3))
                      - 3.949 * (teu ** 0.25)
                      + 0.0073 * (np.log(dwt * teu) ** 3))

        # Manual Depth calculation
        # Depth = -1.589 + 0.012*ln³(DWT) + 0.092*TEU - 1.19E-78*DWT^14*TEU
        depth_pred = (-1.589
                      + 0.012 * (np.log(dwt) ** 3)
                      + 0.092 * teu
                      - 1.19e-78 * (dwt ** 14) * teu)

        print(
            f"{idx:<6} {dwt:>10.0f} {teu:>10.0f} {actual_draft:>15.2f} {draft_pred:>15.2f} {actual_depth:>15.2f} {depth_pred:>15.2f}")

    # Calculate errors
    print("\n" + "=" * 80)
    print("ERROR ANALYSIS")
    print("=" * 80)

    # Draft
    dwt_vals = sample['DWT'].values
    teu_vals = sample['TEU'].values
    actual_draft = sample['DRAFT (M)'].values

    pred_draft = (8.177
                  - 0.419 * (dwt_vals ** (1 / 3))
                  - 3.949 * (teu_vals ** 0.25)
                  + 0.0073 * (np.log(dwt_vals * teu_vals) ** 3))

    draft_errors = actual_draft - pred_draft
    draft_pct_errors = 100 * draft_errors / actual_draft

    print("\nDraft:")
    print(f"  Mean error: {np.mean(draft_errors):.2f} m")
    print(f"  Mean absolute error: {np.mean(np.abs(draft_errors)):.2f} m")
    print(f"  Mean percentage error: {np.mean(draft_pct_errors):.1f}%")
    print(f"  Mean absolute percentage error: {np.mean(np.abs(draft_pct_errors)):.1f}%")

    # Depth
    actual_depth = sample['DEPTH (M)'].values

    pred_depth = (-1.589
                  + 0.012 * (np.log(dwt_vals) ** 3)
                  + 0.092 * teu_vals
                  - 1.19e-78 * (dwt_vals ** 14) * teu_vals)

    depth_errors = actual_depth - pred_depth
    depth_pct_errors = 100 * depth_errors / actual_depth

    print("\nDepth:")
    print(f"  Mean error: {np.mean(depth_errors):.2f} m")
    print(f"  Mean absolute error: {np.mean(np.abs(depth_errors)):.2f} m")
    print(f"  Mean percentage error: {np.mean(depth_pct_errors):.1f}%")
    print(f"  Mean absolute percentage error: {np.mean(np.abs(depth_pct_errors)):.1f}%")


if __name__ == "__main__":
    # ========================================================================
    # CONFIGURE YOUR PATHS HERE
    # ========================================================================
    # #Container ship
    filepath=Path(r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\Container_ship')
    ship_type=ShipType.CONTAINER_SHIP


    fleet = getting_alt_fuel_ships(filepath=filepath,
                                   ship_type=ship_type)

    CONVENTIONAL_DATA = fleet.processed_data['pure_conventional']
    CONVENTIONAL_MATCHED = fleet.processed_data['matched_dwt3.5_speed']
    ALTERNATIVE_DATA = fleet.processed_data['pure_alt_fuel']
    dual_fuel_df = fleet.processed_data['dual_fuel']
    dual_fuel_matched_df = fleet.processed_data['matched_dualfuel_dwt3.5']


    SAVE_DIR = r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\literature_regressions'

    # ========================================================================
    # RUN TESTS
    # ========================================================================

    # First, do manual verification on a few ships
    print("STEP 1: Manual Verification")
    print("=" * 80)
    manual_verification(CONVENTIONAL_DATA)

    # Then, run full evaluation
    print("\n\n")
    print("STEP 2: Full Evaluation with LiteratureFormulaEvaluator")
    print("=" * 80)
    evaluator, results = run_with_evaluator(
        conventional=CONVENTIONAL_DATA,
        conventional_matched=CONVENTIONAL_MATCHED,
        alternative=ALTERNATIVE_DATA,
        save_dir=SAVE_DIR
    )

    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)
    print(f"\nResults summary:")
    print(results[['formula_name', 'target',
                   'conventional_r2', 'matched_r2', 'alternative_r2',
                   'gap_fuel_type', 'gap_fuel_type_rrmse']].to_string(index=False))