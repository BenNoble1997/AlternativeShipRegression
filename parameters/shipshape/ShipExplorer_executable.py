"""
Ship Explorer System - Demonstration Script
===========================================

This script demonstrates the comprehensive functionality of the ShipShapeExplorer class,
which extends ShipShapeBasic with advanced ship performance calculations, appendage modeling,
and correction factor applications.

The examples show:
1. Basic ShipShapeBasic creation for foundation
2. Simple ShipShapeExplorer functionality with speed variations
3. Extended parameter testing with appendages and hull form modifications
4. Advanced correction library system for parameter adjustments

Key Features Demonstrated:
- Ship performance calculations (Froude number, Reynolds number, form factors)
- Appendage resistance modeling (rudders, bilge keels, stabilizers, etc.)
- Dynamic parameter modification with automatic recalculation
- Correction factor system for fine-tuning calculations
- Comprehensive parameter validation and error handling

Author: Ship Design Team
Date: 2025
"""

from config import ShipType
from parameters.shipshape import ShipShapeBasic
from parameters.shipshape import ShipShapeExplorer, CorrectionFactor
import logging
import logging

def print_section_header(title: str, width: int = 70):
    """
    Print a formatted section header for clear demonstration organization.

    Args:
        title: Title to display in the header
        width: Total width of the header line
    """
    print(f"\n{'=' * width}")
    print(f" {title.upper()} ")
    print(f"{'=' * width}")


def print_ship_explorer_summary(explorer: 'ShipShapeExplorer', title: str = "Ship Explorer Parameters"):
    """
    Print a comprehensive summary of ship explorer parameters and calculations.

    Args:
        explorer: ShipShapeExplorer instance
        title: Description of what this explorer represents
    """
    print(f"\n📊 {title}")
    print("-" * 60)
    print(f"Ship Type: {explorer.shiptype.name} (Key: {explorer.shiptype.key})")
    print(f"Speed: {explorer.v_knots:.1f} knots ({explorer.v_ms:.2f} m/s)")
    print(f"Froude Number (F_n): {explorer.F_n:.4f}")
    print(f"Reynolds Number (Re): {explorer.Re:.2e}")
    print(f"Block Coefficient (C_B): {explorer.C_B:.4f}")
    print(f"Wetted Surface (S): {explorer.S:.1f} m²")
    print(f"Form Factor (k1): {explorer.k1:.4f}")
    if hasattr(explorer, 'k2') and explorer.k2:
        print(f"Appendage Factors (k2): {explorer.k2}")


def get_shipbasic_example():
    """
    Create a basic ShipShapeBasic instance for use in ShipShapeExplorer demonstrations.

    This function creates the foundation ship object that will be used throughout
    the demonstration examples. It shows the basic ship creation process with
    minimal required parameters.

    Returns:
        ShipShapeBasic: A configured ship instance ready for explorer testing
    """
    print_section_header("Creating Foundation Ship (ShipShapeBasic)")

    # Database file path - replace with your actual database location
    file_path = r'C:\Users\bnoble\PycharmProjects\T1.6_deliverable\parameters\Databases\clarksons_data\CLARKSONS_27_05.xlsx'

    print("Creating a bulk carrier with basic parameters...")
    print("Input parameters:")
    print("  - Length Overall (L): 205 m")
    print("  - Deadweight Tonnage (DWT): 37,500 tonnes")
    print("  - Midship Coefficient (C_M): 0.98")
    print("  - Ship Type: Bulk Carrier")

    # Create the foundation ship instance
    shipbasic = ShipShapeBasic(
        L=205,  # Length Overall in meters
        shiptype=ShipType.BULK_CARRIER,  # Ship type enumeration
        shipdata=file_path,  # Path to ship database for interpolation
        DWT=37500,  # Deadweight tonnage in tonnes
        C_M=0.98  # Midship coefficient
    )

    # Display the key results from ShipShapeBasic
    print(f"\n🔍 ShipShapeBasic Results:")
    print(f"   Ship type key: {shipbasic.shiptype.key}")
    print(f"   Ship type name: {shipbasic.shiptype.name}")
    print(f"   Ship DWT: {shipbasic.DWT:,.0f} tonnes")
    print(f"   Interpolated L_PP: {shipbasic.L_PP:.2f} m")
    print(f"   Interpolated B: {shipbasic.B:.2f} m")
    print(f"   Interpolated T: {shipbasic.T:.2f} m")

    print(f"\n📝 Note: This ShipShapeBasic instance will serve as the foundation")
    print(f"    for all ShipShapeExplorer demonstrations that follow.")

    return shipbasic


def basic_explorer_test(shipbasic: 'ShipShapeBasic'):
    """
    Demonstrate basic ShipShapeExplorer functionality with speed variations.

    This example shows:
    - Creating a ShipShapeExplorer from a ShipShapeBasic instance
    - How speed changes affect dimensionless numbers (Froude number)
    - Dynamic parameter modification capabilities

    Args:
        shipbasic: A ShipShapeBasic instance to use as foundation
    """
    print_section_header("Basic ShipShapeExplorer Functionality")

    print("Creating ShipShapeExplorer with initial speed of 20 knots...")
    print("The explorer extends the basic ship with performance calculations.")

    # Create the explorer with an initial speed
    explorer = ShipShapeExplorer(
        shipshape=shipbasic,  # Use the foundation ship
        v_knots=20,  # Initial speed in knots
        verbose=True  # Enable detailed logging
    )

    # Display initial performance characteristics
    print_ship_explorer_summary(explorer, "Initial Configuration (20 knots)")

    print(f"\n🔍 Key Performance Indicator:")
    print(f"   Froude Number at 20 knots: {explorer.F_n:.4f}")

    # Demonstrate dynamic speed modification
    print(f"\n🔧 Changing speed from 20 knots to 25 knots...")
    explorer.v_knots = 25

    print(f"   Froude Number at 25 knots: {explorer.F_n:.4f}")

    # Calculate and display the change
    froude_change = ((explorer.F_n - (20 * 0.5144 / (9.81 * shipbasic.L) ** 0.5)) /
                     (20 * 0.5144 / (9.81 * shipbasic.L) ** 0.5)) * 100

    print(f"\n📊 Analysis:")
    print(f"   • Froude number increased by {froude_change:.1f}%")
    print(f"   • The explorer automatically recalculates all dependent parameters")

    print(f"\n📝 Technical Note:")
    print(f"   ShipShapeExplorer provides real-time performance analysis")
    print(f"   by calculating dimensionless numbers and resistance factors.")


def extended_parameter_test(shipbasic: 'ShipShapeBasic'):
    """
    Test ShipShapeExplorer with additional parameters and practice changing them.

    This comprehensive example demonstrates:
    - Advanced explorer configuration with multiple parameters
    - Appendage modeling (rudders, bilge keels, etc.)
    - Hull form modifications (stern shapes, bulbs)
    - Trim and draft condition modeling
    - Dynamic parameter modification with impact analysis

    Args:
        shipbasic: A ShipShapeBasic instance to use as foundation
    """
    print_section_header("Extended Parameter Testing")

    print("Creating advanced ShipShapeExplorer configuration...")
    print("This test demonstrates comprehensive parameter modeling including:")
    print("  • Appendage resistance (rudders, bilge keels)")
    print("  • Hull form variations (stern shapes)")
    print("  • Trim conditions and draft variations")
    print("  • Bulb bow modeling")
    print("  • Bow thruster resistance effects")

    # Create explorer with comprehensive parameter set
    explorer = ShipShapeExplorer(
        shipshape=shipbasic,
        v_knots=18,  # Operating speed
        app_types=[1, 11],  # Rudder behind skeg + bilge keels
        C_sternchoice=3,  # Normal section stern shape
        trim=2.5,  # Ship trim in meters
        _T_F=shipbasic.T - 1.25,  # Forward draft (trim/2 less than mean)
        _T_A=shipbasic.T + 1.25,  # Aft draft (trim/2 more than mean)
        bulbchoice=True,  # Include bulbous bow
        verbose=True  # Enable detailed logging
    )

    # Display comprehensive initial state
    print_ship_explorer_summary(explorer, "Advanced Configuration (18 knots)")

    print(f"\n🔍 Detailed Parameter Analysis:")
    print(f"   Forward Draft (T_F): {explorer.T_F:.2f} m")
    print(f"   Aft Draft (T_A): {explorer.T_A:.2f} m")
    print(f"   Trim Condition: {explorer.T_A - explorer.T_F:.2f} m by stern")
    print(f"   Appendage Factors (k2): {explorer.k2}")
    print(f"   Stern Coefficient: {explorer.C_stern}")
    print(f"   c14 Factor: {explorer.c14:.4f}")
    print(f"   Waterplane Coefficient (C_WP): {explorer.C_WP:.4f}")

    # Test 1: Speed modification and its effects
    print(f"\n--- Test 1: Speed Modification (18 → 22 knots) ---")
    original_froude = explorer.F_n
    original_reynolds = explorer.Re

    explorer.v_knots = 22

    print(f"Speed Effects:")
    print(f"  Froude Number: {original_froude:.4f} → {explorer.F_n:.4f}")
    print(f"  Reynolds Number: {original_reynolds:.2e} → {explorer.Re:.2e}")


    # Test 2: Stern shape modification
    print(f"\n--- Test 2: Stern Shape Modification (Normal → V-shaped) ---")
    original_c_stern = explorer.C_stern
    original_c14 = explorer.c14
    original_c_wp = explorer.C_WP

    explorer.C_sternchoice = 2  # Change to V-shaped section

    print(f"Stern Shape Effects:")
    print(f"  C_stern: {original_c_stern} → {explorer.C_stern}")
    print(f"  c14 Factor: {original_c14:.4f} → {explorer.c14:.4f}")
    print(f"  C_WP: {original_c_wp:.4f} → {explorer.C_WP:.4f}")


    # Test 3: Bow thruster resistance modeling
    print(f"\n--- Test 3: Bow Thruster Resistance Addition ---")
    print("Adding bow thruster parameters to model tunnel resistance...")

    explorer.BT_n = 2  # Number of bow thrusters
    explorer.BT_d = 3.5  # Thruster diameter in meters
    explorer.C_BT = 0.008  # Bow thruster resistance coefficient

    print(f"Bow Thruster Configuration:")
    print(f"  Number of thrusters: {explorer.BT_n}")
    print(f"  Thruster diameter: {explorer.BT_d} m")
    print(f"  Resistance coefficient: {explorer.C_BT}")
    print(f"  Design Impact: Bow thrusters add parasitic resistance")

    # Test 4: Direct parameter override demonstration
    print(f"\n--- Test 4: Direct Parameter Override ---")
    original_s = explorer.S

    print(f"Original wetted surface: {original_s:.1f} m²")
    print("Manually increasing wetted surface by 10% (e.g., fouling simulation)...")

    explorer.S = original_s * 1.1  # Increase by 10%
    print(f"Modified wetted surface: {explorer.S:.1f} m²")


    # Test 5: Clearing override to return to calculated value
    print(f"\n--- Test 5: Clearing Parameter Override ---")
    print("Returning wetted surface to calculated value...")

    explorer.clear_direct_value('S')
    print(f"Restored wetted surface: {explorer.S:.1f} m²")
    print("The explorer automatically recalculates the parameter from base equations")

    print(f"\n📊 Extended Parameter Test Summary:")
    print(f"   • Successfully demonstrated speed-dependent calculations")
    print(f"   • Hull form modifications affect multiple dependent parameters")
    print(f"   • Appendage modeling provides realistic resistance estimates")
    print(f"   • Direct parameter overrides allow for design modifications")
    print(f"   • Parameter clearing restores calculated values seamlessly")


def correction_library_test(shipbasic: 'ShipShapeBasic'):
    """
    Test ShipShapeExplorer with correction library functionality.

    This advanced example demonstrates:
    - Creating comprehensive correction factor libraries
    - Different correction types (offset, factor, function)
    - Conditional corrections based on ship parameters
    - Comparison between baseline and corrected calculations
    - Speed-dependent and condition-dependent corrections

    Args:
        shipbasic: A ShipShapeBasic instance to use as foundation
    """
    print_section_header("Advanced Correction Library System")

    print("The correction library system allows fine-tuning of calculated parameters")
    print("based on empirical data, model test results, or design modifications.")
    print("\nCreating comprehensive correction library with multiple correction types...")

    # Define a comprehensive correction library
    correction_library = {
        'A_m': CorrectionFactor(
            parameter='A_m',
            correction_type='offset',
            value=-2.5,  # Reduce midship area by 2.5 m²
            conditions=None
        ),
        'lcb': CorrectionFactor(
            parameter='lcb',
            correction_type='function',
            value=lambda base, params: base + 0.02 if params['V'] > 20 else base,
            conditions=None
        ),
        'S': CorrectionFactor(
            parameter='S',
            correction_type='factor',
            value=1.05,  # 5% increase in wetted surface
            conditions=None
        ),
        'C_WP': CorrectionFactor(
            parameter='C_WP',
            correction_type='function',
            value=lambda base, params: min(base * 1.02, 0.95),  # Cap at 0.95
            conditions=None
        ),
        'T_F': CorrectionFactor(
            parameter='T_F',
            correction_type='offset',
            value=0.5,  # Add 0.5m to forward draft
            conditions={'shiptype': shipbasic.shiptype}  # Only for this ship type
        ),
        'A_BT': CorrectionFactor(
            parameter='A_BT',
            correction_type='factor',
            value=1.15,  # 15% larger bulb area
            conditions={'T': shipbasic.T}  # Only when draft matches
        )
    }

    print(f"\n📋 Correction Library Contents ({len(correction_library)} corrections):")
    for param, correction in correction_library.items():
        conditions_str = f" (conditions: {correction.conditions})" if correction.conditions else ""
        print(f"   {param}: {correction.correction_type} = {correction.value}{conditions_str}")

    # Create baseline explorer (without corrections)
    print(f"\n--- Baseline Configuration (No Corrections) ---")
    explorer_baseline = ShipShapeExplorer(
        shipshape=shipbasic,
        v_knots=18,
        app_types=[2, 9],  # Rudder behind stern + stabilizer fins
        trim=1.8,
        verbose=False  # Reduce logging for cleaner comparison
    )

    # Collect baseline values for comparison
    baseline_values = {
        'A_m': explorer_baseline.A_m,
        'lcb': explorer_baseline.lcb,
        'S': explorer_baseline.S,
        'C_WP': explorer_baseline.C_WP,
        'T_F': explorer_baseline.T_F,
        'A_BT': explorer_baseline.A_BT
    }

    print("Baseline parameter values:")
    for param, value in baseline_values.items():
        print(f"   {param}: {value:.4f}")

    # Create corrected explorer
    print(f"\n--- Corrected Configuration (With Correction Library) ---")
    explorer_corrected = ShipShapeExplorer(
        shipshape=shipbasic,
        v_knots=18,
        correction_library=correction_library,  # Apply correction library
        app_types=[2, 9],  # Same appendages as baseline
        trim=1.8,
        verbose=True  # Enable verbose to see correction messages
    )

    print("\nCorrected parameter values and changes:")
    corrected_values = {
        'A_m': explorer_corrected.A_m,
        'lcb': explorer_corrected.lcb,
        'S': explorer_corrected.S,
        'C_WP': explorer_corrected.C_WP,
        'T_F': explorer_corrected.T_F,
        'A_BT': explorer_corrected.A_BT
    }

    for param, value in corrected_values.items():
        change = value - baseline_values[param]
        percent_change = (change / baseline_values[param]) * 100 if baseline_values[param] != 0 else 0
        print(f"   {param}: {value:.4f} (Δ{change:+.4f}, {percent_change:+.2f}%)")

    # Test speed-dependent correction (lcb function)
    print(f"\n--- Speed-Dependent LCB Correction Analysis ---")
    print("Testing lcb correction function: adds +0.02 when speed > 20 knots")
    print(f"{'Speed (knots)':<15} {'Baseline lcb':<15} {'Corrected lcb':<15} {'Correction':<12}")
    print("-" * 60)

    # Temporarily disable verbose logging for cleaner speed testing output
    explorer_corrected.verbose = False
    explorer_corrected.logger.setLevel(logging.WARNING)

    for speed in [15, 18, 20, 22, 25]:
        # Create temporary baseline explorer for comparison
        baseline_explorer_temp = ShipShapeExplorer(
            shipshape=shipbasic,
            v_knots=speed,
            verbose=False
        )
        baseline_lcb = baseline_explorer_temp.lcb

        # Test corrected explorer at this speed
        explorer_corrected.v_knots = speed
        corrected_lcb = explorer_corrected.lcb

        correction_applied = "Applied" if speed > 20 else "Not Applied"
        print(f"{speed:<15.0f} {baseline_lcb:<15.4f} {corrected_lcb:<15.4f} {correction_applied:<12}")

    # Test conditional corrections
    print(f"\n--- Conditional Correction Testing ---")
    print("T_F and A_BT corrections have ship-specific conditions...")

    # Reset to original state for final testing
    explorer_corrected.v_knots = 18
    explorer_corrected.verbose = True
    explorer_corrected.logger.setLevel(logging.INFO)

    # Force recalculation to demonstrate conditional application
    explorer_corrected._ship_record = None
    t_f_corrected = explorer_corrected.T_F

    explorer_corrected._ship_record = None
    a_bt_corrected = explorer_corrected.A_BT

    print(f"T_F correction (shiptype condition): {baseline_values['T_F']:.2f} → {t_f_corrected:.2f}")
    print(f"A_BT correction (draft condition): {baseline_values['A_BT']:.2f} → {a_bt_corrected:.2f}")

    # Test failed conditions
    print(f"\n--- Failed Condition Testing ---")
    print("Testing correction with impossible condition (T = 999.0 m)...")

    failed_correction_library = {
        'S': CorrectionFactor(
            parameter='S',
            correction_type='factor',
            value=2.0,  # Would double the surface area
            conditions={'T': 999.0}  # Impossible draft condition
        )
    }

    explorer_failed = ShipShapeExplorer(
        shipshape=shipbasic,
        v_knots=18,
        correction_library=failed_correction_library,
        verbose=True
    )

    print(f"Surface area with impossible condition: {explorer_failed.S:.1f} m²")
    print(f"Should match baseline (condition not met): {baseline_values['S']:.1f} m²")

    # Test correction persistence
    print(f"\n--- Correction Persistence Testing ---")
    print("Verifying corrections persist through parameter changes...")

    explorer_corrected.verbose = False
    explorer_corrected.logger.setLevel(logging.WARNING)

    explorer_corrected.trim = 3.0  # Change trim
    explorer_corrected.v_knots = 24  # Speed > 20 for lcb correction

    print(f"After changing trim and speed:")
    print(f"   A_m: {explorer_corrected.A_m:.4f} (baseline + offset)")
    print(f"   S: {explorer_corrected.S:.1f} m² (baseline × factor)")
    print(f"   lcb: {explorer_corrected.lcb:.4f} (baseline + speed function)")

    print(f"\n📊 Correction Library Test Summary:")
    print(f"   • Multiple correction types (offset, factor, function) work correctly")
    print(f"   • Conditional corrections apply only when conditions are met")
    print(f"   • Speed-dependent corrections enable dynamic performance tuning")
    print(f"   • Failed conditions gracefully fall back to baseline calculations")
    print(f"   • Corrections persist through parameter modifications")
    print(f"   • System provides comprehensive parameter adjustment capabilities")

def main():
    """
    Main demonstration function orchestrating all ShipShapeExplorer examples.

    This function runs a comprehensive demonstration of ShipShapeExplorer
    capabilities, showing progression from basic ship creation through
    advanced correction systems.
    """
    print_section_header("ShipShapeExplorer Class Demonstration", 80)
    print("This script demonstrates the advanced ship performance analysis capabilities")
    print("of the ShipShapeExplorer class, which extends ShipShapeBasic with:")
    print("\n  • Dimensionless number calculations (Froude, Reynolds)")
    print("  • Form factor and resistance coefficient modeling")
    print("  • Appendage resistance analysis")
    print("  • Hull form variation studies")
    print("  • Advanced correction factor systems")
    print("  • Dynamic parameter modification with real-time updates")

    try:
        # Reduce logging verbosity for cleaner demonstration output
        logging.getLogger().setLevel(logging.WARNING)

        # Step 1: Create foundation ship using ShipShapeBasic
        shipbasic = get_shipbasic_example()

        # Step 2: Demonstrate basic explorer functionality
        basic_explorer_test(shipbasic=shipbasic)

        # Step 3: Advanced parameter testing with comprehensive configuration
        extended_parameter_test(shipbasic=shipbasic)

        # Step 4: Correction library system demonstration
        correction_library_test(shipbasic=shipbasic)

        # Conclusion
        print_section_header("ShipShapeExplorer Demonstration Complete", 80)

    except Exception as e:
        print(f"\n❌ Demonstration failed: {str(e)}")
        print("Please ensure all required modules and data files are available.")
        print("Check that ShipShapeBasic, ShipShapeExplorer, and CorrectionFactor classes are imported correctly.")
        raise


if __name__ == '__main__':
    main()


