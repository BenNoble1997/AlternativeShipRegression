"""
Ship Parameter Interpolation System - Demonstration Script
==========================================================

This script demonstrates the capabilities of the ShipShapeBasic class,
showing how to create ship instances, interpolate missing parameters,
and modify ship characteristics dynamically.

Key Features Demonstrated:
- Ship instantiation with minimal parameters
- Parameter interpolation and prediction
- Dynamic parameter modification
- Multiple ship type examples
- Batch processing for parametric studies
"""

import pandas as pd
from typing import List, Dict
import logging

from parameters.shipshape import ShipShapeBasic
from parameters.shipshape.ship_parameter_interpolation_class import ShipParameterInterpolator
from parameters.shipshape.ShipShapeBasic_class import ShipType


def print_separator(title: str = "", char: str = "=", width: int = 70):
    """
    Print a formatted separator line with optional title.

    Args:
        title: Optional title to display in the separator
        char: Character to use for the separator line
        width: Total width of the separator
    """
    if title:
        print(f"\n{char * width}")
        print(f" {title.upper()} ")
        print(f"{char * width}")
    else:
        print(f"{char * width}")


def print_ship_info(ship: 'ShipShapeBasic', description: str = "Ship Parameters"):
    """
    Print formatted ship parameter information.

    Args:
        ship: ShipShapeBasic instance
        description: Description of what this ship represents
    """
    print(f"\n📊 {description}")
    print("-" * 50)
    print(f"Ship Type: {ship.shiptype.name} (Key: {ship.shiptype.key})")
    print(f"Deadweight Tonnage (DWT): {ship.DWT:,.0f} tonnes")
    print(f"Length Overall (L): {ship.L:.2f} m")
    print(f"Length PP (L_PP): {ship.L_PP:.2f} m")
    print(f"Breadth (B): {ship.B:.2f} m")
    print(f"Draft (T): {ship.T:.2f} m")


def example_1_basic_ship_creation():
    """
    Example 1: Basic Ship Creation and Parameter Access

    This example demonstrates:
    - Creating a ship with minimal required parameters (L, DWT, shiptype)
    - How the interpolation system predicts missing parameters
    - Accessing interpolated parameters through properties
    """
    print_separator("Example 1: Basic Ship Creation")

    # File path to ship database (replace with your actual path)
    file_path = r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\CLARKSONS_27_05.xlsx'

    print("Creating a bulk carrier with basic parameters...")
    print("Input parameters:")
    print("  - Length Overall (L): 205 m")
    print("  - Deadweight Tonnage (DWT): 37,500 tonnes")
    print("  - Midship Coefficient (C_M): 0.98")
    print("  - Ship Type: Bulk Carrier")

    # Create the ship instance
    shipbasic = ShipShapeBasic(
        L=205,  # Length Overall in meters
        shiptype=ShipType.BULK_CARRIER,  # Ship type enumeration
        shipdata=file_path,  # Path to ship database
        DWT=37500,  # Deadweight tonnage in tonnes
        C_M=0.98  # Midship coefficient
    )

    # Display basic ship information
    print_ship_info(shipbasic, "Initial Ship Configuration")

    print(f"\n🔍 Key Results:")
    print(f"   Ship type key: {shipbasic.shiptype.key}")
    print(f"   Ship type name: {shipbasic.shiptype.name}")
    print(f"   Ship DWT: {shipbasic.DWT:,.0f} tonnes")
    print(f"   Interpolated L_PP: {shipbasic.L_PP:.2f} m")

    print(f"\n📝 Note: L_PP, B, and T were automatically interpolated from the database")
    print(f"    using regression analysis based on L, DWT, and ship type.")

    return shipbasic


def example_2_parametric_study():
    """
    Example 2: Parametric Study with Varying Ship Lengths

    This example demonstrates:
    - Creating multiple ships with different lengths
    - How interpolated parameters change with input variations
    - Batch processing for design studies
    """
    print_separator("Example 2: Parametric Study - Length Variation")

    file_path = r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\CLARKSONS_27_05.xlsx'

    # Define length variations to test
    Length = [200, 220, 240, 250]

    print("Testing different ship lengths with fixed DWT and C_M...")
    print("Fixed parameters:")
    print("  - DWT: 37,500 tonnes")
    print("  - C_M: 0.98")
    print("  - Ship Type: Bulk Carrier")
    print(f"  - Length variations: {Length} meters")

    print(f"\n{'Length (m)':<12} {'L_PP (m)':<10} {'B (m)':<8} {'T (m)':<8}")
    print("-" * 45)

    # Loop through each length value
    for length in Length:
        # Create ship with current length (verbose=False for cleaner output)
        shipbasic = ShipShapeBasic(
            L=length,
            shiptype=ShipType.BULK_CARRIER,
            shipdata=file_path,
            DWT=37500,
            C_M=0.98,
            verbose=False  # Suppress detailed logging for batch processing
        )

        # Display interpolated parameters in tabular format
        print(f"{length:<12.0f} {shipbasic.L_PP:<10.2f} {shipbasic.B:<8.2f} {shipbasic.T:<8.2f}")

def example_3_dynamic_modification(ship: 'ShipShapeBasic'):
    """
    Example 3: Dynamic Parameter Modification

    This example demonstrates:
    - Modifying ship parameters after creation
    - How dependent parameters automatically update
    - Real-time recalculation of derived parameters

    Args:
        ship: An existing ShipShapeBasic instance to modify
    """
    print_separator("Example 3: Dynamic Parameter Modification")

    print("Demonstrating dynamic parameter updates...")
    print(f"Original configuration:")
    print(f"  - Length Overall (L): {ship.L:.2f} m")
    print(f"  - Length PP (L_PP): {ship.L_PP:.2f} m")

    # Modify the ship's length
    new_length = 210
    print(f"\n🔧 Changing length from {ship.L:.2f}m to {new_length}m...")
    ship.L = new_length

    print(f"Updated configuration:")
    print(f"  - Length Overall (L): {ship.L:.2f} m")
    print(f"  - Length PP (L_PP): {ship.L_PP:.2f} m")



def example_4_different_input_combinations():
    """
    Example 4: Different Input Parameter Combinations

    This example demonstrates:
    - Various ways to specify ship parameters
    - How different input combinations affect interpolation
    - The flexibility of the parameter specification system
    """
    print_separator("Example 4: Different Input Parameter Combinations")

    file_path = r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\parameters\Databases\clarksons_data\CLARKSONS_27_05.xlsx'

    # Test Case 1: Basic parameters with known breadth
    print("Test Case 1: L + DWT + B specified")
    print("Input: L=205m, DWT=37,500t, B=30m, C_M=0.98")

    shipbasic3 = ShipShapeBasic(
        L=205,
        shiptype=ShipType.BULK_CARRIER,
        shipdata=file_path,
        DWT=37500,
        _B=30,  # Note: _B is used to specify breadth as input
        C_M=0.98
    )

    print(f"Result: Interpolated T = {shipbasic3.T:.2f} m")
    print(f"        (Draft calculated from L, DWT, and B)")

    # Test Case 2: Different DWT with same breadth
    print(f"\nTest Case 2: Smaller DWT with same breadth")
    print("Input: L=205m, DWT=35,000t, B=25m, C_M=0.98")

    shipbasic4 = ShipShapeBasic(
        L=205,
        shiptype=ShipType.BULK_CARRIER,
        shipdata=file_path,
        DWT=35000,  # Reduced DWT
        _B=25,  # Reduced breadth
        C_M=0.98
    )

    print(f"Result: Interpolated T = {shipbasic4.T:.2f} m")
    print(f"        (Smaller DWT requires less draft)")

    # Test Case 3: Even smaller ship
    print(f"\nTest Case 3: Even smaller ship")
    print("Input: L=205m, DWT=30,000t, B=25m, C_M=0.98")

    shipbasic5 = ShipShapeBasic(
        L=205,
        shiptype=ShipType.BULK_CARRIER,
        shipdata=file_path,
        DWT=30000,  # Further reduced DWT
        _B=25,
        C_M=0.98
    )

    print(f"Result: Interpolated T = {shipbasic5.T:.2f} m")

    # Test Case 4: Smaller overall ship
    print(f"\nTest Case 4: Smaller overall ship dimensions")
    print("Input: L=190m, DWT=30,000t, B=20m, C_M=0.98")

    shipbasic6 = ShipShapeBasic(
        L=190,  # Reduced length
        shiptype=ShipType.BULK_CARRIER,
        shipdata=file_path,
        DWT=30000,
        _B=20,  # Reduced breadth
        C_M=0.98
    )

    print(f"Result: Interpolated T = {shipbasic6.T:.2f} m")

    print(f"\n🔍 Summary of Results:")
    print(f"   Test 1 (L=205, DWT=37,500, B=30): T = {shipbasic3.T:.2f} m")
    print(f"   Test 2 (L=205, DWT=35,000, B=25): T = {shipbasic4.T:.2f} m")
    print(f"   Test 3 (L=205, DWT=30,000, B=25): T = {shipbasic5.T:.2f} m")
    print(f"   Test 4 (L=190, DWT=30,000, B=20): T = {shipbasic6.T:.2f} m")



def main():
    """
    Main function that runs all demonstration examples.

    This function orchestrates the execution of all examples in logical order,
    providing a comprehensive demonstration of ShipShapeBasic capabilities.
    """
    print_separator("ShipShapeBasic Class Demonstration", "=", 80)
    print("This script demonstrates the key functionality of the ShipShapeBasic class")
    print("using practical examples of ship parameter interpolation and modification.")
    print("\nThe examples cover:")
    print("  1. Basic ship creation with parameter interpolation")
    print("  2. Parametric studies with varying dimensions")
    print("  3. Dynamic parameter modification")
    print("  4. Different input parameter combinations")

    try:
        # Reduce logging verbosity for cleaner demo output
        logging.getLogger().setLevel(logging.WARNING)

        # Example 1: Basic ship creation and parameter access
        ship = example_1_basic_ship_creation()

        # Example 2: Parametric study with length variations
        example_2_parametric_study()

        # Example 3: Dynamic parameter modification
        example_3_dynamic_modification(ship)

        # Example 4: Different input parameter combinations
        example_4_different_input_combinations()

        # Conclusion
        print_separator("Demonstration Complete", "=", 80)
        print("✅ All examples completed successfully!")

        print(f"\n📋 Summary of Capabilities Demonstrated:")
        print(f"   • Ship creation with minimal parameters (L, DWT, shiptype)")
        print(f"   • Automatic interpolation of missing parameters (L_PP, B, T)")
        print(f"   • Dynamic parameter modification with automatic updates")
        print(f"   • Flexible input combinations for different design scenarios")
        print(f"   • Parametric studies for design optimization")

        print(f"\n🔧 Technical Notes:")
        print(f"   • Interpolation uses regression analysis on ship database")
        print(f"   • Parameters marked with '_' (e.g., _B) are input specifications")
        print(f"   • Ship database must contain required columns for interpolation")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {str(e)}")
        print("Please ensure all required modules and data files are available.")
        raise


if __name__ == '__main__':
    main()