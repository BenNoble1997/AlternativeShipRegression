import numpy as np
from multivariable_fractional_polynomials_class import MFPRegressor

def run_simple_example():
    """Simple test with synthetic data."""
    print("=" * 80)
    print("TEST 1: Simple Example with 3 Variables")
    print("=" * 80)

    np.random.seed(42)

    # Generate data: y = 5*sqrt(x1) + 2*x2^2 + 0.1*x3 + noise
    n = 200
    x1 = np.random.uniform(10, 100, n)
    x2 = np.random.uniform(1, 10, n)
    x3 = np.random.uniform(0, 50, n)

    y_true = 5 * np.sqrt(x1) + 2 * x2 ** 2 + 0.1 * x3
    y = y_true + np.random.normal(0, 5, n)

    X = np.column_stack([x1, x2, x3])

    # Fit MFP
    mfp = MFPRegressor(alpha1=0.05, alpha2=0.05, verbose=True)
    mfp.fit(X, y, feature_names=['x1', 'x2', 'x3'])

    # Predict
    y_pred = mfp.predict(X)

    # Metrics
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))

    print(f"\nR²: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")

    return mfp


def run_ship_data_example():
    """Test with ship-like data."""
    print("\n" + "=" * 80)
    print("TEST 2: Ship Design Example")
    print("=" * 80)

    np.random.seed(42)

    # Simulate ship data
    n = 150
    DWT = np.random.uniform(20000, 100000, n)
    LOA = np.random.uniform(150, 300, n)
    Speed = np.random.uniform(12, 20, n)
    Beam = np.random.uniform(20, 40, n)

    # True relationship: Power depends on DWT^(2/3), Speed^3, LOA (linear)
    # Beam is noise variable
    Main_Power = (
            1000 +
            0.5 * DWT ** (2 / 3) +
            50 * Speed ** 3 +
            10 * LOA +
            0.1 * Beam +  # Weak effect
            np.random.normal(0, 500, n)
    )

    X = np.column_stack([DWT, LOA, Speed, Beam])

    # Fit MFP
    mfp = MFPRegressor(alpha1=0.05, alpha2=0.05, verbose=True)
    mfp.fit(X, Main_Power, feature_names=['DWT', 'LOA', 'Speed', 'Beam'])

    # Predictions
    y_pred = mfp.predict(X)

    # Metrics
    r2 = 1 - np.sum((Main_Power - y_pred) ** 2) / np.sum((Main_Power - Main_Power.mean()) ** 2)
    rmse = np.sqrt(np.mean((Main_Power - y_pred) ** 2))

    print(f"\nR²: {r2:.4f}")
    print(f"RMSE: {rmse:.1f} kW")

    return mfp

def model_factory():

    from parametric_model_factory import ModelFactory
    import numpy as np

    # Create factory
    factory = ModelFactory()

    # Check MFP is registered
    print("Available models:", factory.get_available_models())
    # Should include 'mfp'

    # Test applicability
    X = np.random.randn(100, 3)
    y = np.random.randn(100)

    is_applicable, reason = factory.is_model_applicable('mfp', X, y)
    print(f"MFP applicable: {is_applicable}")
    print(f"Reason: {reason}")

    # Test fitting
    if is_applicable:
        mfp_model, _ = factory.create_and_fit('mfp', X, y, verbose=True)

        # Test parameter counting
        n_params_before = factory.get_n_parameters('mfp', X)  # Maximum
        n_params_after = factory.get_n_parameters('mfp', X, fitted_model=mfp_model)  # Actual

        print(f"\nParameter count:")
        print(f"  Before fitting (max): {n_params_before}")
        print(f"  After fitting (actual): {n_params_after}")

        # Test predictions
        y_pred = factory.predict(mfp_model, None, X, 'mfp')
        print(f"\nPredictions shape: {y_pred.shape}")


def mfp_with_signal():
    """Test MFP with data that has real relationships."""
    from parametric_model_factory import ModelFactory
    import numpy as np

    np.random.seed(42)

    # Create factory
    factory = ModelFactory()

    # Generate data with REAL relationships
    n = 200
    x1 = np.random.uniform(1, 10, n)  # Positive for transformations
    x2 = np.random.uniform(1, 10, n)
    x3 = np.random.uniform(1, 10, n)

    # True model: y = 2 + 3*sqrt(x1) - 0.5*log(x2) + noise
    # x3 is irrelevant (should be eliminated)
    y = 2 + 3 * np.sqrt(x1) - 0.5 * np.log(x2) + np.random.normal(0, 0.5, n)

    X = np.column_stack([x1, x2, x3])

    print("=" * 80)
    print("TEST: MFP with Real Signal")
    print("=" * 80)
    print(f"True model: y = 2 + 3*√x1 - 0.5*ln(x2) + noise")
    print(f"x3 is noise (should be eliminated)")
    print()

    # Fit MFP
    mfp_model, _ = factory.create_and_fit('mfp', X, y, verbose=True)

    # Check parameter counting
    n_params_before = factory.get_n_parameters('mfp', X)
    n_params_after = factory.get_n_parameters('mfp', X, fitted_model=mfp_model)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Parameter count:")
    print(f"  Before fitting (max): {n_params_before}")
    print(f"  After fitting (actual): {n_params_after}")

    # Show selected transformations
    print(f"\nSelected Transformations:")
    for var_name, transform in mfp_model.get_selected_transformations().items():
        print(f"  {var_name}: {transform}")

    # Show coefficients
    print(f"\nFitted Coefficients:")
    print(f"  Intercept: {mfp_model.intercept_:.3f} (true: 2.0)")
    for i, coef in enumerate(mfp_model.coef_):
        print(f"  β{i + 1}: {coef:.3f}")

    # Test predictions
    y_pred = factory.predict(mfp_model, None, X, 'mfp')

    # Calculate R²
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    print(f"\nModel Performance:")
    print(f"  R² = {r2:.4f}")
    print(f"  RMSE = {np.sqrt(np.mean((y - y_pred) ** 2)):.4f}")

    print("\n" + "=" * 80)
    print("EXPECTED BEHAVIOR:")
    print("=" * 80)
    print("✓ x1 should be transformed to √x1 (power 0.5)")
    print("✓ x2 should be transformed to ln(x2) (power 0)")
    print("✓ x3 should be EXCLUDED (not significant)")
    print("✓ Coefficients should be close to true values")
    print("=" * 80)




if __name__ == "__main__":

    mfp_with_signal()
