import pandas as pd
import numpy as np

# Define the Rinauro (2024) polynomial regression formulas
# Each formula is a multi-linear regression with polynomial and interaction terms

formulas_data = []


# Helper function to create formula entry
def add_formula(name, target, predictors, coefficients_dict, reference="Rinauro et al. (2024)"):
    """
    Add a formula to the dataset.

    Parameters:
    - name: Formula identifier
    - target: Target variable
    - predictors: List of predictor variables
    - coefficients_dict: Dict mapping term -> coefficient value
    - reference: Citation
    """
    # Extract intercept
    intercept = coefficients_dict.pop('intercept', 0.0)

    # Get terms and coefficients in order
    terms = list(coefficients_dict.keys())
    coefficients = list(coefficients_dict.values())

    # Create row for Excel
    row = {
        'Predictors': str(predictors),  # Store as string representation of list
        'Target': target,
        'formula_type': 'multi_linear',
        'reference': reference,
        'bounds': 5.0,  # Default 5th-95th percentile
        'constant': intercept,  # Intercept term
        'terms': str(terms),  # Store as string representation of list
    }

    # Add coefficient columns (a1, a2, a3, ...)
    for i, (term, coef) in enumerate(zip(terms, coefficients), start=1):
        row[f'a{i}'] = coef

    formulas_data.append(row)


# ============================================================================
# MR1: Y = f(V, TEU)
# ============================================================================

# MR1 - L = f(V, TEU)
add_formula(
    name='Rinauro_MR1_L',
    target='L',
    predictors=['V', 'TEU'],
    coefficients_dict={
        'intercept': -7020.6859,
        'V': 1476.2432,
        'TEU': 0.1056,
        'V^2': -113.4121,
        'TEU*V': -0.0072,
        'TEU^2': -3.26e-07,
        'V^3': 3.8307,
        'V^2*TEU': 0.0001,
        'V^4': -0.0477
    }
)

# MR1 - B = f(V, TEU)
add_formula(
    name='Rinauro_MR1_B',
    target='B',
    predictors=['V', 'TEU'],
    coefficients_dict={
        'intercept': -380.6165,
        'V': 56.2597,
        'TEU': 0.0020,
        'V^2': -2.5729,
        'TEU*V': 2.86e-05,
        'TEU^2': -4.40e-08,
        'V^3': 0.0387
    }
)

# MR1 - D = f(V, TEU)
add_formula(
    name='Rinauro_MR1_D',
    target='D',
    predictors=['V', 'TEU'],
    coefficients_dict={
        'intercept': -316.7987,
        'V': 42.6892,
        'TEU': 0.1637,
        'V^2': -1.8414,
        'TEU*V': -0.0206,
        'TEU^2': -3.86e-08,
        'V^3': 0.0265,
        'V^2*TEU': 0.0009,
        'V^3*TEU': -1.22e-05
    }
)

# MR1 - T = f(V, TEU)
add_formula(
    name='Rinauro_MR1_T',
    target='T',
    predictors=['V', 'TEU'],
    coefficients_dict={
        'intercept': -9.0068,
        'V': 1.5253,
        'TEU': 0.0004,
        'V^2': -0.0303,
        'TEU^2': -4.8235e-09
    }
)

# MR1 - PENG = f(V, TEU)
add_formula(
    name='Rinauro_MR1_PENG',
    target='PENG',
    predictors=['V', 'TEU'],
    coefficients_dict={
        'intercept': -3826801.116,
        'V': 754690.6349,
        'TEU': 4.8726,
        'V^2': -55037.79937,
        'TEU^2': -0.0001,
        'V^3': 1758.773976,
        'V^4': -20.7193
    }
)

# ============================================================================
# MR2: Y = f(L, D, TEU)
# ============================================================================

# MR2 - V = f(L, D, TEU)
add_formula(
    name='Rinauro_MR2_V',
    target='V',
    predictors=['L', 'D', 'TEU'],
    coefficients_dict={
        'intercept': 24.56265605,
        'L': -0.540577487,
        'D': 4.111193311,
        'TEU': 0.010813276,
        'L^2': 0.003648354,
        'L*D': -0.005247444,
        'D^2': -0.312480065,
        'L*TEU': -0.000275754,
        'D*TEU': 0.002420124,
        'TEU^2': -2.97e-07,
        'L^2*D': -0.000194496,
        'L*D^2': 0.002162734,
        'L^2*TEU': 5.72e-07,
        'L*D*TEU': 4.50e-06,
        'D^2*TEU': -0.000109895,
        'D*TEU^2': 9.85e-09,
        'L^2*D*TEU': -2.05e-08,
        'L*D^2*TEU': 2.19e-07
    }
)

# MR2 - PENG = f(L, D, TEU)
add_formula(
    name='Rinauro_MR2_PENG',
    target='PENG',
    predictors=['L', 'D', 'TEU'],
    coefficients_dict={
        'intercept': -51214.97917,
        'L': -174.4098865,
        'D': 18961.94925,
        'TEU': -53.94155239,
        'L^2': 15.30319721,
        'L*D': -384.3496449,
        'D^2': 1008.661712,
        'L*TEU': 0.261,
        'D*TEU': 3.498457681,
        'TEU^2': -0.003742477,
        'L^2*D': -0.983,
        'L*D^2': 24.3,
        'D^3': -114.4079426,
        'L^2*TEU': 0.000546,
        'L*D*TEU': -0.0211,
        'D*TEU^2': 0.000134
    }
)

# ============================================================================
# MR3: Y = f(L, T, TEU)
# ============================================================================

# MR3 - V = f(L, T, TEU)
add_formula(
    name='Rinauro_MR3_V',
    target='V',
    predictors=['L', 'T', 'TEU'],
    coefficients_dict={
        'intercept': -58.8249432,
        'L': -0.959180192,
        'T': 41.75581086,
        'TEU': 0.00166132,
        'L^2': 0.004048978,
        'L*T': 0.071119467,
        'T^2': -5.38304808,
        'L*TEU': -0.000354577,
        'T*TEU': 0.007613745,
        'TEU^2': -6.67e-07,
        'L^2*T': -0.000315294,
        'T^3': 0.208585136,
        'L^2*TEU': 3.05e-07,
        'L*T*TEU': 3.02e-05,
        'T^2*TEU': -0.000757271,
        'T*TEU^2': 3.82e-08,
        'L^2*T*TEU': -2.67e-08,
        'T^3*TEU': 9.94e-06
    }
)

# MR3 - PENG = f(L, T, TEU)
add_formula(
    name='Rinauro_MR3_PENG',
    target='PENG',
    predictors=['L', 'T', 'TEU'],
    coefficients_dict={
        'intercept': -300595.5842,
        'L': 3818.39297,
        'T': 11491.62815,
        'TEU': -150.2089654,
        'L^2': -67.06777786,
        'L*T': 1476.089831,
        'T^2': -7451.858834,
        'L*TEU': -0.262639218,
        'T*TEU': 34.66496327,
        'TEU^2': -0.006746098,
        'L^3': -0.098690935,
        'L^2*T': 17.96549814,
        'L*T^2': -464.1148458,
        'T^3': 2490.556459,
        'L^2*TEU': -0.000935618,
        'L*T*TEU': 0.074302192,
        'T^2*TEU': -2.402734551,
        'T*TEU^2': 0.000411303,
        'L^4': 0.00059148,
        'L^3*T^4': -0.044457043,
        'L^2*T^2': 0.725949793
    }
)

# ============================================================================
# MR4: Y = f(L, B, D, TEU)
# ============================================================================

# MR4 - V = f(L, B, D, TEU)
add_formula(
    name='Rinauro_MR4_V',
    target='V',
    predictors=['L', 'B', 'D', 'TEU'],
    coefficients_dict={
        'intercept': -123.6176,
        'L': 0.4986,
        'B': 17.2011,
        'D': 3.8256,
        'TEU': 0.0383,
        'L^2': 0.0066,
        'L*B': -0.1919,
        'B^2': -0.6788,
        'L*D': 0.0226,
        'B*D': 0.9622,
        'D^2': -1.1840,
        'L*TEU': -9.84e-05,
        'B*TEU': -0.0008,
        'D*TEU': 0.0001,
        'TEU^2': 2.42e-06,
        'L^3': -1.01e-05,
        'L*B^2': 0.0057,
        'B^3': 0.0039,
        'L*D^2': -0.0003,
        'B*D^2': -0.0168,
        'D^3': 0.0253,
        'L^2*TEU': 3.82e-07,
        'B^2*TEU': 2.82e-05,
        'L*D*TEU': -1.15e-05,
        'B*D*TEU': -0.0001,
        'D^2*TEU': 0.0002,
        'B*TEU^2': -5.04e-08,
        'L*B^3': -5.28e-05,
        'B^3*TEU': 2.15e-07,
        'L*D^2*TEU': 2.00e-07,
        'B*D^2*TEU': 2.56e-06,
        'D^3*TEU': -4.80e-06
    }
)

# MR4 - PENG = f(L, B, D, TEU)
add_formula(
    name='Rinauro_MR4_PENG',
    target='PENG',
    predictors=['L', 'B', 'D', 'TEU'],
    coefficients_dict={
        'intercept': -1128750.7849,
        'L': 9643.1847,
        'B': 116584.3452,
        'D': -19552.1683,
        'TEU': 66.9804,
        'L^2': 29.1442,
        'L*B': -1309.8360,
        'B^2': -3268.4354,
        'L*D': -195.6562,
        'B*D': 3652.5626,
        'D^2': -867.6640,
        'L*TEU': 0.8854,
        'B*TEU': -4.7628,
        'D*TEU': -0.6786,
        'TEU^2': 0.0136,
        'L^3': -0.0520,
        'L*B^2': 33.4326,
        'B^3': 20.3311,
        'L*B*D': 12.8520,
        'B*D^2': -147.0299,
        'D^3': 60.3298,
        'L^2*TEU': 0.0026,
        'L*B*TEU': -0.0357,
        'B^2*TEU': 0.1593,
        'L*D*TEU': -0.0967,
        'B*D*TEU': -0.4317,
        'D^2*TEU': 0.9906,
        'B*TEU^2': -0.0003,
        'L*B^3': -0.3174,
        'B^3*TEU': 0.0014,
        'L*B*D*TEU': 0.0012,
        'D^3*TEU': -0.0087
    }
)

# ============================================================================
# CREATE DATAFRAME AND EXPORT
# ============================================================================

# Create DataFrame
df = pd.DataFrame(formulas_data)

# Reorder columns: metadata first, then coefficients
metadata_cols = ['Predictors', 'Target', 'formula_type', 'reference', 'bounds', 'constant', 'terms']
coef_cols = [col for col in df.columns if col.startswith('a') and col not in metadata_cols]
coef_cols_sorted = sorted(coef_cols, key=lambda x: int(x[1:]))  # Sort a1, a2, ..., a10, a11, ...

column_order = metadata_cols + coef_cols_sorted
df = df[column_order]

# Export to Excel
output_file = 'rinauro_2024_multilinear_formulas.xlsx'
df.to_excel(output_file, index=False, sheet_name='Container')

print(f"✓ Created {output_file} with {len(formulas_data)} formulas")
print(f"\nFormula Summary:")
print(f"  MR1 (V, TEU → L, B, D, T, PENG): 5 formulas")
print(f"  MR2 (L, D, TEU → V, PENG): 2 formulas")
print(f"  MR3 (L, T, TEU → V, PENG): 2 formulas")
print(f"  MR4 (L, B, D, TEU → V, PENG): 2 formulas")
print(f"\nColumn structure:")
print(f"  - Predictors: List of predictor variables")
print(f"  - Target: Target variable")
print(f"  - formula_type: 'multi_linear'")
print(f"  - constant: Intercept term")
print(f"  - terms: List of polynomial terms")
print(f"  - a1, a2, ...: Coefficients for each term")
print(f"\nExample formula (MR1_L):")
print(f"  L = {formulas_data[0]['constant']:.2f}")
for i, term in enumerate(eval(formulas_data[0]['terms']), start=1):
    coef = formulas_data[0].get(f'a{i}')
    if coef is not None:
        print(f"      {'+' if coef >= 0 else ''} {coef:.4e} * {term}")