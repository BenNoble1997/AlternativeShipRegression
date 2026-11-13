import pandas as pd
import numpy as np

# ============================================================================
# BULK CARRIER FORMULAE
# ============================================================================

bulk_carrier_data = []

# Power law formulae: b + a*DWT^c
bulk_power = [
    {'Predictor': 'DWT', 'Target': 'Disp', 'a': 2.214420, 'b': 0, 'c': 0.943855},
    {'Predictor': 'DWT', 'Target': 'LBP (M)', 'a': 7.603010, 'b': 0, 'c': 0.300155},
    {'Predictor': 'DWT', 'Target': 'BEAM MLD (M)', 'a': 1.055900, 'b': 0, 'c': 0.309724},
    {'Predictor': 'DWT', 'Target': 'DEPTH (M)', 'a': 0.584268, 'b': 0, 'c': 0.310795},
    {'Predictor': 'DWT', 'Target': 'DRAFT (M)', 'a': 0.480719, 'b': 0, 'c': 0.298295},
    {'Predictor': 'DWT', 'Target': 'L*B*D', 'a': 1.776440, 'b': 9295.650000, 'c': 1.000000},
    {'Predictor': 'DWT', 'Target': 'LSW', 'a': 6.513400, 'b': 0, 'c': 0.678895},
    {'Predictor': 'DWT', 'Target': 'C_B', 'a': 0.515788, 'b': 0, 'c': 0.042626},
    {'Predictor': 'DRAFT (M)', 'Target': 'DWT', 'a': 16.564000, 'b': 0, 'c': 3.219000},
]

for item in bulk_power:
    bulk_carrier_data.append({
        'Predictor': item['Predictor'],
        'Target': item['Target'],
        'formula_type': 'power',
        'a': item['a'],
        'b': item['b'],
        'c': item['c'],
        'd': '-',
        'e': '-',
        'bounds': '-',
        'reference': 'Bulk Carrier Design Formulae'
    })

# Power-log formula: b + a*log(DWT^c)
bulk_carrier_data.append({
    'Predictor': 'DWT',
    'Target': 'LSW/Disp',
    'formula_type': 'power_log',
    'a': -0.044922,
    'b': 0,
    'c': 0.656409,
    'd': '-',
    'e': '-',
    'bounds': '-',
    'reference': 'Bulk Carrier Design Formulae'
})

# Cubic polynomial formulae: a*x³ + b*x² + c*x + d
bulk_cubic = [
    {'Predictor': 'LOA (M)', 'Target': 'BEAM MLD (M)', 'a': 0.00, 'b': -0.00, 'c': 0.28, 'd': 4.06},
    {'Predictor': 'LOA (M)', 'Target': 'Beam/Draft', 'a': -0.00, 'b': 0.00, 'c': -0.01, 'd': 3.26},
    {'Predictor': 'LOA (M)', 'Target': 'DWT', 'a': 0.01, 'b': -2.91, 'c': 296.33, 'd': 8038.60},
    {'Predictor': 'BEAM MLD (M)', 'Target': 'DWT', 'a': -0.86, 'b': 234.38, 'c': 6693.40, 'd': 61073.00},
]

for item in bulk_cubic:
    bulk_carrier_data.append({
        'Predictor': item['Predictor'],
        'Target': item['Target'],
        'formula_type': 'cubic_polynomial',
        'a': item['a'],
        'b': item['b'],
        'c': item['c'],
        'd': item['d'],
        'e': '-',
        'bounds': '-',
        'reference': 'Bulk Carrier Design Formulae'
    })

bulk_df = pd.DataFrame(bulk_carrier_data)

# ============================================================================
# CONTAINER SHIP FORMULAE
# ============================================================================

container_data = []

# Power law formulae: b + a*DWT^c
container_power = [
    {'Predictor': 'DWT', 'Target': 'Disp', 'a': 1.779550, 'b': 0, 'c': 0.975578},
    {'Predictor': 'DWT', 'Target': 'LBP (M)', 'a': 3.541320, 'b': 0, 'c': 0.388442},
    {'Predictor': 'DWT', 'Target': 'BEAM MLD (M)', 'a': 1.552190, 'b': 0, 'c': 0.284381},
    {'Predictor': 'DWT', 'Target': 'DEPTH (M)', 'a': 0.299394, 'b': 0, 'c': 0.389020},
    {'Predictor': 'DWT', 'Target': 'LBP*B*D', 'a': 1.648980, 'b': 0, 'c': 1.061690},
    {'Predictor': 'DWT', 'Target': 'LSW', 'a': 0.973087, 'b': 0, 'c': 0.912420},
    {'Predictor': 'DWT', 'Target': 'LSW/Disp', 'a': 0.519596, 'b': 0, 'c': -0.062243},
    {'Predictor': 'DWT', 'Target': 'C_B', 'a': 0.908683, 'b': 0, 'c': -0.032042},
    {'Predictor': 'LOA (M)', 'Target': 'BEAM MLD (M)', 'a': 0.389900, 'b': 0, 'c': 1.0},  # Linear: Beam = 0.3899 * LOA
    {'Predictor': 'LOA (M)', 'Target': 'DWT', 'a': 0.029000, 'b': 0, 'c': 2.605500},
    {'Predictor': 'BEAM MLD (M)', 'Target': 'DWT', 'a': 1.052900, 'b': 0, 'c': 3.035800},
    {'Predictor': 'DRAFT (M)', 'Target': 'DWT', 'a': 520.070000, 'b': 0, 'c': 0.368900},
]

for item in container_power:
    container_data.append({
        'Predictor': item['Predictor'],
        'Target': item['Target'],
        'formula_type': 'power',
        'a': item['a'],
        'b': item['b'],
        'c': item['c'],
        'd': '-',
        'e': '-',
        'bounds': '-',
        'reference': 'Container Ship Design Formulae'
    })

# Power-log formula for Draft
container_data.append({
    'Predictor': 'DWT',
    'Target': 'DRAFT (M)',
    'formula_type': 'power_log',
    'a': 2.723380,
    'b': -17.158100,
    'c': 1.0,  # log(DWT^1) = log(DWT)
    'd': '-',
    'e': '-',
    'bounds': '-',
    'reference': 'Container Ship Design Formulae'
})

# Cubic polynomial formulae: a*LOA³ + b*LOA² + c*LOA + d
container_cubic = [
    {'Predictor': 'LOA (M)', 'Target': 'BEAM MLD (M)', 'a': 0, 'b': 0, 'c': 0, 'd': 0},  # Placeholder - incomplete data
    {'Predictor': 'LOA (M)', 'Target': 'Beam/Draft', 'a': -0.00, 'b': 0.0001, 'c': -0.03, 'd': 4.10},
    {'Predictor': 'LOA (M)', 'Target': 'DWT', 'a': 0, 'b': 0, 'c': 0, 'd': 0},  # Placeholder - incomplete data
    {'Predictor': 'BEAM MLD (M)', 'Target': 'DWT', 'a': 0, 'b': 0, 'c': 0, 'd': 0},  # Placeholder - incomplete data
    {'Predictor': 'DRAFT (M)', 'Target': 'DWT', 'a': 0, 'b': 0, 'c': 0, 'd': 0},  # Placeholder - incomplete data
]

# Only add complete cubic formulae
for item in container_cubic:
    if item['a'] != 0 or item['b'] != 0 or item['c'] != 0 or item['d'] != 0:
        container_data.append({
            'Predictor': item['Predictor'],
            'Target': item['Target'],
            'formula_type': 'cubic_polynomial',
            'a': item['a'],
            'b': item['b'],
            'c': item['c'],
            'd': item['d'],
            'e': '-',
            'bounds': '-',
            'reference': 'Container Ship Design Formulae'
        })

container_df = pd.DataFrame(container_data)

# ============================================================================
# TANKER FORMULAE
# ============================================================================

tanker_data = []

# Power law formulae: b + a*DWT^c
tanker_power = [
    {'Predictor': 'DWT', 'Target': 'Disp', 'a': 1.1213, 'b': 4773.95, 'c': 1.0},  # Linear with offset
    {'Predictor': 'DWT', 'Target': 'DWT/Disp', 'a': 0.541223, 'b': 0, 'c': 0.039034},
    {'Predictor': 'DWT', 'Target': 'LBP (M)', 'a': 6.864080, 'b': 0, 'c': 0.307233},
    {'Predictor': 'DWT', 'Target': 'BEAM MLD (M)', 'a': 0.962394, 'b': 0, 'c': 0.324898},
    {'Predictor': 'DWT', 'Target': 'DEPTH (M)', 'a': 0.459140, 'b': 0, 'c': 0.330623},
    {'Predictor': 'DWT', 'Target': 'DRAFT (M)', 'a': 0.450110, 'b': 0, 'c': 0.303134},
    {'Predictor': 'DWT', 'Target': 'LBP*B*D', 'a': 3.050490, 'b': 0, 'c': 0.962234},
    {'Predictor': 'DWT', 'Target': 'LSW', 'a': 2.918900, 'b': 0, 'c': 0.755480},
    {'Predictor': 'DWT', 'Target': 'LSW/Disp', 'a': 1.624330, 'b': 0, 'c': -0.207684},
]

for item in tanker_power:
    tanker_data.append({
        'Predictor': item['Predictor'],
        'Target': item['Target'],
        'formula_type': 'power',
        'a': item['a'],
        'b': item['b'],
        'c': item['c'],
        'd': '-',
        'e': '-',
        'bounds': '-',
        'reference': 'Tanker Design Formulae'
    })

# Note: The cubic polynomial data for tanker appears incomplete in your input
# Adding placeholders for reference
tanker_cubic_incomplete = [
    # {'Predictor': 'LOA (M)', 'Target': 'BEAM MLD (M)', ...},  # Missing coefficients
    # {'Predictor': 'LOA (M)', 'Target': 'Beam/Draft', ...},    # Missing coefficients
    # etc.
]

tanker_df = pd.DataFrame(tanker_data)

# ============================================================================
# CREATE EXCEL FILE WITH MULTIPLE SHEETS
# ============================================================================

# Create Excel writer object

output_file = r'C:\Users\bnoble\GitHub\ShipDesignDigitalTwin\knowledge_layer\literature_formulae_old.xlsx'

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    bulk_df.to_excel(writer, sheet_name='Bulk_Carrier', index=False)
    container_df.to_excel(writer, sheet_name='Container_Ship', index=False)
    tanker_df.to_excel(writer, sheet_name='Tanker', index=False)

print(f"✓ Created {output_file}")
print(f"\nSheets created:")
print(f"  - Bulk_Carrier: {len(bulk_df)} formulae")
print(f"  - Container_Ship: {len(container_df)} formulae")
print(f"  - Tanker: {len(tanker_df)} formulae")

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")

print("\nBulk Carrier formulae:")
print(bulk_df[['Predictor', 'Target', 'formula_type']].to_string(index=False))

print("\n\nContainer Ship formulae:")
print(container_df[['Predictor', 'Target', 'formula_type']].to_string(index=False))

print("\n\nTanker formulae:")
print(tanker_df[['Predictor', 'Target', 'formula_type']].to_string(index=False))

print(f"\n{'='*80}")
print("NOTES")
print(f"{'='*80}")
print("1. Some Container Ship and Tanker cubic polynomial formulae were incomplete")
print("   in the source data and have been omitted.")
print("2. You can add the 'bounds' column values manually in Excel if needed.")
print("3. Column names match your DataFrame columns (e.g., 'BEAM MLD (M)', 'DRAFT (M)')")
print("4. Use ShipType enum to load specific sheets:")
print("   formulae = LiteratureFormulaLibrary.from_file('literature_formulae.xlsx',")
print("                                                  shiptype=ShipType.BULK_CARRIER)")