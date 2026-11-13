# Ship Parameter Estimation - Formula Reference

This document provides a comprehensive reference for all parameters and formulas used in the ShipShapeBasic and ShipShapeExplorer classes for ship parameter estimation.

## Table of Contents

1. [Basic Input Parameters](#basic-input-parameters)
2. [Speed and Dimensionless Parameters](#speed-and-dimensionless-parameters)
3. [Coefficients](#coefficients)
4. [Volume and Displacement](#volume-and-displacement)
5. [Draft Parameters](#draft-parameters)
6. [Area Parameters](#area-parameters)
7. [Position Parameters](#position-parameters)
8. [Form Factors](#form-factors)
9. [Resistance Parameters](#resistance-parameters)
10. [Appendage Parameters](#appendage-parameters)
11. [Bow Thruster Parameters](#bow-thruster-parameters)

---

## Basic Input Parameters

### Required Input Parameters

| Parameter | Symbol | Unit | Description |
|-----------|--------|------|-------------|
| L | L | m | Length Overall |
| L_PP | L_PP | m | Length Between Perpendiculars |
| B | B | m | Breadth (Beam) |
| T | T | m | Draft |
| DWT | DWT | tonnes | Deadweight Tonnage |
| GT | GT | tonnes | Gross Tonnage |
| C_M | C_M | - | Midship Coefficient |
| v_knots | V | knots | Service Speed |

---

## Speed and Dimensionless Parameters

### Speed Conversion
**Formula:** `v_ms = v_knots × 0.5144`

Where:
- `v_ms` = Speed in m/s
- `v_knots` = Speed in knots

### Reynolds Number
**Formula:** `Re = (v_ms × L) / ν`

Where:
- `Re` = Reynolds number
- `v_ms` = Speed in m/s
- `L` = Length overall (m)
- `ν` = Kinematic viscosity of seawater

### Froude Number
**Formula:** `F_n = v_ms / √(g × L)`

Where:
- `F_n` = Froude number
- `v_ms` = Speed in m/s
- `g` = Gravitational acceleration (9.81 m/s²)
- `L` = Length overall (m)

---

## Coefficients

### Block Coefficient (C_B)

#### Method 1: Volume-based calculation
**Formula:** `C_B = DWT_v / (L × B × T)`

Where:
- `DWT_v` = Displacement volume (m³)
- `L`, `B`, `T` = Principal dimensions (m)

#### Method 2: Katsoulis formulation (when service speed is available)
**Formula:** `C_B = (V_s / (a₁ × L^a₃ × B^a₄ × T^a₅))^(1/a₂)`

Coefficients by ship type:
- **Bulk Carrier:** [3.7682, -0.3236, 0.3566, -0.0455, -0.1866]
- **Container Ship:** [0.9804, -0.9809, 0.7416, -0.3742, -0.025]
- **Oil Tanker:** [1.633, -0.3280, 0.7256, -0.5122, -0.0214]
- **Other:** [0.7261, -1.63, 0.6846, -0.5007, 0.2805]

*Reference: Katsoulis formula DOI:10.6911/WSRJ.202507_11(7).0007*

### Midship Coefficient (C_M)

Three methods are averaged (if within valid range 0.96-1.0):

1. **Method 1:** `C_M₁ = 0.9 + 0.1 × C_B`
2. **Method 2:** `C_M₂ = 1 / (1 + (1 - C_B)^3.5)`
3. **Method 3:** `C_M₃ = 1.006 - (0.0056 × C_B^(-3.56))`

*Reference: Ship Design for Efficiency and Economy - H. Schneekluth and V. Bertram*

### Prismatic Coefficient (C_P)
**Formula:** `C_P = C_B / C_M`

### Waterplane Coefficient (C_WP)

#### Ship-type specific formulations:

**Tanker, Cargo, Bulk Carrier (Bertram and Wobig 1999):**
`C_WP = 0.763 × (C_P + 0.34)`

**Container Ship (Bertram and Wobig):**
`C_WP = 3.226 × (C_P - 0.36)`

**Generic (Papanikolaou 2014):**
`C_WP = (1 + 2 × C_B) / 3`

#### Holtrop adjustment factor:
- **Stern choice 2:** `hb = 0.743 × C_B + 0.297`
- **Stern choice 3:** `hb = (1 + 2 × C_B) / 3`
- **Other:** `hb = 0.95 × C_P + 0.17 × (1 - C_P)^(1/3)`

**Final formula:** `C_WP = (C_WP_type × hb) / 2`

*Reference: Kim et al 2023 https://doi.org/10.1016/j.oceaneng.2023.114758*

---

## Volume and Displacement

### Displacement Volume
**Formula:** `DWT_v = (DWT × 1000) / ρ_SW`

Where:
- `DWT_v` = Displacement volume (m³)
- `DWT` = Deadweight tonnage (tonnes)
- `ρ_SW` = Seawater density (1025 kg/m³)

---

## Draft Parameters

### Aft Draft (T_A)
**Formula:** `T_A = T - (trim / 2)`

Where `trim` is positive by the stern.

### Forward Draft (T_F)
**Formula:** `T_F = T_A + trim`

If trim is not specified, both `T_A` and `T_F` default to `T`.

### Draught ratio
Average annual draught ratio, refers to ratio between current draught (T_C) and design draught (T_D).

(T_C/T_D)

(Olmer et al., 2017)
 taken from  https://doi.org/10.1016/j.oceaneng.2023.114758

| Ship Type | Ballast | Laden | Average |
|-----------|---------|-------|---------|
| Bulk dry | 0.57 | 0.91 | 0.80 |
| Chemical | 0.65 | 0.89 | 0.85 |
| General cargo | 0.65 | 0.89 | 0.84 |
| Liquefied gas | 0.67 | 0.88 | 0.87 |
| Oil | 0.60 | 0.89 | 0.82 |
| Container | – | – | 0.87 |
| Passenger | – | – | 0.93 |
| Ro-Ro | – | – | 0.91 |


---

## Area Parameters

### Midship Area (A_m)
**Formula:** `A_m = B × T × C_M`

### Bulb Transverse Area (A_BT)
**Formula:** `A_BT = 0.09 × A_m`

### Transom Area (A_T)
**Formula:** `A_T = 0.8 × A_BT`

### Waterplane Area (A_W)
**Formula:** `A_W = L × B × C_WP`

### Height of Bulb Center (h_B)
**Formula:** `h_B = 0.45 × T`

---

## Position Parameters

### Longitudinal Center of Buoyancy (lcb)
**Formula:** `lcb = (50 × F_n - 10.5) / 100`

Where:
- `lcb` = LCB position as fraction of L_PP from midships
- `F_n` = Froude number

### Length of Run (L_R)
**Formula:** `L_R = L × (1 - C_P + 0.06 × C_P × lcb / (4 × C_P - 1))

from Holtrop and Mennen`

### Half Angle of Entrance (I_E)
From Holtrop and Mennen
**Formula:**
```
I_E = 1 + 89 × exp(-((L/B)^0.80856) × ((1-C_WP)^0.30484) × 
                   ((1-C_P-0.0225×lcb)^0.6367) × ((L_R/B)^0.34574) × 
                   (100×DWT_v/L³)^0.16302)
                  
```

---

## Form Factors

### Form Factor k1
**Formula:**

From Holtrop and Mennen
```
k1 = (0.93 + 0.487188 × c14 × (B/L)^1.06806 × (T/L)^0.46106 × 
      (L/L_R)^0.121563 × (L³/DWT_v)^0.36486 × (1-C_P)^(-0.604247)) - 1
```

Where `c14 = 1 + 0.01 × C_stern`

### Stern Coefficient (C_stern)
Based on stern shape choice:
- **1** (Pram with gondola): -25
- **2** (V-shaped section): -10
- **3** (Normal section): 0
- **4** (U-shaped section with Hogner stern): 10

---
Taken from Holtrop and Mennen 

## Resistance Parameters

### Frictional Resistance Coefficient (C_F)
**Formula:** `C_F = 0.075 / (log₁₀(Re) - 2)²`

Where `Re` is the Reynolds number.

---

### Hull Wetted Surface Area (S) - Holtrop Method
**Formula:**
```
S = L × (2T + B) × C_M^0.5 × (0.453 + 0.4425×C_B - 0.2862×C_M - 
    0.003467×B/T + 0.3696×C_WP) + 2.38×A_BT/C_B
```

### Alternative Wetted Surface Area Methods

#### General Displacement Volume
**Formula:** `∇_TD = (LDT + DWT) × 10³ / ρ_S`

Where:
- `∇_TD` = Total displacement volume at design draft (m³)
- `LDT` = Light displacement tonnage (tonnes)
- `DWT` = Deadweight tonnage (tonnes)
- `ρ_S` = Seawater density (kg/m³)

or alternatively ∇_TD = DWT_v

#### Ship-Type Specific Wetted Surface Formulations

**General Formula:**
`S_TD = α × (∇_TD / T_D + β × L_WL × T_D)`

Where:
- `S_TD` = Wetted surface area at design draft (m²)
- `T_D` = Design draft (m)
- `L_WL` = Length at waterline (m)
- `α`, `β` = Ship-type specific coefficients

| Ship Type | α | β | Draft Correction Formula |
|-----------|---|---|-------------------------|
| **Container** | 0.99 | 1.9 | `S_TC = S_TD - 2.4(T_D - T_C)` |
| **Bulk Carrier** | 0.99 | 1.9 | `S_TC = S_TD - 2(T_D - T_C)` |
| **Tanker** | 0.99 | 1.9 | `S_TC = S_TD - 2(T_D - T_C)` |
| **Ro-Ro** | 0.87 | 2.7 | `S_TC = S_TD - 3(T_D - T_C)` |
| **Passenger** | 1.11 | 1.7 | `S_TC = S_TD - 3(T_D - T_C)` |

*Reference: Kristensen and Lützen (2012)*

`L_WL` = `1.01L_PP` for Container, RO_RO 
`L_WL` = `1.02L_PP` for others


#### Current Draft Wetted Surface Area
**Formula:** `S_TC = S_TD - γ(T_D - T_C)`

Where:
- `S_TC` = Wetted surface area at current draft (m²)
- `T_C` = Current draft (m)
- `γ` = Draft correction factor (ship-type specific, see table above)

### Appendage Wetted Surface Area (S_APP)
**Formula:** `S_APP = 0.001 × (1 + 50 × C_B² × (B/L_PP)²)`

---

## Appendage Parameters

### Appendage Types and k2 Values

| Type | Description | k2 Value |
|------|-------------|----------|
| 1 | Rudder behind skeg | 1.5 |
| 2 | Rudder behind stern | 1.3 |
| 3 | Twin-screw balance rudders | 2.8 |
| 4 | Shaft brackets | 3.0 |
| 5 | Skeg | 1.5 |
| 6 | Strut bossings | 3.0 |
| 7 | Hull bossings | 2.0 |
| 8 | Shafts | 3.0 |
| 9 | Stabilizer fins | 2.8 |
| 10 | Dome | 2.7 |
| 11 | Bilge keels | 1.4 |

---

## Bow Thruster Parameters

### Bow Thruster Area (A_BT)
When bow thrusters are specified:
- Calculated based on thruster diameter and number
- Default resistance coefficient: 0.008

### Bow Thruster Resistance Coefficient (C_BT)
- **Range:** 0.003 - 0.012
- **Default:** 0.008

---

## Constants Used

| Constant | Symbol | Value | Unit |
|----------|--------|-------|------|
| Gravitational acceleration | g | 9.81 | m/s² |
| Seawater density | ρ_SW | 1025 | kg/m³ |
| Kinematic viscosity (seawater) | ν | - | m²/s |
| Knots to m/s conversion | - | 0.5144 | - |

---

## Ship Type Enumeration

| Ship Type | Key | Description |
|-----------|-----|-------------|
| BULK_CARRIER | 1 | Bulk Carrier |
| TANKER | 2 | Tanker |
| CRUISE_SHIP | 3 | Cruise Ship |
| RO_RO | 4 | RO-RO |
| CARGO | 5 | Cargo |
| CONTAINER_SHIP | 6 | Container Ship |

---

## Parameter Dependencies

The calculation follows a specific dependency order:

1. **Basic parameters** → Speed conversions, dimensionless numbers
2. **Volume/coefficients** → Areas dependent on coefficients
3. **Draft parameters** → Areas dependent on drafts
4. **Position parameters** → Form factors dependent on positions
5. **Wetted surfaces** → Resistance coefficients

---

## Notes

- All formulas include provisions for user-provided overrides via direct parameter setting
- Missing parameters are interpolated using regression analysis on historical ship data
- Correction factors can be applied to modify calculated values
- Parameter validation ensures physical feasibility and warns of out-of-range values

---

*This documentation is based on the ShipShapeBasic and ShipShapeExplorer classes. References should be added for each formula as indicated in the code comments.*