# P1-4a: Packaging and Bump-Pitch Specification

## 1. Physical Architecture & Die Geometry

| Parameter | Specification | Rationale |
|---|---|---|
| **Technology Node** | 14nm FinFET (TSMC N16 / Samsung 14LPP) | Meets 1.0 GHz frequency with $\ge 15\%$ timing slack |
| **Die Area** | **$20.56\text{ mm}^2$** ($4.53 \times 4.53\text{ mm}$) | Locked by `sonic/roofline.py` and `tests/test_spec.py` |
| **Package Type** | FO-WLP (Fan-Out Wafer-Level Package) / FCCSP | Low parasitics (<0.35 nH inductance) for LPDDR5X-8533 signaling |
| **Package Footprint**| $9.0 \times 9.0\text{ mm}^2$, 256-ball BGA | 0.50 mm BGA pitch for ultra-compact edge integration |
| **Die Bump Pitch** | **130 µm** pitch area array | Cu pillar micro-bumps (1,215 bump sites on $4.53 \times 4.53\text{ mm}^2$ die) |

---

## 2. Pin & Ball Allocation Summary

```
                      +-----------------------------+
                      |   LPDDR5X PHY Ch 0 & Ch 1   | (32 Signal, 16 P/G)
                      +-----------------------------+
+-------------------+ +-----------------------------+ +-------------------+
|  PCIe Gen4 / Host | |     SONIC S1 CORE DIE       | |  Power / Core VDD |
|  (12 Signal, 6 P) | |    20.56 mm² (4x 64x64)     | |  (0.80V Array)    |
+-------------------+ +-----------------------------+ +-------------------+
                      +-----------------------------+
                      |   LPDDR5X PHY Ch 2 & Ch 3   | (32 Signal, 16 P/G)
                      +-----------------------------+
```

| Domain | Signal Balls | Power / Ground Balls | Total Balls |
|---|---:|---:|---:|
| **LPDDR5X PHY (4x 16-bit Ch)** | 64 | 32 | 96 |
| **Host Interface (PCIe Gen4 x2 / USB4)** | 12 | 6 | 18 |
| **Core Supply ($V_{DD} = 0.80\text{V}$)** | — | 78 | 78 |
| **SRAM & Retained State ($V_{DDM} = 0.70\text{V}$)** | — | 16 | 16 |
| **I/O & PHY Supply ($V_{DDQ} = 1.05\text{V}, 1.8\text{V}$)** | — | 24 | 24 |
| **JTAG, Clocks, Boot & Reset** | 8 | 4 | 12 |
| **Unassigned / Mechanical Anchors** | — | 12 | 12 |
| **Total** | **84** | **172** | **256** |

---

## 3. Signal Integrity & Thermal Envelope

- **Signal Integrity**: Micro-bump parasitics ($L \approx 0.12\text{ nH}$, $C \approx 38\text{ fF}$) allow clean 4266 MHz WCK differential clocking without on-die equalization overshoots.
- **Thermal Dissipation**: 
  - Peak prefill burst power: **1.35 W** (within the **$\le 2.0\text{ W}$ SKU A / $\le 3.5\text{ W}$ SKU B envelope**).
  - Ambient temperature range: $-20^\circ\text{C}$ to $+70^\circ\text{C}$ passive cooling.
  - $R_{\theta JA} = 18.5^\circ\text{C/W}$ natural convection $\implies \Delta T = 1.35\text{W} \times 18.5^\circ\text{C/W} = 25.0^\circ\text{C} \implies T_j \le 95^\circ\text{C}$.
