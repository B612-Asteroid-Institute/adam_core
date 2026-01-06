# Photometry: V-centric magnitudes and filter conversions

This module implements a **V-band-centric** photometric workflow:

- **Internally, we treat absolute magnitude \(H\) as \(H_V\)** (Johnson–Cousins V).
- When we need magnitudes in other filters (instrument or standard), we convert **from V to the target filter** using a combination of:
  - empirically calibrated linear transformations (e.g., Johnson/Cousins ↔ SDSS, LSST ↔ SDSS) stored in `FILTER_CONVERSIONS` in `simple_magnitude.py`, and
  - **assumed “typical asteroid colors”** when a conversion *requires* color information (notably V ↔ U/B/R/I within Johnson–Cousins).

The intent is to make it easy to:

1. compute/publish \(H_V\) consistently, and
2. reliably transform \(H_V \rightarrow H_\mathrm{filter}\) for the filters used by surveys/instruments (LSST, ZTF, DECam, etc.).

---

## Key point: V ↔ U/B/R/I needs a color assumption

There is **no unique single-filter** transformation from V to U/B/R/I without additional spectral information. In practice, this means:

- If you know the object’s color (or taxonomic type), you should use it.
- If you don’t, you can adopt a **default asteroid color model** (e.g., a population-weighted mix of C and S types).

This README documents a conservative “typical asteroid” approach that yields **simple constant offsets** (slope = 1) for V ↔ U/B/R/I.

**Default mixture note:** the current in-code defaults use an **equal-weight C/S (50/50) mixture**. This is meant as a pragmatic “NEO-ish”
default (NEO/NEA samples are generally more S-rich than the full main belt). It is *not* intended as a global main-belt population model.

---

## How conversions work in code

In `simple_magnitude.py`:

- `FILTER_CONVERSIONS` stores transforms as:
  - \(\mathrm{mag}_\mathrm{to} = a \cdot \mathrm{mag}_\mathrm{from} + b\)
- `find_conversion_path()` finds a short conversion chain (BFS) when a direct transform isn’t available.
- `convert_magnitude()` applies the chain step-by-step.

**Important implications:**

- Multi-hop conversions accumulate uncertainty. Prefer direct transforms where possible.
- When we add V ↔ U/B/R/I defaults, those should be treated as **assumption-based** (not universal physical truths).

---

## Johnson–Cousins V ↔ U/B/R/I using typical asteroid colors

### Definitions

All color indices below use the standard magnitude color-index definitions:

- \(B-V = B - V\)
- \(U-B = U - B\)
- \(V-R = V - R\)
- \(V-I = V - I\)

From these, you can derive the constant-offset conversions:

#### V ↔ B

- \(B = V + (B-V)\)
- \(V = B - (B-V)\)

#### V ↔ R

- \(R = V - (V-R)\)
- \(V = R + (V-R)\)

#### V ↔ I (`I_BAND`)

In this repository, `I_BAND` is intended to represent the Johnson–Cousins **I (Cousins I\(_C\))** band.

- \(I = V - (V-I)\)
- \(V = I + (V-I)\)

#### V ↔ U

Combine \(U-B\) and \(B-V\):

- \(U = V + (B-V) + (U-B)\)
- \(V = U - \big[(B-V) + (U-B)\big]\)

So \(U-V = (B-V) + (U-B)\).

---

## Recommended sources and values

### 1) \(B-V\) and \(U-B\): Bowell & Lumme (1979, *Asteroids*)

**Primary source**

- Bowell, E. & Lumme, K. (1979). *Colorimetry and magnitudes of asteroids*.
  In: Gehrels, T. (ed.), **Asteroids**. University of Arizona Press.

**Table used**

- **Table VII: “Mean Color Indices of Asteroids”** (section “H. Mean color indices”).
- The table reports mean observed colors at mean phase angle \(\alpha\) and also provides **\(\alpha = 0^\circ\)** (zero-phase) colors.

**Zero-phase mean colors (from Table VII)**

- **C-type**: \(B-V = 0.70\), \(U-B = 0.34\)
- **S-type**: \(B-V = 0.84\), \(U-B = 0.42\)

Derived:

- **C-type**: \(U-V = 0.70 + 0.34 = 1.04\)
- **S-type**: \(U-V = 0.84 + 0.42 = 1.26\)

Notes:

- These values are explicitly tied to the dataset and methodology in that chapter (including a phase-reddening correction described around Table VI/VII).
- U-band asteroid photometry is intrinsically harder and noisier; treat U-related conversions as lower-confidence defaults.

### 2) \(V-R\) and \(V-I\): Erasmus et al. (2019, KMTNet-SAAO multi-band photometry)

**Primary source**

- Erasmus, N., McNeill, A., Mommert, M., Trilling, D. E., Sickafoose, A. A., & Paterson, K. (2019).
  *A Taxonomic Study of Asteroid Families from KMTNet-SAAO Multi-band Photometry*.
  The Astrophysical Journal Supplement Series (ApJS).
  DOI: **10.3847/1538-4365/ab1344**
  arXiv: **1903.08019**

**Important detail (Table 1 footnote b)**

Erasmus et al. report **solar-corrected** colors in their Table 1:

- “Colors have been corrected for solar colors by subtracting the respective
  \(V-R = 0.41\) and \(V-I = 0.75\) solar colors (Binney & Merrifield 1998).”

So to recover absolute (non-solar-corrected) colors:

- \((V-R)_\mathrm{abs} = (V-R)_\mathrm{table} + 0.41\)
- \((V-I)_\mathrm{abs} = (V-I)_\mathrm{table} + 0.75\)

**How we turned this into C/S defaults**

Erasmus et al. provide a probabilistic taxonomy per object. Their Table 1 includes a final `Tax.` label (e.g., `C` or `S`).
Using the table rows with `Tax.` exactly `C` or `S`, we computed simple means:

- **C-type** (n=672): \(V-R \approx 0.3662\), \(V-I \approx 0.6831\)
- **S-type** (n=1166): \(V-R \approx 0.4960\), \(V-I \approx 0.8531\)

Empirical scatter in that dataset (population scatter + measurement noise):

- C-type: \(\sigma(V-R) \approx 0.0566\), \(\sigma(V-I) \approx 0.0464\)
- S-type: \(\sigma(V-R) \approx 0.0402\), \(\sigma(V-I) \approx 0.0608\)

**Solar colors citation**

Erasmus et al. cite:

- Binney, J. & Merrifield, M. (1998). *Galactic Astronomy*. Princeton University Press.
  (Used for \(V-R_\odot = 0.41\), \(V-I_\odot = 0.75\) in their reduction.)

---

## Default “typical asteroid” as a C/S mixture

If you want a single default for “typical asteroid colors,” define an S-type weight \(f_S\) (0–1) and compute:

- \((B-V)_\mathrm{mix} = (1-f_S)\,(B-V)_C + f_S\,(B-V)_S\)
- \((U-B)_\mathrm{mix} = (1-f_S)\,(U-B)_C + f_S\,(U-B)_S\)
- \((V-R)_\mathrm{mix} = (1-f_S)\,(V-R)_C + f_S\,(V-R)_S\)
- \((V-I)_\mathrm{mix} = (1-f_S)\,(V-I)_C + f_S\,(V-I)_S\)

Example **50/50 mixture** (\(f_S = 0.5\)):

- \(B-V = 0.77\)
- \(U-B = 0.38\)
- \(U-V = 1.15\)
- \(V-R \approx 0.431\)
- \(V-I \approx 0.768\)

This yields constant-offset transforms that are “good enough” for population-level defaults when no better color information exists.

---

## How this enables \(H_V \rightarrow\) instrument filters

Once V ↔ U/B/R/I is available (either per-type or as a default mixture), you can usually reach an instrument filter via the existing transform graph.

Examples (conceptual):

- \(H_V \rightarrow H_g\) (SDSS) via the existing Jordi et al. relation in `FILTER_CONVERSIONS`
- \(H_V \rightarrow H_\mathrm{LSST\_r}\) directly (existing LSST transforms)
- \(H_V \rightarrow H_\mathrm{DECam\_r}\) via SDSS r and the DECam↔SDSS transforms

The goal is to avoid ad-hoc survey-by-survey hacks: V is the hub; conversions are centralized and documented.

---

## Caveats and best practices

- **These are defaults**. Individual asteroids can deviate materially from mean C/S colors.
- **Phase angle matters**. Colors can change with phase (phase reddening). Bowell & Lumme provide a zero-phase correction framework; Erasmus et al. uses measured colors from specific observing geometries and reductions.
- **U-band is the weakest link**. U is difficult observationally and more sensitive to spectral slope/UV drop-off.
- **Prefer measured colors** when available (e.g., from multi-band observations), or use taxonomic-type-specific colors rather than a global mixture.


