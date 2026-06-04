# Stellar Classification Dataset - Feature Reference

## Overview

This dataset is derived from astronomical survey observations (similar to SDSS - Sloan Digital Sky Survey) and is used to classify celestial objects into:

- STAR
- GALAXY
- QSO (Quasi-Stellar Object / Quasar)

The features can be grouped into:

1. Positional Features
2. Photometric Features
3. Cosmological Features
4. Astrophysical Features
5. Target Variable

---

# 1. id

## Description

Unique identifier for each observation.

## Type

Integer

## Example

```text
12345
```

## Notes

- Used only for identification.
- Contains no physical information about the object.
- Should generally be excluded from model training.

---

# 2. alpha (Right Ascension)

## Description

Right Ascension (RA) is the celestial equivalent of longitude on Earth.

It specifies the east-west position of an object on the celestial sphere.

## Range

```text
0° to 360°
```

## Interpretation

```text
alpha = 0°
alpha = 360°
```

represent nearly the same direction in the sky.

## Example

```text
alpha = 210°
```

means the object is located 210 degrees around the celestial sphere from the reference point.

## Feature Engineering Ideas

Because alpha is circular:

```python
alpha_sin = sin(radians(alpha))
alpha_cos = cos(radians(alpha))
```

are often more useful than the raw value.

## Physical Meaning

Represents the sky location of the object.

---

# 3. delta (Declination)

## Description

Declination is the celestial equivalent of latitude on Earth.

It specifies how far north or south an object is from the celestial equator.

## Theoretical Range

```text
-90° to +90°
```

## Interpretation

```text
+90° = North Celestial Pole
0°   = Celestial Equator
-90° = South Celestial Pole
```

## Example

```text
delta = +45°
```

means the object lies 45 degrees north of the celestial equator.

## Feature Engineering Ideas

```python
delta_sin = sin(radians(delta))
delta_cos = cos(radians(delta))
```

## Physical Meaning

Represents the sky location of the object.

---

# 4. u

## Description

Apparent magnitude measured through the SDSS ultraviolet filter.

## Wavelength

```text
~355 nm
```

## Interpretation

Measures how bright the object appears in ultraviolet light.

## Important

Magnitude scale is reversed:

```text
Smaller magnitude = Brighter object
Larger magnitude  = Dimmer object
```

## Example

```text
u = 14
```

is brighter than

```text
u = 20
```

## Physical Meaning

Useful for identifying hot and blue objects.

---

# 5. g

## Description

Apparent magnitude measured through the SDSS green-blue filter.

## Wavelength

```text
~475 nm
```

## Interpretation

Measures brightness in the blue-green portion of the spectrum.

## Physical Meaning

Important for stellar color analysis.

---

# 6. r

## Description

Apparent magnitude measured through the SDSS red filter.

## Wavelength

```text
~622 nm
```

## Interpretation

Measures brightness in the red portion of the spectrum.

## Physical Meaning

Useful for color and temperature estimation.

---

# 7. i

## Description

Apparent magnitude measured through the SDSS near-infrared filter.

## Wavelength

```text
~763 nm
```

## Interpretation

Measures brightness in the near-infrared region.

## Physical Meaning

Helpful for identifying cooler stars and red galaxies.

---

# 8. z

## Description

Apparent magnitude measured through the SDSS infrared filter.

## Wavelength

```text
~905 nm
```

## Interpretation

Measures brightness in the infrared region.

## Physical Meaning

Strongly related to red and cool objects.

---

# Magnitude Scale (u, g, r, i, z)

All five photometric features use the astronomical magnitude system.

## Rule

```text
Lower Magnitude = Brighter
Higher Magnitude = Dimmer
```

## Brightness Relationship

```text
1 magnitude difference ≈ 2.512x brightness difference
5 magnitude difference = 100x brightness difference
```

## Example

| Magnitude | Brightness |
| --------- | ---------- |
| 15        | Bright     |
| 20        | Dim        |
| 25        | Very Dim   |

---

# Color Indices

Astronomers often use magnitude differences rather than raw magnitudes.

## Examples

```python
u_g = u - g
g_r = g - r
r_i = r - i
i_z = i - z
```

## Why?

Color indices describe the shape of an object's spectrum.

They are often more informative than raw magnitudes.

---

# 9. redshift

## Description

Measures how much an object's emitted light has been stretched before reaching Earth.

## Formula

z = (Observed Wavelength - Emitted Wavelength) / Emitted Wavelength

## Interpretation

### Positive Redshift

```text
z > 0
```

Object is moving away or is affected by cosmic expansion.

### Negative Redshift

```text
z < 0
```

Blueshift.

Object is moving toward us.

## Example

```text
z = 0.5
```

means:

Observed Wavelength = 1.5 × Emitted Wavelength

## Physical Meaning

Higher redshift generally indicates:

```text
Farther distance
Older emitted light
Greater cosmic expansion effects
```

## Typical Pattern

```text
STAR    -> Very low redshift
GALAXY  -> Moderate redshift
QSO     -> High redshift
```

## Machine Learning Importance

Usually one of the strongest predictive features.

---

# 10. spectral_type

## Description

Temperature-based stellar spectral classification.

## Categories

### O/B

Very hot blue stars.

```text
Highest temperature
Strong UV emission
```

### A/F

Hot white stars.

```text
High temperature
Moderate UV emission
```

### G/K

Sun-like stars.

```text
Intermediate temperature
Yellow/orange appearance
```

### M

Cool red stars.

```text
Lowest temperature
Strong infrared emission
```

## Temperature Order

```text
O/B → A/F → G/K → M

Hot -----------------> Cool
```

## Machine Learning Importance

Strongly related to:

- Color
- Temperature
- Spectral energy distribution

---

# 11. galaxy_population

## Description

Broad galaxy evolutionary grouping.

## Categories

### Blue_Cloud

Young galaxies.

Characteristics:

```text
Active star formation
Large population of young stars
Bluer colors
```

### Red_Sequence

Older galaxies.

Characteristics:

```text
Little star formation
Older stellar populations
Redder colors
```

## Physical Meaning

Represents the evolutionary state of a galaxy.

## Expected Relationships

```text
Blue_Cloud  -> Lower color indices
Red_Sequence -> Higher color indices
```

---

# 12. spectral*type*&\_galaxy_population

## Description

Interaction feature combining:

```text
spectral_type
+
galaxy_population
```

## Example Values

```text
O/B_Blue_Cloud
M_Red_Sequence
G/K_Blue_Cloud
```

## Purpose

Captures relationships between:

- Stellar temperature
- Galaxy evolutionary state

## Machine Learning Importance

Potentially more informative than either feature alone.

---

# 13. class (Target Variable)

## Description

Target variable representing the type of celestial object.

---

## STAR

A single self-luminous object powered by nuclear fusion.

Examples:

```text
Sun
Sirius
Betelgeuse
```

Characteristics:

```text
Individual object
Produces its own light
Typically very low redshift
```

---

## GALAXY

A gravitationally bound collection of billions of stars, gas, dust, and dark matter.

Examples:

```text
Milky Way
Andromeda
```

Characteristics:

```text
Contains billions of stars
Moderate redshift
May belong to Blue Cloud or Red Sequence
```

---

## QSO (Quasi-Stellar Object / Quasar)

An extremely luminous active galactic nucleus powered by a supermassive black hole.

Characteristics:

```text
Very distant object
Very high luminosity
Typically high redshift
```

Can outshine the entire host galaxy.

---

# Summary

| Feature Group      | Columns                            |
| ------------------ | ---------------------------------- |
| Identifier         | id                                 |
| Position           | alpha, delta                       |
| Photometry         | u, g, r, i, z                      |
| Cosmology          | redshift                           |
| Stellar Properties | spectral_type                      |
| Galaxy Evolution   | galaxy_population                  |
| Interaction        | spectral*type*&\_galaxy_population |
| Target             | class                              |
