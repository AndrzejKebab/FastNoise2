# FractalDamped tests

`FastNoiseDampedTests` covers the value-plus-gradient functions and the `FractalDamped` node. It runs through CTest
when `FASTNOISE2_TESTS` is on.

`GradientProbe` is a test-only SIMD node. It is bound to a library node the same way a node binds its source. It calls
the library node's `GenGradient` and the static `ValueGradient` functions once per feature set that the CPU supports.

## Tolerances

| Check | Tolerance | Why |
|---|---|---|
| `GenGradient` value vs `Gen` | 4 ulps of the output range's largest magnitude, at every point | `/fp:fast` (`-ffast-math`) lets the compiler order the same source differently in `Gen` and in `ValueGradient`. |
| Static `ValueGradient` value vs `Gen` | 4 range ulps + 2 × \|∇n\| × ulp(largest lattice coordinate), with ∇n per cell in output units | The test calls a separately compiled copy. At the ±1e6 world points (1e4 cells), float spacing is 0.001 cell, and GCC orders the skew additions differently in the two copies. That moves the sample by about one position ulp. Near the origin the second term is close to 0. |
| Gradient vs differences | 0.02 + 1% of the gradient length, per cell, in [-1, 1] output units, with a step of 1e-3 cells | On a kink (3D Simplex boundaries), central differences straddle two slopes. The analytic gradient then has to match the forward or the backward difference instead. |
| Analytic vs Approximate | max 0.3, mean 0.03, for \|pos\| ≤ 1000 at Feature Scale 100 and 3 octaves | Approximate uses eps = cell / 8. For Simplex, that central difference is off by up to 70% of the gradient in 2D and 27% in 3D (the test prints both figures). In this domain, \|p_i\| ≤ 4000, where float spacing is 0.0005 against eps 12.5. Outside it, eps can drop below float spacing on high octaves, so a failure there is not a defect. |
| One Approximate octave vs a central difference built by hand | 1e-6 | Same arithmetic. |
| Auto vs explicit Perlin, 4D Analytic vs Approximate, zero slope vs `FractalFBm` | exact | Same operations. |
