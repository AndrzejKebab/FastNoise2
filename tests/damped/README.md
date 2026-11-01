# FractalDamped tests

`FastNoiseDampedTests` covers the value-plus-gradient functions and the `FractalDamped` node. It runs through CTest
when `FASTNOISE2_TESTS` is on.

`GradientProbe` is a test-only SIMD node. It is bound to a library node the same way a node binds its source. It calls
the library node's `GenGradient` and the static `ValueGradient` functions once per feature set that the CPU supports.

## Tolerances

| Check | Tolerance | Why |
|---|---|---|
| `GenGradient` value vs `Gen` | 4 ulps of the output range's largest magnitude, at every point | `/fp:fast` (`-ffast-math`) lets the compiler order the same source differently in `Gen` and in `ValueGradient`. |
| Static `ValueGradient` value vs `Gen` | 4 range ulps + k × \|∇n\| × ulp(largest lattice coordinate), with ∇n per cell in output units. k counts the roundings from the input coordinate to the vertex offset: Simplex and SuperSimplex 8 in 2D, 10 in 3D, 12 in 4D; Perlin and Value 2 | The test calls a separately compiled copy. At the ±1e6 world points (1e4 cells), float spacing is 0.001 cell, and GCC orders the skew additions differently in the two copies. That moves the sample by about one position ulp. Near the origin the second term is close to 0. |
| Gradient vs differences | 0.02 + 1% of the gradient length, per cell, in [-1, 1] output units, with a step of 1e-3 cells | On a kink (3D Simplex boundaries), central differences straddle two slopes. The analytic gradient then has to match the forward or the backward difference instead. |
| Analytic vs ApproximateAccurate | max 0.3, mean 0.03 | Domain: near the origin (\|pos\| up to about 1900, Feature Scale 100, 3 octaves) and around 1e5 cells, at every SIMD level. Measured worst on ClangCL: max 0.194, mean 0.020 (2D, 1e5 cells). |
| Analytic vs ApproximateFast | max 0.3; mean 0.05 in 2D, 0.03 in 3D and 4D | Same domain. Measured worst on ClangCL: max 0.235, mean 0.045 (2D, 1e5 cells); the 2D mean bound is that plus a 0.005 margin. |
| Accurate vs Fast | Accurate's maximum error is lower, in 2D and 3D, in the domain | Accurate costs 2N extra samples per octave against Fast's N, so it has to be the more accurate one. |
| Around 1e6 cells | finite only; printed, not bounded | Positions resolve to 0.08–0.3 cells there, so the step floor (\|p\| / 2^21) replaces eps and every finite-difference variant samples the same rounded points (2D max 0.84, mean 0.17). |
| One Approximate octave vs differences built by hand | 1e-5 | Same arithmetic, with the realised step `(p + eps) - p`. Run at 1e5 cells too, where p + eps rounds; it fails if a compiler folds the step back to eps. |
| Auto vs explicit Perlin, zero slope vs `FractalFBm` | exact | Same operations. |
