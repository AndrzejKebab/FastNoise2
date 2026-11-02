# Building this fork

These commands build the library, `NodeEditor` and `NodeEditorIpc` from the commit this file is in. Build every
platform from that one commit: the NodeEditor and the library exchange encoded node trees, and node IDs only agree
within one commit.

Each release names its commit and has a `PROVENANCE.txt` per platform. That file lists the machine or CI job, the
compiler version, the container image and the commands that produced the binaries. The release ships the binaries the
tests ran against; nothing is rebuilt for it.

## Windows x64

Use ClangCL, as upstream's README recommends: MSVC has SIMD compiler bugs that cause incorrect generation. In MSVC
Release builds of this fork, the gradient-vs-central-difference checks fail near the origin; MSVC Debug builds pass.

Requires Visual Studio 2026 with "Desktop development with C++" and "C++ Clang tools for Windows". "Developer
PowerShell for VS 2026" starts as x86, so switch it to x64 first, then run from the repository root:

```
& "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl -DBUILD_SHARED_LIBS=ON -DFASTNOISE2_TOOLS=ON -DFASTNOISE2_TESTS=ON
cmake --build build
```

Outputs: `build/Release/bin/FastNoise.dll`, `NodeEditorIpc.dll`, `NodeEditor.exe`.

`tools/package-windows.sh build out` (Git Bash) runs the tests on those binaries and packs exactly them, with the test
output and `PROVENANCE.txt`, into `out/windows-x64.zip`.

## Linux x64

Built on the GitHub `ubuntu-24.04` runner with its default GCC (job `linux` in
`.github/workflows/damped-release.yml`):

```
sudo apt-get install -y ninja-build libgl1-mesa-dev libwayland-dev libxkbcommon-dev xorg-dev
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DFASTNOISE2_TOOLS=ON -DFASTNOISE2_TESTS=ON
cmake --build build
```

The job fails if a binary needs newer symbol versions than the shipped one
(`.github/scripts/check-linux-limits.sh`):

| Binary | Highest allowed |
|---|---|
| `libFastNoise.so` | GLIBC_2.14, GLIBCXX_3.4.29 |
| `libNodeEditorIpc.so` | GLIBC_2.34, GLIBCXX_3.4.21 |
| `NodeEditor` | GLIBC_2.38, GLIBCXX_3.4.32 |

## macOS universal (arm64 + x86_64)

Built on the GitHub `macos-latest` runner (job `macos`):

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DFASTNOISE2_TOOLS=ON -DFASTNOISE2_TESTS=ON -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64" -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0
cmake --build build --parallel 4
```

The job checks that every binary has minos 11.0 for both slices. Job `macos-x86_64` runs the tests against the same
universal binaries on an Intel runner.

## Tests

`FASTNOISE2_TESTS=ON` builds `FastNoiseDampedTests` and `FastNoiseRegressionDump`; see `tests/damped/README.md`.

To compare against the base commit (`721400f`), build the dump against a base checkout first. Then pass the result
directory to the fork's configure:

```
git worktree add ../base 721400f
cmake -S tests/regression -B build-base -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DFASTNOISE2_SOURCE=<absolute path to ../base>
cmake --build build-base
build-base/bin/FastNoiseRegressionDump metadata baseline/metadata.txt
build-base/bin/FastNoiseRegressionDump samples tests/fixtures/base-trees.txt baseline/samples.bin
cmake -S . -B build ... -DFASTNOISE2_REGRESSION_BASELINE=<absolute path to baseline>
ctest --test-dir build --output-on-failure
```

Use the same compiler and flags for both builds. Compare outputs only within one machine.
