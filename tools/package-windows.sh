#!/usr/bin/env bash
# Usage: tools/package-windows.sh <build dir> <out dir>
# Runs the tests on the built binaries, then packs exactly those binaries with their test output and provenance.
set -euo pipefail

build="$1"
out="$2"
vs="/c/Program Files/Microsoft Visual Studio/18/Community"
ctest="$vs/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/ctest.exe"
dumpbin="$(ls -d "$vs"/VC/Tools/MSVC/*/bin/Hostx64/x64/dumpbin.exe | tail -1)"

rm -rf "$out/windows-x64"
mkdir -p "$out/windows-x64/bin"
"$ctest" --test-dir "$build" --output-on-failure --verbose > "$out/windows-x64/test-output.txt" 2>&1
cp "$build"/Release/bin/{FastNoise.dll,NodeEditorIpc.dll,NodeEditor.exe} "$out/windows-x64/bin/"

{
    echo "commit: $(git rev-parse HEAD)"
    os=$(powershell -NoProfile -Command "(Get-CimInstance Win32_OperatingSystem).Caption" | tr -d '\r')
    cpu=$(powershell -NoProfile -Command "(Get-CimInstance Win32_Processor).Name" | tr -d '\r' | sed 's/ *$//')
    echo "machine: local, $os, $cpu"
    echo "container: none"
    echo "compiler: $("$vs/VC/Tools/Llvm/x64/bin/clang-cl.exe" --version | head -1)"
    echo "configure: cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl -DBUILD_SHARED_LIBS=ON -DFASTNOISE2_TOOLS=ON -DFASTNOISE2_TESTS=ON -DFASTNOISE2_REGRESSION_BASELINE=<base dump dir>"
    echo "build: cmake --build build"
    echo "tests: $(grep -oE '[0-9]+ checks, [0-9]+ failures' "$out/windows-x64/test-output.txt" | head -1); $(grep -oE '[0-9]+% tests passed.*' "$out/windows-x64/test-output.txt")"
    echo "FastNoise.dll dependencies: $("$dumpbin" //nologo //dependents "$out/windows-x64/bin/FastNoise.dll" | grep -i '\.dll' | grep -v 'Dump of' | sed 's/^ *//' | tr '\n' ' ')"
} > "$out/windows-x64/PROVENANCE.txt"

(cd "$out" && powershell -NoProfile -Command "Compress-Archive -Path windows-x64 -DestinationPath windows-x64.zip -Force")
cat "$out/windows-x64/PROVENANCE.txt"
