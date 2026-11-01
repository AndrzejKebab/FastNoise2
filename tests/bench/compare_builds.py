"""Benchmark two FastNoiseBenchmark builds in alternating rounds and report medians and speed-ups.

Usage: python compare_builds.py <before FastNoiseBenchmark> <after FastNoiseBenchmark> [filter] [rounds]

Each round runs both executables once, alternating which goes first, pinned to one core at high priority
where the OS allows it. Alternating spreads background load across both builds.
"""
import csv
import io
import os
import statistics
import subprocess
import sys


def run(exe, bench_filter):
    args = [exe, f"--benchmark_filter={bench_filter}", "--benchmark_format=csv", "--benchmark_repetitions=3",
            "--benchmark_report_aggregates_only=true"]
    if os.name == "nt":
        # start /high /affinity pins to one core; /wait /b keeps output in this console
        args = ["cmd", "/c", "start", "/wait", "/b", "/high", "/affinity", "0x10"] + args
    out = subprocess.run(args, capture_output=True, text=True, check=True).stdout
    rows = csv.reader(io.StringIO(out[out.index("name,"):]))
    next(rows)
    return {row[0].replace("_median", ""): float(row[3]) / 1000 for row in rows if row[0].endswith("_median")}


def main():
    before, after = sys.argv[1], sys.argv[2]
    bench_filter = sys.argv[3] if len(sys.argv) > 3 else "UniformGrid[234]D/AVX2/"
    rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 5

    samples = {"before": {}, "after": {}}
    for r in range(rounds):
        order = [("before", before), ("after", after)] if r % 2 == 0 else [("after", after), ("before", before)]
        for tag, exe in order:
            for name, us in run(exe, bench_filter).items():
                samples[tag].setdefault(name, []).append(us)

    print(f"{'benchmark':<62} {'before us':>10} {'after us':>10} {'speed-up':>9}")
    for name in samples["before"]:
        b = statistics.median(samples["before"][name])
        a = statistics.median(samples["after"].get(name, [float("nan")]))
        print(f"{name:<62} {b:>10.1f} {a:>10.1f} {b / a:>8.2f}x")


if __name__ == "__main__":
    main()
