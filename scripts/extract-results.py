#!/usr/bin/env python3
"""Validate one HW5 collection and extract its 5 CPU + 42 CUDA + 1 offload rows."""

import argparse
import csv
from decimal import Decimal, InvalidOperation, getcontext
import io
from pathlib import Path
import re
import sys

getcontext().prec = 40
THREADS = (32, 64, 128, 256, 512, 1024)
BLOCKS = (1, 4, 16, 64, 256, 1024, 4096)
TIME = "gpu__time_duration.avg"
OCCUPANCY = "sm__warps_active.avg.pct_of_peak_sustained_active"
DRAM = "dram__throughput.avg.pct_of_peak_sustained_elapsed"
METRICS = (TIME, OCCUPANCY, DRAM)
TIME_TO_MS = {
    "nsecond": Decimal("0.000001"), "ns": Decimal("0.000001"),
    "usecond": Decimal("0.001"), "us": Decimal("0.001"),
    "msecond": Decimal("1"), "ms": Decimal("1"),
    "second": Decimal("1000"), "s": Decimal("1000"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    try:
        result = Decimal(value.strip().replace(",", ""))
    except InvalidOperation as error:
        raise ValueError(f"Invalid numeric metric: {value!r}") from error
    require(result.is_finite(), f"Non-finite metric: {value!r}")
    return result


def check_errors(text, path):
    for marker in ("==ERROR==", "GPUassert:", "No kernels were profiled", "Error reading input", "Error writing output"):
        require(marker not in text, f"{path}: error marker {marker!r}")


def profile(path, implementation):
    """Read NCU raw CSV (metric rows, or transposed metric columns)."""
    text = path.read_text(encoding="utf-8-sig")
    check_errors(text, path)
    rows = [[cell.strip() for cell in row] for row in csv.reader(io.StringIO(text))]
    # Recognize the layout independently of metric availability. A saved report
    # can have a valid header while omitting a metric requested by HW5.
    headers = [i for i, row in enumerate(rows) if "ID" in row and "Kernel Name" in row]
    require(len(headers) == 1, f"{path}: expected one NCU raw CSV header")
    start = headers[0]
    header = rows[start]
    records = [dict(zip(header, row)) for row in rows[start + 1:] if len(row) == len(header)]
    values = {}
    launches = set()
    if "Metric Name" in header:
        require(all(name in header for name in ("Metric Unit", "Metric Value")),
                f"{path}: missing NCU metric unit/value columns")
        for record in records:
            kernel = record["Kernel Name"]
            if not kernel:
                continue
            launches.add((record.get("ID", ""), kernel))
            metric = record["Metric Name"]
            if metric in METRICS:
                require(metric not in values, f"{path}: duplicate metric {metric}")
                values[metric] = (record["Metric Unit"], number(record["Metric Value"]))
    else:
        missing = [metric for metric in METRICS if metric not in header]
        require(not missing, f"{path}: missing required metrics: {', '.join(missing)}")
        units = [record for record in records if not record["Kernel Name"] and record.get(TIME)]
        data = [record for record in records if record["Kernel Name"]]
        require(len(units) == 1 and len(data) == 1, f"{path}: expected one units row and one kernel row")
        record = data[0]
        launches.add((record.get("ID", ""), record["Kernel Name"]))
        values = {metric: (units[0][metric], number(record[metric])) for metric in METRICS}
    require(len(launches) == 1, f"{path}: expected exactly one profiled kernel launch, found {launches}")
    kernel = next(iter(launches))[1]
    expected = "sobel_kernel_gpu" if implementation == "cuda" else "do_sobel_filtering"
    require(expected in kernel, f"{path}: wrong kernel {kernel!r}")
    missing = [metric for metric in METRICS if metric not in values]
    require(not missing, f"{path}: missing required metrics: {', '.join(missing)}")
    unit, duration = values[TIME]
    unit = unit.replace("µ", "u").replace("μ", "u")
    require(unit in TIME_TO_MS and duration > 0, f"{path}: invalid runtime/unit {duration} {unit!r}")
    for metric in (OCCUPANCY, DRAM):
        percent_unit, value = values[metric]
        require(percent_unit in ("%", "percent") and 0 <= value <= 100,
                f"{path}: invalid percentage {metric}: {value} {percent_unit!r}")
    return {
        "runtime_ms": format(duration * TIME_TO_MS[unit], "f"),
        "achieved_occupancy_pct": format(values[OCCUPANCY][1], "f"),
        "dram_pct": format(values[DRAM][1], "f"),
        "kernel": kernel,
    }


def successful_run(prefix, log_suffix):
    require(prefix.with_suffix(".exit-status").read_text().strip() == "0", f"{prefix}: failed exit status")
    path = prefix.with_suffix(log_suffix)
    text = path.read_text()
    check_errors(text, path)
    require("Wrote the output file ../data/processed-raw-int8-4x-cpu.dat" in text,
            f"{path}: missing successful image-write message")
    return text


def extract(root, output):
    for kind in ("cpu", "gpu"):
        require((root / kind / "COMPLETE").is_file(), f"{kind}: collection is incomplete")
    revision = (root / "cpu/revision.txt").read_text()
    require(re.fullmatch(r"[0-9a-f]{40}\s*", revision), "Invalid source revision")
    require(revision == (root / "gpu/revision.txt").read_text(), "CPU/GPU source revisions differ")
    input_hash = (root / "cpu/input.sha256").read_text()
    require(re.match(r"[0-9a-f]{64}\s", input_hash), "Invalid input hash")
    require(input_hash == (root / "gpu/input.sha256").read_text(), "CPU/GPU inputs differ")
    require("Successfully resumed profiling." in (root / "gpu/dcgm-resume.log").read_text(), "DCGM resume is not confirmed")
    cpu_expected = {f"cpu-t{t}.log" for t in (1, 2, 4, 8, 16)}
    require({p.name for p in (root / "cpu").glob("cpu-t*.log")} == cpu_expected,
            "CPU configuration coverage differs from 1,2,4,8,16")
    gpu_expected = {f"cuda-N{t}-B{b}.csv" for t in THREADS for b in BLOCKS} | {"offload.csv"}
    require({p.name for p in (root / "gpu").glob("*.csv")} == gpu_expected,
            "GPU configuration coverage differs from the required 42 CUDA + 1 offload")
    cpu_rows, cuda_rows = [], []
    for threads in (1, 2, 4, 8, 16):
        prefix = root / "cpu" / f"cpu-t{threads}"
        text = successful_run(prefix, ".log")
        times = re.findall(r"^\s*Elapsed time is :\s+([0-9.eE+\-]+)\s*$", text, re.M)
        require(len(times) == 1, f"{prefix}: expected exactly one chrono result")
        elapsed = number(times[0])
        require(elapsed > 0, f"{prefix}: nonpositive CPU time")
        cpu_rows.append({"threads": threads, "runtime_ms": format(elapsed * 1000, "f"),
                         "raw_log": str(prefix.with_suffix(".log").relative_to(root))})
    for threads in THREADS:
        for blocks in BLOCKS:
            prefix = root / "gpu" / f"cuda-N{threads}-B{blocks}"
            text = successful_run(prefix, ".stdout.log")
            launches = re.findall(r"GPU configuration: (\d+) blocks, (\d+) threads per block", text)
            require(launches == [(str(blocks), str(threads))], f"{prefix}: wrong reported launch shape")
            require(prefix.with_suffix(".ncu-rep").stat().st_size > 0, f"{prefix}: missing binary report")
            row = profile(prefix.with_suffix(".csv"), "cuda")
            cuda_rows.append({"threads_per_block": threads, "blocks": blocks, **row,
                              "raw_profile": str(prefix.with_suffix(".csv").relative_to(root)),
                              "raw_log": str(prefix.with_suffix(".stdout.log").relative_to(root))})
    prefix = root / "gpu/offload"
    successful_run(prefix, ".stdout.log")
    require(prefix.with_suffix(".ncu-rep").stat().st_size > 0, f"{prefix}: missing binary report")
    offload = {"implementation": "offload", **profile(prefix.with_suffix(".csv"), "offload"),
               "raw_profile": "gpu/offload.csv", "raw_log": "gpu/offload.stdout.log"}
    # Validate everything before creating any derived output. No silent replacement.
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("cpu.csv", cpu_rows), ("cuda.csv", cuda_rows), ("offload.csv", [offload])):
        with (output / name).open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Validated and extracted 5 CPU, 42 CUDA, 1 offload rows to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection", nargs="?", type=Path)
    parser.add_argument("output", nargs="?", type=Path)
    parser.add_argument("--check-profile", type=Path, help="Validate an existing NCU raw CSV without collecting again")
    parser.add_argument("--kernel", choices=("cuda", "offload"))
    args = parser.parse_args()
    try:
        if args.check_profile:
            if not args.kernel or args.collection or args.output:
                parser.error("--check-profile requires --kernel and no positional arguments")
            print(profile(args.check_profile, args.kernel))
        else:
            if not args.collection or not args.output or args.kernel:
                parser.error("provide COLLECTION_DIR OUTPUT_DIR")
            extract(args.collection, args.output)
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
