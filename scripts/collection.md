# HW5 performance collection

Prepared November 2, 2026. Collection tools are in this repository's `scripts/`
directory so they travel with commits/pushes/pulls. Results stay outside the
repository. The assignment grid
is five CPU concurrencies, 42 CUDA launch pairs and one offload configuration,
with one retained run per configuration. No averaging, medians, warm-up runs,
offload tuning or automatic retries are added.

| File | Purpose |
| --- | --- |
| `collect-cpu.sbatch` | CPU node; `OMP_NUM_THREADS=1,2,4,8,16`; capture existing chrono output. |
| `collect-gpu.sbatch` | One GPU; exact CUDA grid followed by one offload profile; save raw CSVs and `.ncu-rep` reports. |
| `extract-results.py` | Validate completion, coverage, source/input agreement, exit status, kernel names, metrics and units; produce three CSV tables. |
| `test-collection.py` | Offline integration tests with synthetic tools/data. Never submits a job or measures a kernel. |

Both batch collectors take two positional arguments: an absolute path to the
existing `build/` directory and an absolute path to a fresh collection directory.
They load `PrgEnv-nvidia`, check the build is under scratch and tracked files are
clean, then run from `build/` so the supplied `../data` paths work.

**CPU collector:** reserves one CPU node/task, runs the five thread counts in
ascending order and logs the program's existing chrono output. It controls
concurrency through `OMP_NUM_THREADS` and fixes placement through environment
variables and Slurm binding. It saves the first CPU image, checks size/content/
border, and compares each later image byte-for-byte with that baseline. Every
run gets its own command, stdout/stderr and exit-status file. `cpu/COMPLETE` is
written only after all five succeed. It does not turn chrono seconds into ms;
the extractor performs that conversion.

**GPU collector:** requires the CPU completion marker and matching source/input
records. It pauses DCGM, profiles every required CUDA pair with `-N`/`-B`, then
profiles offload once with automatic team/thread selection. NCU filters select
the computation kernel; raw CSVs, binary reports and application logs are saved
separately. CUDA images must match the first CUDA baseline; the offload image is
saved and checked separately under the agreed cross-implementation byte policy.
An EXIT handler resumes DCGM, including after normal profiling failures.
`gpu/COMPLETE` requires the full run set and a successful resume command.

**Extractor:** reads completed CPU/GPU directories, validates exact configuration
coverage, successful exits, the reported CUDA launch shapes, one correct kernel
per profile, and the three exact metrics. It rejects missing/duplicate results,
unknown units, non-finite values and profiler errors. Decimal arithmetic converts
time to ms while preserving available digits. The output is five CPU rows,
42 CUDA rows and one offload row, each pointing to its raw evidence. Its
`--check-profile` mode validates an exported existing report without making a
new measurement.

**Offline tests:** run the collectors against temporary synthetic Slurm/NCU
tools and images. They test exact loop coverage, unit/precision handling, CSV
layouts, early failure, overwrite refusal and DCGM cleanup. Use them when
changing these scripts; they neither submit jobs nor establish GPU performance.
From the repository root: `python3 scripts/test-collection.py -v`.

Defaults: account `m3930`; CPU `regular`, GPU `shared`; one task and 32 Slurm
CPU slots per job; CPU walltime 10 minutes, GPU 30 minutes. Account/queue can
be overridden with `sbatch --account=... --qos=...`. Keep the resource settings
consistent with the scripts' `srun` commands. The GPU request uses the live
queue's observed 32-slots-per-GPU requirement. These batch jobs still need their
first target run; earlier verification used interactive allocations.

CPU placement uses `OMP_PLACES=cores`, `OMP_PROC_BIND=spread`, dynamic teams off,
and `srun --cpu-bind=cores`. Thirty-two Slurm CPU slots accommodate 16 physical
cores on Perlmutter. This fixes placement across the five measurements; no
affinity sweep is performed. See [NERSC job examples](https://docs.nersc.gov/jobs/examples/).

GPU collection uses the supplied sweep's basic set and kernel replay, requesting
these exact metrics for both implementations:

```text
gpu__time_duration.avg
sm__warps_active.avg.pct_of_peak_sustained_active
dram__throughput.avg.pct_of_peak_sustained_elapsed
```

The second metric is achieved occupancy, replacing the starter script's cycle
activity metric. Filters select `sobel_kernel_gpu` or `do_sobel_filtering` so a
runtime setup kernel cannot consume the one-launch limit. `--page raw --csv
--print-units base --print-fp` preserves numeric output in base units; the binary
reports are also kept. [NVIDIA CLI reference](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
documents the export, filtering and unit options. DCGM is paused once and resumed
by an EXIT handler, following [NERSC profiling guidance](https://docs.nersc.gov/tools/performance/nvidiaproftools/).

## 1. Commit/push locally, then fetch the tools on Perlmutter

From the repository root on your Mac, review the new files, then commit and push
them manually:

```bash
git add scripts/collect-cpu.sbatch scripts/collect-gpu.sbatch \
    scripts/extract-results.py scripts/test-collection.py scripts/collection.md
git commit -m "Add HW5 collection scripts"
git push origin master
```

On the Perlmutter login node, update the existing detached checkout:

```bash
export HW5_REPO_DIR=/pscratch/sd/a/andyshen/hw5-verify.3CWG57/repo
(
  set -e
  cd "$HW5_REPO_DIR"
  git fetch origin
  git checkout --detach origin/master
  git rev-parse HEAD
  git status --short
)
```

Stop if checkout reports local changes. This updates the scripts without a merge
or forced overwrite. Record the resulting commit; it will be newer than the
implementation-only revision `f17f750`.

## 2. Select the verified build and a fresh collection directory

On the Perlmutter login node, in the same shell used for the remaining steps:

```bash
export HW5_BUILD_DIR="$HW5_REPO_DIR/build"
export HW5_TOOLS_DIR="$HW5_REPO_DIR/scripts"
HW5_COLLECTION_DIR=$(mktemp -d "$SCRATCH/hw5-collection.XXXXXX")
export HW5_COLLECTION_DIR
printf '%s\n' "$HW5_COLLECTION_DIR"
git -C "$HW5_BUILD_DIR/.." rev-parse HEAD
git -C "$HW5_BUILD_DIR/.." status --short
```

The checkout should contain the collection tools and have no tracked changes.
The C++/CUDA implementations verified at `f17f750` are unchanged by adding these
scripts, so their existing verified build can be reused. All three executables
must exist. Keep the printed collection path for download later.

## 3. Check NCU CSV format using an existing report

This imports an already collected report on the login node; it does not launch
the application or collect another GPU measurement. Check the actual installed
NCU export before submitting the production sweep.

```bash
module load PrgEnv-nvidia
ncu --import "$SCRATCH/hw5-verify.3CWG57/logs/offload-smoke-f17f750.ncu-rep" \
    --page raw --csv --print-units base --print-fp \
    --log-file "$HW5_COLLECTION_DIR/offload-format-check.csv"
python3 "$HW5_TOOLS_DIR/extract-results.py" \
    --check-profile "$HW5_COLLECTION_DIR/offload-format-check.csv" --kernel offload
```

Expected: the Sobel offload kernel and all three required metrics, with runtime
converted to ms (approximately 0.50541 for this diagnostic report). If this check
fails, retain the CSV and resolve its format/metric issue before submission.
The extractor accepts NCU's metric-row and transposed metric-column CSV layouts.

## 4. Submit CPU, then GPU with a dependency

The dependency and the GPU collector's CPU-completion check prevent the two jobs
from overwriting the common harness output simultaneously. Both use the same
build. Do not change sources, data or executables while either job is pending/running.

```bash
(
  set -e
  cpu_job=$(sbatch --parsable \
      --output="$HW5_COLLECTION_DIR/slurm-cpu-%j.out" \
      --error="$HW5_COLLECTION_DIR/slurm-cpu-%j.err" \
      "$HW5_TOOLS_DIR/collect-cpu.sbatch" "$HW5_BUILD_DIR" "$HW5_COLLECTION_DIR")
  cpu_job=${cpu_job%%;*}
  printf '%s\n' "$cpu_job" > "$HW5_COLLECTION_DIR/cpu-job.txt"
  gpu_job=$(sbatch --parsable --dependency="afterok:$cpu_job" \
      --output="$HW5_COLLECTION_DIR/slurm-gpu-%j.out" \
      --error="$HW5_COLLECTION_DIR/slurm-gpu-%j.err" \
      "$HW5_TOOLS_DIR/collect-gpu.sbatch" "$HW5_BUILD_DIR" "$HW5_COLLECTION_DIR")
  gpu_job=${gpu_job%%;*}
  printf '%s\n' "$gpu_job" > "$HW5_COLLECTION_DIR/gpu-job.txt"
  squeue -j "$cpu_job,$gpu_job"
)
```

Batch jobs continue after logout. Check progress later with the saved job IDs
and `squeue`; inspect `sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed` after
completion. If CPU fails, the dependent GPU job cannot start: inspect the CPU
logs and cancel that pending GPU job with `scancel GPU_JOB_ID`. Use a fresh
collection directory for a replacement attempt; preserve failed logs.

## 5. Validate, download and extract

After both jobs finish, on the login node:

```bash
cat "$HW5_COLLECTION_DIR/cpu/COMPLETE" "$HW5_COLLECTION_DIR/gpu/COMPLETE"
cat "$HW5_COLLECTION_DIR/gpu/dcgm-resume.log"
python3 "$HW5_TOOLS_DIR/extract-results.py" \
    "$HW5_COLLECTION_DIR" "$HW5_COLLECTION_DIR/derived"
```

Extraction must report **5 CPU, 42 CUDA, 1 offload** rows. It rejects missing or
extra configurations, multiple launches, missing/wrong metrics, invalid values
or units, application/profiler error markers and incomplete cleanup. It keeps
available decimal precision and converts CPU seconds/NCU time units to ms.
There is no automatic winner/trial selection or aggregation.

Back up the complete collection to a new directory under home, preserving any
earlier snapshots. Copy the exact collection path to a fresh directory under
local `hw5/results/`; rerunning extraction locally uses the same command with
local paths. `derived/cpu.csv`, `cuda.csv` and `offload.csv` each link back to
their raw configuration logs. New extraction destinations must not already exist.

Each job also retains source/collector snapshots, revision, input/executable
hashes, build flags, compiler/CMake configuration, environment/job information,
per-run command and exit status. One image per implementation is saved; all CPU
outputs must match their first CPU image and all CUDA outputs their first CUDA
image. Size/nonzero/border checks run after measurement. Cross-implementation
reference-byte differences remain under the existing documented policy.

The jobs write `COMPLETE` only after the entire configuration set succeeds (and,
for GPU, after DCGM resume succeeds). A partial directory is not report data.
These scripts do not rebuild, change harness files, regenerate inputs or alter
the supplied timers. Plot adaptation follows after actual collection data is
validated; no sample heatmap numbers are used.
