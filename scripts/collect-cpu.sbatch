#!/bin/bash -l
#SBATCH --account=m3930
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:10:00
#SBATCH --job-name=hw5-cpu
#SBATCH --output=hw5-cpu-%j.out
#SBATCH --error=hw5-cpu-%j.err

# Usage: sbatch collect-cpu.sbatch ABS_BUILD_DIR ABS_COLLECTION_DIR
set -euo pipefail
[[ $# == 2 ]] || { echo "Usage: $0 BUILD_DIR COLLECTION_DIR" >&2; exit 2; }
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${SCRATCH:?SCRATCH is not set}"
module load PrgEnv-nvidia
export LC_ALL=C

build_dir=$(cd "$1" && pwd -P)
repo_dir=$(cd "$build_dir/.." && pwd -P)
scratch_dir=$(cd "$SCRATCH" && pwd -P)
collection_dir=$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$2")
[[ $build_dir == "$scratch_dir/"* ]] || { echo "Build must be under SCRATCH" >&2; exit 2; }
[[ $collection_dir != "$repo_dir" && $collection_dir != "$repo_dir/"* ]] || { echo "Put results outside the harness" >&2; exit 2; }
[[ -z $(git -C "$repo_dir" status --porcelain --untracked-files=no) ]] || { echo "Tracked source changes must be resolved before collection" >&2; exit 2; }
[[ -x $build_dir/sobel_cpu && -f $build_dir/CMakeCache.txt ]]
mkdir -p "$collection_dir"
result_dir="$collection_dir/cpu"
mkdir "$result_dir"  # Refuse to overwrite an earlier collection.
mkdir "$result_dir/source" "$result_dir/images"
cd "$build_dir"  # The harness reads/writes ../data.

cp "$0" "$result_dir/collector.sbatch"
cp "$repo_dir/sobel_cpu.cpp" "$repo_dir/CMakeLists.txt" "$result_dir/source/"
cp CMakeCache.txt "$result_dir/"
cp CMakeFiles/sobel_cpu.dir/flags.make "$result_dir/cpu-flags.make"
cp CMakeFiles/sobel_cpu.dir/link.txt "$result_dir/cpu-link.txt"
cp CMakeFiles/[0-9]*/CMakeCXXCompiler.cmake "$result_dir/"
git -C "$repo_dir" rev-parse HEAD > "$result_dir/revision.txt"
sha256sum sobel_cpu > "$result_dir/executable.sha256"
sha256sum ../data/zebra-gray-int8-4x > "$result_dir/input.sha256"
# Set host concurrency and placement through environment variables.
export OMP_DYNAMIC=false OMP_PROC_BIND=spread OMP_PLACES=cores
unset OMP_SCHEDULE OMP_THREAD_LIMIT NVCOMPILER_CPU_HARD_THREAD_LIMIT
{
    date -u
    hostname
    printf 'Job: %s\nBuild: %s\nCPUs/task: %s\n' "$SLURM_JOB_ID" "$build_dir" "$SLURM_CPUS_PER_TASK"
    module list
    env | sort | awk '/^(OMP_|NVCOMPILER_OMP_|NVCOMPILER_CPU_|CUDA_VISIBLE_DEVICES=)/'
    scontrol show job "$SLURM_JOB_ID"
    srun -n 1 -c 32 --cpu-bind=cores lscpu
    srun -n 1 -c 32 --cpu-bind=cores awk '/Cpus_allowed_list/' /proc/self/status
} > "$result_dir/environment.log" 2>&1

# 32 Slurm CPU slots provide room for 16 physical cores on Perlmutter.
for threads in 1 2 4 8 16
do
    prefix="$result_dir/cpu-t${threads}"
    command=(srun -n 1 -c 32 --cpu-bind=cores env OMP_NUM_THREADS="$threads" ./sobel_cpu)
    printf '%q ' "${command[@]}" > "$prefix.command.txt"
    printf '\n' >> "$prefix.command.txt"
    if "${command[@]}" 2>&1 | tee "$prefix.log"; then
        printf '0\n' > "$prefix.exit-status"
    else
        status=$?
        printf '%s\n' "$status" > "$prefix.exit-status"
        exit "$status"
    fi
    grep -Fq 'Wrote the output file ../data/processed-raw-int8-4x-cpu.dat' "$prefix.log"
    if [[ $threads == 1 ]]; then
        cp ../data/processed-raw-int8-4x-cpu.dat "$result_dir/images/cpu.dat"
        python3 - "$result_dir/images/cpu.dat" <<'PY' > "$result_dir/image-checks.log"
from pathlib import Path
import sys
data = Path(sys.argv[1]).read_bytes()
cols, rows = 7112, 5146
if len(data) != cols * rows or not any(data):
    raise SystemExit('Image size/nonzero check failed')
border = data[:cols] + data[-cols:] + data[cols:-cols:cols] + data[2*cols-1:-cols:cols]
if len(border) != 24512 or any(border):
    raise SystemExit('Zero-border check failed')
print('cpu-t1: size/nonzero/zero-border PASS')
PY
    else
        cmp "$result_dir/images/cpu.dat" ../data/processed-raw-int8-4x-cpu.dat >> "$result_dir/image-checks.log" 2>&1
        printf 'cpu-t%s: matches baseline\n' "$threads" >> "$result_dir/image-checks.log"
    fi
done
printf '5 CPU configurations completed\n' > "$result_dir/COMPLETE"
