#!/bin/bash -l
#SBATCH --account=m3930
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --job-name=hw5-gpu
#SBATCH --output=hw5-gpu-%j.out
#SBATCH --error=hw5-gpu-%j.err

# Submit after the CPU job: both use the harness's common output path.
# Usage: sbatch collect-gpu.sbatch ABS_BUILD_DIR ABS_COLLECTION_DIR
set -euo pipefail
[[ $# == 2 ]] || { echo "Usage: $0 BUILD_DIR COLLECTION_DIR" >&2; exit 2; }
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${SCRATCH:?SCRATCH is not set}"
module load PrgEnv-nvidia
export LC_ALL=C OMP_TARGET_OFFLOAD=MANDATORY
# Leave offload team/thread selection to the compiler/runtime, as assigned.
unset OMP_NUM_THREADS OMP_NUM_TEAMS OMP_TEAMS_THREAD_LIMIT NVCOMPILER_OMP_CUDA_GRID
unset OMP_PROC_BIND OMP_PLACES OMP_SCHEDULE

build_dir=$(cd "$1" && pwd -P)
repo_dir=$(cd "$build_dir/.." && pwd -P)
scratch_dir=$(cd "$SCRATCH" && pwd -P)
collection_dir=$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$2")
[[ $build_dir == "$scratch_dir/"* ]] || { echo "Build must be under SCRATCH" >&2; exit 2; }
[[ $collection_dir != "$repo_dir" && $collection_dir != "$repo_dir/"* ]] || { echo "Put results outside the harness" >&2; exit 2; }
[[ -s $collection_dir/cpu/COMPLETE ]] || { echo "Complete the CPU collection first" >&2; exit 2; }
[[ -z $(git -C "$repo_dir" status --porcelain --untracked-files=no) ]] || { echo "Tracked source changes must be resolved before collection" >&2; exit 2; }
[[ -x $build_dir/sobel_gpu && -x $build_dir/sobel_cpu_omp_offload ]]
result_dir="$collection_dir/gpu"
mkdir "$result_dir"  # Refuse to overwrite an earlier collection.
mkdir "$result_dir/source" "$result_dir/images"
cd "$build_dir"

cp "$0" "$result_dir/collector.sbatch"
cp "$repo_dir/sobel_gpu.cu" "$repo_dir/sobel_cpu_omp_offload.cpp" "$repo_dir/CMakeLists.txt" "$result_dir/source/"
cp CMakeCache.txt "$result_dir/"
cp CMakeFiles/[0-9]*/CMakeCXXCompiler.cmake CMakeFiles/[0-9]*/CMakeCUDACompiler.cmake "$result_dir/"
for target in sobel_gpu sobel_cpu_omp_offload
do
    cp "CMakeFiles/$target.dir/flags.make" "$result_dir/$target-flags.make"
    cp "CMakeFiles/$target.dir/link.txt" "$result_dir/$target-link.txt"
done
git -C "$repo_dir" rev-parse HEAD > "$result_dir/revision.txt"
cmp "$collection_dir/cpu/revision.txt" "$result_dir/revision.txt"
sha256sum sobel_gpu sobel_cpu_omp_offload > "$result_dir/executable.sha256"
sha256sum ../data/zebra-gray-int8-4x > "$result_dir/input.sha256"
cmp "$collection_dir/cpu/input.sha256" "$result_dir/input.sha256"
{
    date -u
    hostname
    printf 'Job: %s\nBuild: %s\nCPUs/task: %s\n' "$SLURM_JOB_ID" "$build_dir" "$SLURM_CPUS_PER_TASK"
    module list
    env | sort | awk '/^(OMP_|NVCOMPILER_OMP_|NVCOMPILER_CPU_|CUDA_VISIBLE_DEVICES=)/'
    scontrol show job "$SLURM_JOB_ID"
    srun -n 1 -c 32 --gpus-per-task=1 nvidia-smi
    ncu --version
    nvcc --version
} > "$result_dir/environment.log" 2>&1

paused=0
cleanup()
{
    status=$?
    trap - EXIT
    if [[ $paused == 1 ]]; then
        if ! srun -n 1 -c 32 --gpus-per-task=1 dcgmi profile --resume > "$result_dir/dcgm-resume.log" 2>&1; then
            status=1
        fi
    fi
    if [[ $status == 0 ]]; then
        printf '42 CUDA configurations and 1 offload configuration completed\n' > "$result_dir/COMPLETE"
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
srun -n 1 -c 32 --gpus-per-task=1 dcgmi profile --pause > "$result_dir/dcgm-pause.log" 2>&1
paused=1

metrics=gpu__time_duration.avg,sm__warps_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed
profile()
{
    case_name=$1
    kernel=$2
    shift 2
    prefix="$result_dir/$case_name"
    command=(srun -n 1 -c 32 --gpus-per-task=1 ncu
        --set basic --metrics "$metrics" --replay-mode kernel
        --kernel-name "regex:$kernel" --launch-count 1 --check-exit-code yes
        --page raw --csv --print-units base --print-fp
        --export "$prefix" --log-file "$prefix.csv" "$@")
    printf '%q ' "${command[@]}" > "$prefix.command.txt"
    printf '\n' >> "$prefix.command.txt"
    if "${command[@]}" 2>&1 | tee "$prefix.stdout.log"; then
        printf '0\n' > "$prefix.exit-status"
    else
        status=$?
        printf '%s\n' "$status" > "$prefix.exit-status"
        exit "$status"
    fi
    [[ -s $prefix.csv && -s $prefix.ncu-rep ]]
    grep -Fq 'Wrote the output file ../data/processed-raw-int8-4x-cpu.dat' "$prefix.stdout.log"
}

check_image()
{
    python3 - "$1" <<'PY' >> "$result_dir/image-checks.log"
from pathlib import Path
import sys
path = Path(sys.argv[1])
data = path.read_bytes()
cols, rows = 7112, 5146
if len(data) != cols * rows or not any(data):
    raise SystemExit('Image size/nonzero check failed')
border = data[:cols] + data[-cols:] + data[cols:-cols:cols] + data[2*cols-1:-cols:cols]
if len(border) != 24512 or any(border):
    raise SystemExit('Zero-border check failed')
print(path.name, 'size/nonzero/zero-border PASS')
PY
}

for threads in 32 64 128 256 512 1024
do
    for blocks in 1 4 16 64 256 1024 4096
    do
        profile "cuda-N${threads}-B${blocks}" sobel_kernel_gpu ./sobel_gpu -N "$threads" -B "$blocks"
        if [[ $threads == 32 && $blocks == 1 ]]; then
            cp ../data/processed-raw-int8-4x-cpu.dat "$result_dir/images/cuda.dat"
            check_image "$result_dir/images/cuda.dat"
        else
            cmp "$result_dir/images/cuda.dat" ../data/processed-raw-int8-4x-cpu.dat >> "$result_dir/image-checks.log" 2>&1
        fi
        printf 'cuda-N%s-B%s: matches CUDA baseline\n' "$threads" "$blocks" >> "$result_dir/image-checks.log"
    done
done
profile offload do_sobel_filtering ./sobel_cpu_omp_offload
cp ../data/processed-raw-int8-4x-cpu.dat "$result_dir/images/offload.dat"
check_image "$result_dir/images/offload.dat"
# The EXIT handler resumes DCGM before writing COMPLETE.
