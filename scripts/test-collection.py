#!/usr/bin/env python3
"""Offline integration checks: synthetic tools/data, no Slurm/GPU measurements."""
import csv
import importlib.util
from itertools import product
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("extract_results", SCRIPTS / "extract-results.py")
extractor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extractor)

MOCK = r'''#!/usr/bin/env python3
import csv, hashlib, json, os, pathlib, sys
tool = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
def record(kind, **values):
    with open(os.environ['MOCK_CALLS'], 'a') as stream:
        stream.write(json.dumps(dict(kind=kind, **values)) + '\n')
def write_image():
    path = pathlib.Path('../data/processed-raw-int8-4x-cpu.dat')
    with path.open('wb') as stream:
        stream.truncate(36598352)
        stream.seek(7113)
        stream.write(b'\x01')
    print(' Read data from the file ../data/zebra-gray-int8-4x')
    print(' Wrote the output file ../data/processed-raw-int8-4x-cpu.dat')
if tool == 'git':
    if 'rev-parse' in args:
        print('f17f7505db5e03cf39aa37725ec30f01f6be938f')
elif tool == 'sha256sum':
    for name in args:
        print(hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest(), ' ', name, sep='')
elif tool == 'awk':
    if '/proc/self/status' in args:
        print('Cpus_allowed_list: 0-31')
    else:
        os.execv('/usr/bin/awk', ['awk', *args])
elif tool == 'srun':
    while args and args[0].startswith('-'):
        flag = args.pop(0)
        if flag in ('-n', '-c'):
            args.pop(0)
    os.execvpe(args[0], args, os.environ)
elif tool == 'sobel_cpu':
    threads = int(os.environ['OMP_NUM_THREADS'])
    record('cpu', threads=threads, places=os.environ.get('OMP_PLACES'), bind=os.environ.get('OMP_PROC_BIND'))
    if os.environ.get('FAIL_CPU_THREADS') == str(threads):
        print('injected CPU failure')
        sys.exit(7)
    write_image()
    print(' Elapsed time is :', 0.12 / threads)
elif tool == 'ncu' and '--version' not in args:
    def value(flag):
        return args[args.index(flag) + 1]
    prefix = pathlib.Path(value('--export'))
    name = prefix.name
    record('profile', name=name, args=args, mandatory=os.environ.get('OMP_TARGET_OFFLOAD'))
    if os.environ.get('FAIL_CASE') == name:
        pathlib.Path(value('--log-file')).write_text('==ERROR== injected profiler failure\n')
        sys.exit(9)
    metrics = value('--metrics').split(',')
    assert metrics == ['gpu__time_duration.avg', 'sm__warps_active.avg.pct_of_peak_sustained_active', 'dram__throughput.avg.pct_of_peak_sustained_elapsed']
    assert value('--page') == 'raw' and value('--print-units') == 'base'
    assert value('--launch-count') == '1'
    kernel = 'sobel_kernel_gpu(float *, float *)' if name.startswith('cuda-') else 'nvkernel__Z18do_sobel_filteringPfS_ii_F1L114_2'
    with open(value('--log-file'), 'w') as stream:
        stream.write('==PROF== synthetic fixture\n')
        writer = csv.writer(stream)
        writer.writerow(['ID', 'Kernel Name', 'Metric Name', 'Metric Unit', 'Metric Value'])
        for metric, unit, val in zip(metrics, ['nsecond', '%', '%'], ['123,456.789123', '12.345678901234', '23.456789012345']):
            if os.environ.get('OMIT_DRAM_CASE') == name and metric.startswith('dram__'):
                continue
            writer.writerow(['0', kernel, metric, unit, val])
    prefix.with_suffix('.ncu-rep').write_bytes(b'SYNTHETIC REPORT, NOT A MEASUREMENT')
    if name.startswith('cuda-'):
        print(' GPU configuration:', value('-B'), 'blocks,', value('-N'), 'threads per block')
    else:
        print(' Elapsed time is : 1.5')
    write_image()
elif tool == 'dcgmi':
    operation = 'resume' if '--resume' in args else 'pause'
    record('dcgm', operation=operation)
    if operation == 'resume' and os.environ.get('FAIL_RESUME'):
        print('injected resume failure')
        sys.exit(6)
    print('Successfully ' + ('resumed' if operation == 'resume' else 'paused') + ' profiling.')
else:
    print('synthetic', tool, *args)
'''


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hw5-collection-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.build = self.repo / "build"
        self.build.mkdir(parents=True)
        self.results = self.root / "results"
        self.calls = self.root / "calls.jsonl"
        for name in ("sobel_cpu.cpp", "sobel_gpu.cu", "sobel_cpu_omp_offload.cpp", "CMakeLists.txt"):
            shutil.copy2(REPO / name, self.repo / name)
        (self.repo / 'scripts').mkdir()
        shutil.copy2(SCRIPTS / 'extract-results.py', self.repo / 'scripts/extract-results.py')
        (self.repo / "data").mkdir()
        (self.repo / "data/zebra-gray-int8-4x").write_bytes(b"synthetic input")
        (self.build / "CMakeCache.txt").write_text("synthetic build\n")
        compiler_dir = self.build / "CMakeFiles/3.28.3"
        compiler_dir.mkdir(parents=True)
        for lang in ("CXX", "CUDA"):
            (compiler_dir / f"CMake{lang}Compiler.cmake").write_text("synthetic compiler\n")
        for target in ("sobel_cpu", "sobel_gpu", "sobel_cpu_omp_offload"):
            directory = self.build / f"CMakeFiles/{target}.dir"
            directory.mkdir()
            for name in ("flags.make", "link.txt"):
                (directory / name).write_text("synthetic flags\n")
        self.bin = self.root / "bin"
        self.bin.mkdir()
        dispatcher = self.bin / "dispatcher"
        dispatcher.write_text(MOCK)
        dispatcher.chmod(0o755)
        for name in ("git", "sha256sum", "awk", "srun", "scontrol", "lscpu", "nvidia-smi", "ncu", "nvcc", "dcgmi"):
            (self.bin / name).symlink_to(dispatcher)
        for name in ("sobel_cpu", "sobel_gpu", "sobel_cpu_omp_offload"):
            (self.build / name).symlink_to(dispatcher)
        shell_env = self.root / "bash-env"
        shell_env.write_text('module() { printf "synthetic module %s\\n" "$*"; }\n')
        self.env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ['PATH'],
                        BASH_ENV=str(shell_env), SCRATCH=str(self.root), SLURM_JOB_ID="12345",
                        SLURM_CPUS_PER_TASK="32", MOCK_CALLS=str(self.calls))

    def run_job(self, kind, **overrides):
        return subprocess.run(['bash', str(SCRIPTS / f'collect-{kind}.sbatch'), str(self.build), str(self.results)],
                              env=dict(self.env, **overrides), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def events(self, kind):
        return [event for line in self.calls.read_text().splitlines()
                if (event := json.loads(line))['kind'] == kind]

    def test_complete_grid_and_precise_extraction(self):
        for kind in ('cpu', 'gpu'):
            result = self.run_job(kind)
            self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual([event['threads'] for event in self.events('cpu')], [1, 2, 4, 8, 16])
        self.assertTrue(all(event['places'] == 'cores' and event['bind'] == 'spread' for event in self.events('cpu')))
        names = [event['name'] for event in self.events('profile')]
        expected = [f'cuda-N{t}-B{b}' for t, b in product((32,64,128,256,512,1024), (1,4,16,64,256,1024,4096))] + ['offload']
        self.assertEqual(names, expected)
        self.assertTrue(all(event['mandatory'] == 'MANDATORY' for event in self.events('profile')))
        self.assertEqual([event['operation'] for event in self.events('dcgm')], ['pause', 'resume'])
        output = self.root / 'derived'
        extractor.extract(self.results, output)
        with (output / 'cuda.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 42)
        self.assertEqual(rows[0]['runtime_ms'], '0.123456789123')
        self.assertEqual(rows[0]['achieved_occupancy_pct'], '12.345678901234')
        with (output / 'cpu.csv').open() as stream:
            self.assertEqual(next(csv.DictReader(stream))['runtime_ms'], '120.00')
        # Missing coverage must fail before writing any derived output.
        (self.results / 'gpu/cuda-N64-B4.csv').unlink()
        with self.assertRaisesRegex(ValueError, 'coverage'):
            extractor.extract(self.results, self.root / 'bad-derived')
        self.assertFalse((self.root / 'bad-derived').exists())
        self.assertNotEqual(self.run_job('cpu').returncode, 0)  # No overwrite.

    def test_cpu_failure_stops_and_leaves_no_completion_marker(self):
        result = self.run_job('cpu', FAIL_CPU_THREADS='4')
        self.assertEqual(result.returncode, 7, result.stdout)
        self.assertEqual([event['threads'] for event in self.events('cpu')], [1, 2, 4])
        self.assertFalse((self.results / 'cpu/COMPLETE').exists())
        self.assertEqual((self.results / 'cpu/cpu-t4.exit-status').read_text().strip(), '7')

    def test_gpu_failure_stops_and_resumes_dcgm(self):
        self.assertEqual(self.run_job('cpu').returncode, 0)
        result = self.run_job('gpu', FAIL_CASE='cuda-N64-B4')
        self.assertEqual(result.returncode, 9, result.stdout)
        self.assertEqual(len(self.events('profile')), 9)
        self.assertNotIn('offload', [event['name'] for event in self.events('profile')])
        self.assertEqual([event['operation'] for event in self.events('dcgm')], ['pause', 'resume'])
        self.assertFalse((self.results / 'gpu/COMPLETE').exists())

    def test_resume_failure_prevents_completion(self):
        self.assertEqual(self.run_job('cpu').returncode, 0)
        result = self.run_job('gpu', FAIL_RESUME='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.results / 'gpu/COMPLETE').exists())

    def test_missing_metric_stops_first_profile_and_resumes_dcgm(self):
        self.assertEqual(self.run_job('cpu').returncode, 0)
        result = self.run_job('gpu', OMIT_DRAM_CASE='cuda-N32-B1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('missing required metrics: dram__throughput', result.stdout)
        self.assertEqual([event['name'] for event in self.events('profile')], ['cuda-N32-B1'])
        self.assertEqual([event['operation'] for event in self.events('dcgm')], ['pause', 'resume'])
        self.assertFalse((self.results / 'gpu/COMPLETE').exists())

    def test_wide_csv_and_invalid_metrics(self):
        path = self.root / 'wide.csv'
        with path.open('w') as stream:
            writer = csv.writer(stream)
            writer.writerow(['ID', 'Kernel Name', *extractor.METRICS])
            writer.writerow(['', '', 'usecond', '%', '%'])
            writer.writerow(['0', 'sobel_kernel_gpu', '505.41', '0', '27.50'])
        self.assertEqual(extractor.profile(path, 'cuda')['runtime_ms'], '0.50541')
        with self.assertRaisesRegex(ValueError, 'wrong kernel'):
            extractor.profile(path, 'offload')
        original = path.read_text()
        for old, new in [('usecond', 'cycle'), ('505.41', 'nan'), (extractor.OCCUPANCY, 'smsp__cycles_active.avg.pct_of_peak_sustained_elapsed')]:
            path.write_text(original.replace(old, new))
            with self.assertRaises(ValueError):
                extractor.profile(path, 'cuda')

    def test_long_csv_rejects_duplicate_launch_and_profiler_error(self):
        path = self.root / 'long.csv'
        with path.open('w') as stream:
            writer = csv.writer(stream)
            writer.writerow(['ID', 'Kernel Name', 'Metric Name', 'Metric Unit', 'Metric Value'])
            for metric, unit, value in zip(extractor.METRICS, ['msecond', '%', '%'], ['2.66', '12.43', '5.23']):
                writer.writerow(['0', 'sobel_kernel_gpu', metric, unit, value])
        self.assertEqual(extractor.profile(path, 'cuda')['runtime_ms'], '2.66')
        original = path.read_text()
        with path.open('a') as stream:
            csv.writer(stream).writerow(['1', 'setup_kernel', 'other', 'byte', '1'])
        with self.assertRaisesRegex(ValueError, 'one profiled kernel'):
            extractor.profile(path, 'cuda')
        path.write_text(original + '==ERROR== failed counter collection\n')
        with self.assertRaisesRegex(ValueError, 'error marker'):
            extractor.profile(path, 'cuda')

    def test_saved_detailed_report_missing_dram_has_precise_error(self):
        # Target export: valid wide CSV, ns timing and occupancy, but no exact
        # HW5 DRAM metric. Never substitute an available overview metric.
        path = self.root / 'saved-detailed.csv'
        with path.open('w') as stream:
            writer = csv.writer(stream)
            writer.writerow(['ID', 'Kernel Name', extractor.TIME, extractor.OCCUPANCY])
            writer.writerow(['', '', 'ns', '%'])
            writer.writerow(['0', 'nvkernel__Z18do_sobel_filteringPfS_ii_F1L114_2', '505,408.00', '55.26'])
        with self.assertRaisesRegex(ValueError, 'missing required metrics: dram__throughput'):
            extractor.profile(path, 'offload')


if __name__ == '__main__':
    unittest.main()
