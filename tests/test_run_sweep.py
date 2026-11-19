"""Check the sweep launcher with short fake simulations in temporary folders."""

import itertools
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest


RUN_SCRIPT = Path(__file__).resolve().parents[1] / "artifacts/run/run.sh"
COMMON_PARAMETERS = {
    "NMOL": 100,
    "box_length": 40,
    "time": 10000,
    "bond_epsilon": 20,
    "bond_r0": 0.305638339862,
    "bond_cutoff": 0.5,
    "MCstep": 0.1,
    "rotation_step": 0.1,
    "GSL_SEED": 10,
}
SIGMA_PAIRS = set(itertools.product(
    map(math.radians, (4, 8, 12, 16, 20)),
    map(math.radians, (2, 6)),
))
FAKE_BINARY = """\
import json
import math
import os
from pathlib import Path
import sys
import time

started = time.monotonic_ns()
arguments = sys.argv[1:]
parameters = dict(zip(arguments[::2], arguments[1::2]))
pair = (float(parameters["--sigma_angle"]), float(parameters["--sigma_torsion"]))
inputs_available = Path("input.json").is_file()
Path("started.json").write_text(json.dumps({"pid": os.getpid()}))
with Path("energy.txt").open("a") as stream:
    stream.write(f"{pair[0]} {pair[1]}\\n")
print("simulation stdout")
print("simulation stderr", file=sys.stderr)
delay = float(os.environ.get("SWEEP_TEST_DELAY", "0.15"))
if os.environ.get("SWEEP_TEST_STAGGERED") == "1":
    first_case = (math.isclose(pair[0], math.radians(4), abs_tol=1e-15)
                  and math.isclose(pair[1], math.radians(2), abs_tol=1e-15))
    delay = 1.2 if first_case else 0.06
time.sleep(delay)
Path("observed.json").write_text(json.dumps({
    "binary": str(Path(sys.argv[0]).resolve()),
    "arguments": arguments,
    "cwd": str(Path.cwd()),
    "inputs_available": inputs_available,
    "started": started,
    "finished": time.monotonic_ns(),
}))
if (os.environ.get("SWEEP_TEST_FAIL") == "1"
        and math.isclose(pair[0], math.radians(8), abs_tol=1e-15)
        and math.isclose(pair[1], math.radians(2), abs_tol=1e-15)):
    print("intentional simulation failure", file=sys.stderr)
    raise SystemExit(7)
"""


class SweepLauncherTests(unittest.TestCase):
    """Verify input preservation, isolation, bounded parallelism, and failures."""

    def setUp(self) -> None:
        """Reproduce the executable/script layout without touching real outputs."""
        temporary = tempfile.TemporaryDirectory(prefix="nanorod sweep tests ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.project = self.root / "project with spaces"
        self.run_directory = self.project / "artifacts/run"
        self.run_directory.mkdir(parents=True)
        self.script = self.run_directory / "run.sh"
        shutil.copy2(RUN_SCRIPT, self.script)
        self.binary = self.project / "NANOROD"
        self.binary.write_text(f"#!{sys.executable}\n" + textwrap.dedent(FAKE_BINARY))
        self.binary.chmod(0o755)
        self.working_directory = self.root / "unrelated working directory"
        self.working_directory.mkdir()
        executables = self.root / "executables"
        executables.mkdir()
        nproc = executables / "nproc"
        nproc.write_text('#!/bin/sh\nprintf "%s\\n" "${SWEEP_TEST_VISIBLE_CPUS:-2}"\n')
        nproc.chmod(0o755)
        for program in ("python", "python3"):
            forbidden_interpreter = executables / program
            forbidden_interpreter.write_text(
                '#!/bin/sh\nprintf "%s\\n" "$0" >> "$SWEEP_TEST_PYTHON_CALLS"\n'
                'printf "Launcher must not invoke Python\\n" >&2\nexit 97\n'
            )
            forbidden_interpreter.chmod(0o755)
        self.environment = os.environ.copy()
        self.environment["PATH"] = str(executables) + os.pathsep + os.environ["PATH"]
        self.python_calls = self.root / "unexpected python calls.txt"
        self.environment["SWEEP_TEST_PYTHON_CALLS"] = str(self.python_calls)
        self.environment.pop("SWEEP_TEST_FAIL", None)
        self.environment.pop("SWEEP_TEST_DELAY", None)
        self.environment.pop("SWEEP_TEST_STAGGERED", None)
        self.environment.pop("SWEEP_TEST_VISIBLE_CPUS", None)
        self.environment.pop("SLURM_CPUS_PER_TASK", None)
        self.environment.pop("NANOROD_BINARY", None)

    def tearDown(self) -> None:
        """Every launcher path must work without invoking Python."""
        self.assertFalse(self.python_calls.exists(), "Launcher invoked Python")

    def run_sweep(self) -> subprocess.CompletedProcess:
        """Launch through the system Bash from a different working directory."""
        return subprocess.run(
            ["/bin/bash", str(self.script)],
            cwd=self.working_directory,
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    def check_sweep(self, sweep_directory: Path) -> None:
        """Assert every requested pair got exactly its own inputs and outputs."""
        run_directories = sorted(path for path in sweep_directory.iterdir()
                                 if path.is_dir())
        self.assertEqual(len(run_directories), len(SIGMA_PAIRS))
        observed_pairs = set()
        timeline = []
        for directory in run_directories:
            with self.subTest(run=directory.name):
                metadata = json.loads((directory / "input.json").read_text())
                observed = json.loads((directory / "observed.json").read_text())
                parameters = metadata["parameters"]
                arguments = observed["arguments"]
                argument_values = dict(zip(arguments[::2], arguments[1::2]))
                pair = (parameters["sigma_angle"], parameters["sigma_torsion"])
                observed_pairs.add(pair)
                self.assertEqual(parameters, {
                    **COMMON_PARAMETERS,
                    "sigma_angle": pair[0],
                    "sigma_torsion": pair[1],
                })
                self.assertEqual(
                    directory.name,
                    f"sigma_angle_{argument_values['--sigma_angle']}_"
                    f"sigma_torsion_{argument_values['--sigma_torsion']}",
                )
                self.assertEqual(metadata["sigma_units"], "radians")
                self.assertEqual(metadata["available_cpu_cores"], 2)
                self.assertEqual(metadata["parallel_runs"], 2)
                self.assertEqual(metadata["command"][0], str(self.binary))
                self.assertEqual(observed["binary"], str(self.binary))
                self.assertEqual(metadata["command"][1:], observed["arguments"])
                self.assertEqual(len(arguments), 2 * len(parameters))
                parsed = {key[2:]: float(value)
                          for key, value in zip(arguments[::2], arguments[1::2])}
                self.assertEqual(parsed, parameters)
                self.assertEqual(Path(observed["cwd"]), directory)
                self.assertTrue(observed["inputs_available"])
                self.assertEqual(
                    (directory / "energy.txt").read_text(),
                    f"{float(pair[0])} {float(pair[1])}\n",
                )
                log = (directory / "simulation.log").read_text()
                self.assertIn("simulation stdout", log)
                self.assertIn("simulation stderr", log)
                timeline.extend(((observed["started"], 1),
                                 (observed["finished"], -1)))
        self.assertEqual(len(observed_pairs), len(SIGMA_PAIRS))
        for observed, expected in zip(sorted(observed_pairs), sorted(SIGMA_PAIRS)):
            self.assertAlmostEqual(observed[0], expected[0], delta=1e-15)
            self.assertAlmostEqual(observed[1], expected[1], delta=1e-15)
        active = peak = 0
        for _, change in sorted(timeline):
            active += change
            peak = max(peak, active)
        self.assertEqual(active, 0)
        self.assertEqual(peak, 2, "Use both cores without oversubscribing them")
        self.assertEqual(list(self.working_directory.iterdir()), [])

    def test_complete_sweep_and_safe_rerun(self) -> None:
        """All pairs run with preserved arguments and fresh output directories."""
        syntax = subprocess.run(["/bin/bash", "-n", str(self.script)],
                                capture_output=True, text=True, check=False)
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        for expected_count in (1, 2):
            result = self.run_sweep()
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            sweeps = list(self.run_directory.glob("sweep_*"))
            self.assertEqual(len(sweeps), expected_count)
            for sweep in sweeps:
                self.check_sweep(sweep)

    def test_failed_case_does_not_skip_other_cases(self) -> None:
        """A failed simulation is logged and produces a failing launcher status."""
        self.environment["SWEEP_TEST_FAIL"] = "1"
        result = self.run_sweep()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        sweeps = list(self.run_directory.glob("sweep_*"))
        self.assertEqual(len(sweeps), 1)
        self.check_sweep(sweeps[0])
        failed_logs = [path for path in sweeps[0].glob("*/simulation.log")
                       if "intentional simulation failure" in path.read_text()]
        self.assertEqual(len(failed_logs), 1)

    def test_slurm_allocation_caps_visible_cores(self) -> None:
        """The launcher uses its allocation even if CPU affinity is broader."""
        self.environment["SWEEP_TEST_VISIBLE_CPUS"] = "8"
        self.environment["SLURM_CPUS_PER_TASK"] = "2"
        result = self.run_sweep()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        sweeps = list(self.run_directory.glob("sweep_*"))
        self.assertEqual(len(sweeps), 1)
        self.check_sweep(sweeps[0])

    def test_invalid_slurm_allocation_fails_before_outputs(self) -> None:
        """Invalid CPU limits cannot silently bypass scheduler allocations."""
        for allocation in ("", "0", "-2", "2.5", "abc"):
            with self.subTest(allocation=allocation):
                self.environment["SLURM_CPUS_PER_TASK"] = allocation
                result = self.run_sweep()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("SLURM_CPUS_PER_TASK", result.stdout + result.stderr)
                self.assertEqual(list(self.run_directory.glob("sweep_*")), [])

    def test_missing_binary_fails_before_creating_outputs(self) -> None:
        """A missing executable fails clearly without making empty run folders."""
        self.binary.unlink()
        result = self.run_sweep()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("NANOROD", result.stdout + result.stderr)
        self.assertEqual(list(self.run_directory.glob("sweep_*")), [])

    def test_explicit_binary_path_runs_every_case(self) -> None:
        """A separately compiled executable is used and recorded for all cases."""
        original_binary = self.binary
        build_directory = self.root / "separate build directory"
        build_directory.mkdir()
        self.binary = original_binary.rename(build_directory / "NANOROD")
        self.environment["NANOROD_BINARY"] = str(self.binary)
        self.assertFalse(original_binary.exists())
        result = self.run_sweep()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        sweeps = list(self.run_directory.glob("sweep_*"))
        self.assertEqual(len(sweeps), 1)
        self.check_sweep(sweeps[0])

    def test_invalid_explicit_binary_does_not_fall_back(self) -> None:
        """A bad explicit path fails even when the default executable exists."""
        missing_binary = self.root / "missing build/NANOROD"
        self.environment["NANOROD_BINARY"] = str(missing_binary)
        self.assertTrue(self.binary.is_file())
        result = self.run_sweep()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(str(missing_binary), result.stdout + result.stderr)
        self.assertEqual(list(self.run_directory.glob("sweep_*")), [])

    def test_json_preserves_quotes_and_backslashes_in_paths(self) -> None:
        """Shell-written metadata correctly escapes unusual valid paths."""
        self.project = self.project.rename(self.root / 'project "quoted" with \\ path')
        self.run_directory = self.project / "artifacts/run"
        self.script = self.run_directory / "run.sh"
        self.binary = self.project / "NANOROD"
        result = self.run_sweep()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        sweeps = list(self.run_directory.glob("sweep_*"))
        self.assertEqual(len(sweeps), 1)
        self.check_sweep(sweeps[0])

    def test_scheduler_fills_freed_slots_within_limit(self) -> None:
        """A long first case cannot block the queue after another case finishes."""
        self.environment["SWEEP_TEST_STAGGERED"] = "1"
        result = self.run_sweep()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        sweeps = list(self.run_directory.glob("sweep_*"))
        self.assertEqual(len(sweeps), 1)
        self.check_sweep(sweeps[0])
        observations = [json.loads(path.read_text())
                        for path in sweeps[0].glob("*/observed.json")]
        slow_case = None
        for observation in observations:
            arguments = observation["arguments"]
            values = dict(zip(arguments[::2], arguments[1::2]))
            if (math.isclose(float(values["--sigma_angle"]), math.radians(4))
                    and math.isclose(float(values["--sigma_torsion"]), math.radians(2))):
                slow_case = observation
                break
        self.assertIsNotNone(slow_case)
        later_starts = sum(observation["started"] < slow_case["finished"]
                           for observation in observations if observation is not slow_case)
        self.assertGreater(later_starts, 1, "A free worker must take queued cases")

    def test_interrupt_stops_active_and_pending_cases(self) -> None:
        """Interrupting only the launcher stops its children and queued cases."""
        self.check_cancellation(signal.SIGINT)

    def test_sigterm_stops_active_and_pending_cases(self) -> None:
        """Scheduler termination stops active children without launching more."""
        self.check_cancellation(signal.SIGTERM)

    def check_cancellation(self, signum: int) -> None:
        """Verify a launcher-only signal promptly stops its full workload."""
        self.environment["SWEEP_TEST_DELAY"] = "60"
        process = subprocess.Popen(
            ["/bin/bash", str(self.script)],
            cwd=self.working_directory,
            env=self.environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 10
            started_files = []
            while time.monotonic() < deadline:
                started_files = list(self.run_directory.glob(
                    "sweep_*/sigma_*/started.json"
                ))
                if len(started_files) == 2:
                    break
                time.sleep(0.02)
            self.assertEqual(len(started_files), 2)
            process.send_signal(signum)
            stdout, stderr = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 128 + signum, stdout + stderr)
            run_directories = list(self.run_directory.glob("sweep_*/sigma_*"))
            self.assertEqual(len(run_directories), 2, "Queued cases must not start")
            for started_file in started_files:
                pid = json.loads(started_file.read_text())["pid"]
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)
        finally:
            # The isolated group contains only this test launcher and its fakes.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate(timeout=5)


if __name__ == "__main__":
    unittest.main()
