"""Check Great Lakes launch helpers using temporary projects and command mocks."""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = PROJECT_DIRECTORY / "artifacts" / "run"
SCRIPT_NAMES = ("run_greatlakes.sbatch", "submit_greatlakes.sh")
DEFAULT_MODULES = ["gcc/13.2.0", "boost", "gsl"]

# This recorder replaces every external build/scheduler command. It never
# compiles the model, imports cluster software, or contacts a scheduler.
MOCK_COMMAND = r'''
import json
import os
from pathlib import Path
import subprocess
import sys

kind, *arguments = sys.argv[1:]
event = {
    "command": kind,
    "arguments": arguments,
    "cwd": str(Path.cwd()),
    "module_environment": os.environ.get("GL_TEST_MODULE_ENV", ""),
    "nanorod_binary": os.environ.get("NANOROD_BINARY", ""),
}
with Path(os.environ["GL_TEST_EVENTS"]).open("a") as output:
    output.write(json.dumps(event) + "\n")

if kind == "gsl-config":
    print("-I/mock/gsl/include" if "--cflags" in arguments
          else "-L/mock/gsl/lib -lgsl -lgslcblas -lm")
elif kind == "make":
    is_clean = "clean" in arguments
    status = int(os.environ.get(
        "GL_TEST_CLEAN_EXIT" if is_clean else "GL_TEST_BUILD_EXIT", "0"))
    if not status and not is_clean:
        Path(os.environ["GL_TEST_BUILT"]).touch()
    sys.exit(status)
elif kind in ("python", "python3"):
    # Catch accidental runtime Python dependencies without executing them.
    sys.exit(95)
elif kind == "NANOROD":
    if not Path(os.environ["GL_TEST_BUILT"]).exists():
        sys.exit(91)
    sys.exit(int(os.environ.get("GL_TEST_HELP_EXIT", "0")))
elif kind == "srun":
    try:
        start = arguments.index("bash")
    except ValueError:
        sys.exit(92)
    sys.exit(subprocess.run(arguments[start:], check=False).returncode)
elif kind == "run":
    if not Path(os.environ["GL_TEST_BUILT"]).exists():
        sys.exit(93)
    if os.environ.get("GL_TEST_MODULE_ENV") != "loaded":
        sys.exit(94)
    sys.exit(int(os.environ.get("GL_TEST_RUN_EXIT", "0")))
elif kind == "sbatch":
    print("Submitted batch job 12345")
    sys.exit(int(os.environ.get("GL_TEST_SUBMIT_EXIT", "0")))
'''


class GreatLakesScriptTests(unittest.TestCase):
    """Exercise shell orchestration without using production run directories."""

    def setUp(self) -> None:
        """Create a mock toolchain and a quoted-path project for each test."""
        temporary = tempfile.TemporaryDirectory(prefix="great lakes scripts ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.project = self.root / "six arm project"
        self.run_directory = self.project / "artifacts" / "run"
        self.run_directory.mkdir(parents=True)
        for name in SCRIPT_NAMES:
            shutil.copy2(SOURCE_DIRECTORY / name, self.run_directory / name)
        shutil.copy2(
            PROJECT_DIRECTORY / "compile_greatlakes.sh",
            self.project / "compile_greatlakes.sh",
        )
        (self.project / "Makefile").write_text("# Mocked by tests.\n")
        self.events_path = self.root / "events.jsonl"
        recorder = self.root / "record command.py"
        recorder.write_text(MOCK_COMMAND, encoding="utf-8")
        self.recorder_command = (
            f"{shlex.quote(sys.executable)} {shlex.quote(str(recorder))}"
        )
        mock_bin = self.root / "mock bin"
        mock_bin.mkdir()
        for name in ("make", "g++", "gsl-config", "python", "python3", "srun", "sbatch"):
            self.write_command(mock_bin / name, name)
        self.write_command(self.project / "NANOROD", "NANOROD")
        self.write_command(self.run_directory / "run.sh", "run")
        bash_environment = self.root / "bash environment.sh"
        bash_environment.write_text(
            "module() {\n"
            f"  {self.recorder_command} module \"$@\" || return $?\n"
            "  if [[ ${1:-} == load ]]; then\n"
            "    export GL_TEST_MODULE_ENV=loaded\n"
            "  fi\n"
            "}\n",
            encoding="utf-8",
        )
        self.environment = {
            key: value for key, value in os.environ.items()
            if not key.startswith(("SLURM_", "GL_TEST_"))
            and key not in ("GL_MODULES", "NANOROD_BINARY")
        }
        self.environment.update({
            "PATH": str(mock_bin) + os.pathsep + os.environ.get("PATH", ""),
            "BASH_ENV": str(bash_environment),
            "GL_TEST_EVENTS": str(self.events_path),
            "GL_TEST_BUILT": str(self.root / "build succeeded"),
        })

    def write_command(self, path: Path, name: str) -> None:
        """Create a harmless command that records arguments and environment."""
        path.write_text(
            f"#!/usr/bin/env bash\nexec {self.recorder_command} "
            f"{shlex.quote(name)} \"$@\"\n", encoding="utf-8",
        )
        path.chmod(0o755)

    def events(self, command: str = "") -> list[dict]:
        """Read the ordered mock-command history, optionally filtering it."""
        if not self.events_path.exists():
            return []
        events = [json.loads(line) for line in self.events_path.read_text().splitlines()]
        return [event for event in events if not command or event["command"] == command]

    def run_script(
        self, script: Path, *arguments: str, environment: dict | None = None,
    ) -> subprocess.CompletedProcess:
        """Run a copied helper outside its project directory with a timeout."""
        return subprocess.run(
            ["bash", str(script), *arguments], cwd=self.root,
            env=self.environment if environment is None else environment,
            capture_output=True, text=True, timeout=15, check=False,
        )

    def allocation(self, **overrides: str) -> dict:
        """Return a small valid Slurm allocation, with optional invalid fields."""
        environment = dict(self.environment)
        environment.update({
            "SLURM_JOB_ID": "12345", "SLURM_JOB_NUM_NODES": "1",
            "SLURM_NNODES": "1", "SLURM_NTASKS": "1",
            "SLURM_CPUS_PER_TASK": "3",
        })
        environment.update(overrides)
        return environment

    def run_batch(self, **overrides: str) -> subprocess.CompletedProcess:
        """Mimic Slurm executing its spooled copy from an unrelated directory."""
        spool_directory = self.root / "slurm spool"
        spool_directory.mkdir(exist_ok=True)
        spooled_script = spool_directory / "slurm_script"
        shutil.copy2(self.run_directory / "run_greatlakes.sbatch", spooled_script)
        return self.run_script(
            spooled_script, str(self.run_directory),
            environment=self.allocation(**overrides),
        )

    def test_spooled_batch_builds_before_running_with_module_environment(self) -> None:
        """A spooled script finds the real project and retains loaded modules."""
        result = self.run_batch(NANOROD_BINARY="/old/unused/NANOROD")
        self.assertEqual(result.returncode, 0, result.stderr)
        events = self.events()
        commands = [event["command"] for event in events]
        self.assertNotIn("python", commands)
        self.assertNotIn("python3", commands)
        self.assertLess(commands.index("make"), commands.index("NANOROD"))
        self.assertLess(commands.index("NANOROD"), commands.index("srun"))
        self.assertLess(commands.index("srun"), commands.index("run"))
        loaded = [argument for event in self.events("module")
                  if event["arguments"][0] == "load"
                  for argument in event["arguments"][1:]]
        self.assertEqual(loaded, DEFAULT_MODULES)
        builds = self.events("make")
        self.assertEqual(len(builds), 2)
        for build in builds:
            arguments = build["arguments"]
            build_directory = (
                arguments[arguments.index("-C") + 1]
                if "-C" in arguments else build["cwd"]
            )
            self.assertEqual(Path(build_directory).resolve(), self.project)
        self.assertIn("clean", builds[0]["arguments"])
        build_arguments = " ".join(builds[1]["arguments"])
        self.assertRegex(build_arguments, r"-j\s*3(?:\s|$)")
        self.assertIn("CXX=g++", builds[1]["arguments"])
        self.assertIn("-I/mock/gsl/include", build_arguments)
        self.assertIn("-L/mock/gsl/lib", build_arguments)
        self.assertEqual(self.events("NANOROD")[0]["arguments"], ["--help"])
        launch = self.events("srun")[0]
        for flag in ("--ntasks=1", "--cpus-per-task=3", "--cpu-bind=cores"):
            self.assertIn(flag, launch["arguments"])
        self.assertEqual(self.events("run")[0]["module_environment"], "loaded")
        self.assertEqual(self.events("run")[0]["cwd"], str(self.run_directory))
        self.assertEqual(
            self.events("run")[0]["nanorod_binary"], str(self.project / "NANOROD"),
        )

    def test_spooled_batch_supports_direct_child_job_directory(self) -> None:
        """Job files may sit directly below the Makefile and compile helper."""
        direct_child = self.project / "job files"
        self.run_directory.rename(direct_child)
        self.run_directory = direct_child
        result = self.run_batch()
        self.assertEqual(result.returncode, 0, result.stderr)
        for build in self.events("make"):
            arguments = build["arguments"]
            self.assertEqual(arguments[arguments.index("-C") + 1], str(self.project))
        run = self.events("run")[0]
        self.assertEqual(run["cwd"], str(direct_child))
        self.assertEqual(run["nanorod_binary"], str(self.project / "NANOROD"))

    def test_batch_rejects_missing_makefile_or_root_compile_helper(self) -> None:
        """Incomplete uploads fail before loading modules or invoking make."""
        for name in ("Makefile", "compile_greatlakes.sh"):
            with self.subTest(missing=name):
                original = self.project / name
                saved = self.root / ("saved " + name)
                original.rename(saved)
                try:
                    result = self.run_batch()
                finally:
                    saved.rename(original)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(name, result.stderr)
                self.assertEqual(self.events(), [])

    def test_compile_default_parallelism_and_module_override(self) -> None:
        """The standalone helper defaults to two build workers and accepts modules."""
        environment = dict(self.environment, GL_MODULES="gcc/custom boost/custom gsl/custom")
        result = self.run_script(
            self.project / "compile_greatlakes.sh", environment=environment,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        loaded = [argument for event in self.events("module")
                  if event["arguments"][0] == "load"
                  for argument in event["arguments"][1:]]
        self.assertEqual(loaded, ["gcc/custom", "boost/custom", "gsl/custom"])
        build_arguments = " ".join(self.events("make")[-1]["arguments"])
        self.assertRegex(build_arguments, r"-j\s*2(?:\s|$)")

    def test_build_failure_never_starts_sweep(self) -> None:
        """A failed compilation propagates its status and prevents srun."""
        result = self.run_batch(GL_TEST_BUILD_EXIT="17")
        self.assertEqual(result.returncode, 17, result.stderr)
        self.assertEqual(self.events("srun"), [])
        self.assertEqual(self.events("run"), [])

    def test_sweep_failure_and_cancellation_propagate(self) -> None:
        """The batch job preserves sweep failures and interrupted exit status."""
        for status in (23, 130):
            with self.subTest(status=status):
                result = self.run_batch(GL_TEST_RUN_EXIT=str(status))
                self.assertEqual(result.returncode, status, result.stderr)

    def test_batch_requires_slurm_and_single_node_task(self) -> None:
        """Interactive or incompatible allocations cannot start simulations."""
        batch = self.run_directory / "run_greatlakes.sbatch"
        result = self.run_script(batch, str(self.run_directory))
        self.assertNotEqual(result.returncode, 0)
        for overrides in ({"SLURM_NTASKS": "2"},
                          {"SLURM_JOB_NUM_NODES": "2", "SLURM_NNODES": "2"}):
            with self.subTest(overrides=overrides):
                result = self.run_batch(**overrides)
                self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.events("srun"), [])

    def test_batch_requires_absolute_run_directory(self) -> None:
        """A spooled job cannot silently infer a project from its spool path."""
        batch = self.run_directory / "run_greatlakes.sbatch"
        for arguments in ((), ("relative/run",), (str(self.root / "missing"),)):
            with self.subTest(arguments=arguments):
                result = self.run_script(
                    batch, *arguments, environment=self.allocation(),
                )
                self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.events("make"), [])
        self.assertEqual(self.events("srun"), [])

    def test_batch_requests_the_documented_default_resources(self) -> None:
        """Default requests fit one ten-core sweep within a one-day allocation."""
        directives = {}
        for line in (self.run_directory / "run_greatlakes.sbatch").read_text().splitlines():
            if not line.startswith("#SBATCH "):
                continue
            tokens = shlex.split(line[len("#SBATCH "):])
            if "=" in tokens[0]:
                key, value = tokens[0].split("=", 1)
            else:
                key, value = tokens[0], tokens[1]
            directives[key] = value
        expected = {
            "--partition": "standard", "--nodes": "1", "--ntasks": "1",
            "--cpus-per-task": "10", "--mem": "10G",
        }
        for option, value in expected.items():
            self.assertEqual(directives[option], value)
        self.assertIn(directives["--time"], ("1-00:00:00", "24:00:00"))

    def test_submission_preserves_account_flags_and_quoted_paths(self) -> None:
        """The submit wrapper supplies paths before passing user sbatch options."""
        extra = ["--time=00:30:00", "--job-name", "six arm scan"]
        result = self.run_script(
            self.run_directory / "submit_greatlakes.sh", "research_account", *extra,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = self.events("sbatch")[0]["arguments"]
        positional = arguments[-2:]
        self.assertEqual(positional, [
            str(self.run_directory / "run_greatlakes.sbatch"), str(self.run_directory),
        ])
        self.assertEqual(arguments[-5:-2], extra)
        supplied = arguments[:-5]
        options = {}
        index = 0
        while index < len(supplied):
            option = supplied[index]
            if "=" in option:
                key, value = option.split("=", 1)
            else:
                key, value = option, supplied[index + 1]
                index += 1
            options[key] = value
            index += 1
        self.assertEqual(options["--account"], "research_account")
        self.assertEqual(options["--chdir"], str(self.run_directory))
        log_directory = self.run_directory / "logs"
        self.assertTrue(log_directory.is_dir())
        for option in ("--output", "--error"):
            self.assertEqual(Path(options[option]).parent, log_directory)
        self.assertEqual(self.events("make"), [])

    def test_submission_help_missing_account_and_failure(self) -> None:
        """Usage errors avoid submission and scheduler failures remain failures."""
        script = self.run_directory / "submit_greatlakes.sh"
        missing = self.run_script(script)
        self.assertNotEqual(missing.returncode, 0)
        help_result = self.run_script(script, "--help")
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertEqual(self.events("sbatch"), [])
        environment = dict(self.environment, GL_TEST_SUBMIT_EXIT="29")
        failed = self.run_script(script, "research_account", environment=environment)
        self.assertEqual(failed.returncode, 29, failed.stderr)


if __name__ == "__main__":
    unittest.main()
