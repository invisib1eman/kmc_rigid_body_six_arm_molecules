"""Exercise command-line validation and reported energies in isolated runs."""

import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CommandLineTests(unittest.TestCase):
    """Verify executable inputs and output consistency across short runs."""

    binary: Path

    def run_model(self, directory: str, *arguments: str) -> subprocess.CompletedProcess:
        """Run the model without exposing test outputs to a research run."""
        return subprocess.run(
            [str(self.binary), *arguments],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    def test_help(self) -> None:
        """All active energy controls are discoverable without starting a run."""
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_model(directory, "--help")
            self.assertEqual(result.returncode, 0, result.stderr)
            for option in ("bond_epsilon", "bond_cutoff", "sigma_angle",
                           "sigma_torsion", "single_partner", "rotation_step"):
                self.assertIn(option, result.stdout)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_invalid_inputs(self) -> None:
        """Invalid physics and simulation parameters fail before writing outputs."""
        invalid = [
            ("--sigma_angle", "0"),
            ("--sigma_torsion", "-1"),
            ("--sigma_angle", "nan"),
            ("--bond_epsilon", "-1"),
            ("--bond_cutoff", "0.2"),
            ("--arm_epsilon", "0"),
            ("--contact_fraction", "1"),
            ("--rotation_step", "-0.1"),
            ("--MCstep", "0"),
            ("--MCstep", "1e-200"),
            ("--MCstep", "1e200"),
            ("--time", "-1"),
            ("--NMOL", "0"),
            ("--NGRID", "0"),
            ("--box_length", "2"),
        ]
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                with tempfile.TemporaryDirectory() as directory:
                    result = self.run_model(directory, *arguments)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("Error:", result.stderr)
                    self.assertEqual(list(Path(directory).iterdir()), [])

    def test_parameterized_run(self) -> None:
        """Energy components and the MC log describe the same accepted states."""
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_model(
                directory, "--NMOL", "12", "--NGRID", "5",
                "--box_length", "12", "--time", "0.01",
                "--bond_epsilon", "7", "--sigma_angle", "0.25",
                "--sigma_torsion", "0.55", "--rotation_step", "0.07",
                "--single_partner", "0", "--GSL_SEED", "18",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            output = Path(directory)
            log = (output / "_MC.log").read_text()
            for setting in ("bond_epsilon=7", "sigma_angle=0.25", "single_partner=0"):
                self.assertIn(setting, log)
            logged_energy = {}
            for line in log.splitlines():
                if not line or line.startswith("#") or line.split()[0] == "sweep":
                    continue
                step, _, energy, acceptance = line.split()
                self.assertTrue(math.isfinite(float(energy)))
                self.assertTrue(0 <= float(acceptance) <= 1)
                logged_energy[int(step)] = float(energy)
            components = (output / "energy.txt").read_text().splitlines()
            self.assertGreater(len(components), 0)
            for index in range(0, len(components), 4):
                self.assertEqual(components[index], "TIMESTEP")
                step = int(components[index + 1])
                repulsion, attraction, total = map(float, components[index + 3].split())
                self.assertTrue(math.isclose(total, repulsion + attraction, rel_tol=1e-12, abs_tol=1e-10))
                self.assertTrue(math.isclose(total, logged_energy[step], rel_tol=1e-12, abs_tol=1e-10))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: test_cli.py /path/to/NANOROD")
    CommandLineTests.binary = Path(sys.argv.pop()).resolve()
    unittest.main()
