"""Check snapshot conversion and native initialization in isolated short runs."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from run_snapshot import (DEFAULT_SNAPSHOT, PROJECT, rigid_orientations,
                          write_initial_state)
from snapshot_energy import BODY_VECTORS, load_snapshot, write_snapshot


class SnapshotLaunchTests(unittest.TestCase):
    """Exercise rigid-body fitting, input validation, and actual MC startup."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.binary = PROJECT / "NANOROD"
        if not cls.binary.exists() or not DEFAULT_SNAPSHOT.exists():
            raise unittest.SkipTest("Build NANOROD and generate the expanded snapshot first")
        result = subprocess.run([str(cls.binary), "--help"], capture_output=True, text=True, timeout=10)
        if "--initial_state" not in result.stdout:
            raise unittest.SkipTest("Rebuild NANOROD to enable --initial_state")

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="snapshot launch tests ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.snapshot = load_snapshot(DEFAULT_SNAPSHOT)

    def launch(self, output: Path, *arguments: str) -> subprocess.CompletedProcess:
        """Run the Python launcher from a directory unrelated to the project."""
        return subprocess.run(
            [sys.executable, str(PROJECT / "tests/run_snapshot.py"), str(DEFAULT_SNAPSHOT),
             "--skip-build", "--time", "0.002", "--grid", "2", "--output-dir", str(output), *arguments],
            cwd=self.root, capture_output=True, text=True, timeout=30, check=False,
        )

    def test_orientations_reconstruct_source(self) -> None:
        orientations, error = rigid_orientations(self.snapshot)
        self.assertLess(error, 1e-13)
        for index, quaternion in enumerate(orientations):
            scalar, vector = quaternion[0], quaternion[1:]
            rotated = ((scalar ** 2 - vector @ vector) * BODY_VECTORS
                       + 2 * (BODY_VECTORS @ vector)[:, None] * vector
                       + 2 * scalar * np.cross(vector, BODY_VECTORS))
            np.testing.assert_allclose(rotated + self.snapshot.centers[index],
                                       self.snapshot.patches[index], atol=1e-14, rtol=0)

    def test_native_load_and_simulation(self) -> None:
        output = self.root / "simulation with spaces"
        source_hash = hashlib.sha256(DEFAULT_SNAPSHOT.read_bytes()).hexdigest()
        result = self.launch(output)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        metadata = json.loads((output / "input.json").read_text())
        self.assertEqual(metadata["status"], "finished")
        self.assertEqual(metadata["NMOL"], 18)
        self.assertEqual(metadata["box_length_nm"], 20)
        self.assertEqual(hashlib.sha256(DEFAULT_SNAPSHOT.read_bytes()).hexdigest(), source_hash)
        self.assertIn("Loaded 18 molecules", (output / "simulation.log").read_text())
        initial = load_snapshot(output / "nanorod_Dump.lammpstrj")
        final = load_snapshot(output / "nanorod_Dump.lammpstrj", -1)
        np.testing.assert_allclose(initial.centers, self.snapshot.centers, atol=6e-6, rtol=0)
        np.testing.assert_allclose(initial.patches, self.snapshot.patches, atol=6e-6, rtol=0)
        self.assertGreater(final.timestep, initial.timestep)
        self.assertFalse(np.allclose(final.centers, initial.centers))
        repeated = self.launch(output)
        self.assertNotEqual(repeated.returncode, 0)
        self.assertIn("already exists", repeated.stderr)

    def test_prepare_only_does_not_run(self) -> None:
        output = self.root / "prepared"
        result = self.launch(output, "--prepare-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((output / "initial_state.kmc").exists())
        self.assertEqual(json.loads((output / "input.json").read_text())["status"], "prepared")
        self.assertFalse((output / "simulation.log").exists())

    def test_distorted_molecule_is_rejected(self) -> None:
        self.snapshot.patches[0, 0] += [0.1, 0, 0]
        with self.assertRaises(ValueError):
            rigid_orientations(self.snapshot)
        path = self.root / "distorted.lammpstrj"
        write_snapshot(self.snapshot, path)
        output = self.root / "not created"
        result = subprocess.run([sys.executable, str(PROJECT / "tests/run_snapshot.py"), str(path),
                                 "--skip-build", "--output-dir", str(output)],
                                capture_output=True, text=True, timeout=15)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not a KMC rigid body", result.stderr)
        self.assertFalse(output.exists())

    def test_native_reader_rejects_bad_states_and_conflicts(self) -> None:
        path = self.root / "state.kmc"
        quaternions, _ = rigid_orientations(self.snapshot)
        write_initial_state(self.snapshot, quaternions, path)
        valid = path.read_text()
        rows = valid.splitlines()
        duplicate = rows.copy()
        duplicate[3] = "0 " + duplicate[3].split(" ", 1)[1]
        bad_quaternion = rows.copy()
        values = bad_quaternion[2].split(); values[4:] = ["0", "0", "0", "0"]
        bad_quaternion[2] = " ".join(values)
        cases = [
            (valid.replace("KMC_RIGID_SNAPSHOT 1", "KMC_RIGID_SNAPSHOT 9"), []),
            ("\n".join(duplicate) + "\n", []),
            ("\n".join(bad_quaternion) + "\n", []),
            ("\n".join(rows[:-1]) + "\n", []),
            (valid + "unexpected\n", []),
            (valid, ["--NMOL", "19"]),
            (valid, ["--box_length", "21"]),
        ]
        for index, (content, options) in enumerate(cases):
            with self.subTest(case=index):
                path.write_text(content)
                output = self.root / f"invalid_{index}"; output.mkdir()
                result = subprocess.run([str(self.binary), "--initial_state", str(path),
                                         "--time", "0", "--NGRID", "2", *options],
                                        cwd=output, capture_output=True, text=True, timeout=10)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Error:", result.stderr)
                self.assertEqual(list(output.iterdir()), [])

    def test_native_reader_accepts_shuffled_rows(self) -> None:
        path = self.root / "shuffled.kmc"
        quaternions, _ = rigid_orientations(self.snapshot)
        write_initial_state(self.snapshot, quaternions, path)
        lines = path.read_text().splitlines()
        path.write_text("\n".join(lines[:2] + list(reversed(lines[2:]))) + "\n")
        output = self.root / "shuffled run"; output.mkdir()
        result = subprocess.run([str(self.binary), "--initial_state", str(path),
                                 "--NMOL", "18", "--box_length", "20", "--time", "0", "--NGRID", "2"],
                                cwd=output, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        loaded = load_snapshot(output / "nanorod_Dump.lammpstrj")
        np.testing.assert_allclose(loaded.centers, self.snapshot.centers, atol=6e-6, rtol=0)


if __name__ == "__main__":
    unittest.main()
