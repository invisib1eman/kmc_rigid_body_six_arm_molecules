"""Python behavior checks for the standalone snapshot editor."""

import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from snapshot_energy import (BODY_VECTORS, EnergyParameters, Snapshot,
                             calculate_energy, load_snapshot, minimum_image,
                             pair_energy, write_snapshot)


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).with_name("snapshot_energy.py")
SNAPSHOT = PROJECT / "artifacts/snapshots/rotated_pair_axis_arrays/snapshot.lammpstrj"


def dimer(gap: float = 0.305638339862) -> Snapshot:
    """Make the same aligned dimer used by the C++ nonbonded tests."""
    axis = BODY_VECTORS[0] / np.linalg.norm(BODY_VECTORS[0])
    centers = np.array([[0, 0, 0], axis * (2 * np.linalg.norm(BODY_VECTORS[0]) + gap)])
    rotation = np.diag([-1, 1, -1])
    patches = np.array([BODY_VECTORS, BODY_VECTORS @ rotation.T + centers[1]])
    return Snapshot(0, np.tile([-10, 10], (3, 1)), centers, patches)


class SnapshotEditorTests(unittest.TestCase):
    """Check edits, file safety, periodicity, and model edge cases."""

    def test_aligned_and_twisted_dimer(self) -> None:
        parameters = EnergyParameters(bond_epsilon=20)
        snapshot = dimer()
        pair = pair_energy(snapshot, 0, 1, parameters)
        self.assertAlmostEqual(pair["core_repulsion_kBT"], 0)
        self.assertAlmostEqual(pair["tip_repulsion_kBT"], 0)
        self.assertAlmostEqual(pair["attraction_kBT"], -20)
        self.assertEqual(pair["best_patch_pair"], [0, 0])
        snapshot.rotate(1, BODY_VECTORS[0], math.degrees(parameters.sigma_torsion))
        twisted = calculate_energy(snapshot, parameters)
        self.assertAlmostEqual(twisted["attraction_kBT"], -20 * math.exp(-0.5), places=10)

    def test_periodic_rigid_edits_roundtrip(self) -> None:
        snapshot = dimer()
        original = snapshot.copy()
        parameters = EnergyParameters()
        snapshot.translate(1, [40, -60, 20])
        self.assertAlmostEqual(calculate_energy(snapshot, parameters)["total_energy_kBT"], -10, places=9)
        snapshot.rotate(0, [1, 2, 3], 27)
        distances = np.linalg.norm(snapshot.patches[0, :, None] - snapshot.patches[0, None], axis=-1)
        expected = np.linalg.norm(BODY_VECTORS[:, None] - BODY_VECTORS[None], axis=-1)
        np.testing.assert_allclose(distances, expected, atol=2e-14)
        np.testing.assert_allclose(original.centers, dimer().centers)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "edited.lammpstrj"
            write_snapshot(snapshot, path)
            loaded = load_snapshot(path)
            np.testing.assert_allclose(loaded.centers, minimum_image(snapshot.centers, 20), atol=1e-13)
            np.testing.assert_allclose(loaded.patches - loaded.centers[:, None],
                                       snapshot.patches - snapshot.centers[:, None], atol=1e-13)
            for key in ("core_repulsion_kBT", "tip_repulsion_kBT", "attraction_kBT"):
                self.assertAlmostEqual(calculate_energy(loaded, parameters)[key],
                                       calculate_energy(snapshot, parameters)[key], places=9)

    def test_multiple_frames_and_shuffled_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "first", Path(directory) / "second"
            snapshot = dimer()
            write_snapshot(snapshot, first)
            snapshot.timestep = 25
            snapshot.translate(1, [0.2, 0, 0])
            write_snapshot(snapshot, second)
            lines = second.read_text().splitlines()
            lines[8] = lines[8].replace("index", "id")
            second.write_text("\n".join(lines[:9] + list(reversed(lines[9:]))) + "\n")
            trajectory = Path(directory) / "trajectory"
            trajectory.write_text(first.read_text() + second.read_text())
            self.assertEqual(load_snapshot(trajectory, 0).timestep, 0)
            self.assertEqual(load_snapshot(trajectory, -2).timestep, 0)
            self.assertEqual(load_snapshot(trajectory, -1).timestep, 25)
            np.testing.assert_allclose(load_snapshot(trajectory, 1).centers, snapshot.centers)
            with self.assertRaises(ValueError):
                load_snapshot(trajectory, 2)

    def test_no_silent_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot"
            write_snapshot(dimer(), path)
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_snapshot(dimer(0.4), path)
            self.assertEqual(path.read_bytes(), original)
            write_snapshot(dimer(0.4), path, overwrite=True)
            self.assertNotEqual(path.read_bytes(), original)

    def test_center_scaling_preserves_rigid_bodies(self) -> None:
        snapshot = dimer(0.2)
        vectors = snapshot.patches - snapshot.centers[:, None]
        centers = snapshot.centers.copy()
        origin = centers.mean(axis=0)
        snapshot.scale_centers(1.15)
        np.testing.assert_allclose(snapshot.centers, origin + 1.15 * (centers - origin))
        np.testing.assert_allclose(snapshot.patches - snapshot.centers[:, None], vectors, atol=2e-15)
        self.assertAlmostEqual(np.linalg.norm(snapshot.centers[1] - snapshot.centers[0]),
                               1.15 * np.linalg.norm(centers[1] - centers[0]))
        self.assertEqual(snapshot.box_length, 20)
        for factor in (0, -1, math.inf):
            with self.assertRaises(ValueError):
                snapshot.scale_centers(factor)

    def test_invalid_dump_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot"
            write_snapshot(dimer(), path)
            text = path.read_text()
            malformed = [text.replace("pp pp pp", "ff pp pp"),
                         text.replace("-10 10", "-10 11", 1),
                         text.replace("2 2 ", "1 2 ", 1),
                         text.replace("2 2 ", "2 1 ", 1),
                         "\n".join(text.splitlines()[:-1]) + "\n"]
            for variant in malformed:
                with self.subTest(variant=variant[:80]):
                    path.write_text(variant)
                    with self.assertRaises(ValueError):
                        load_snapshot(path)

    def test_invalid_parameters_and_edits(self) -> None:
        for parameter in ({"sigma_angle": 0}, {"sigma_torsion": math.nan},
                          {"bond_cutoff": 0.1}, {"core_epsilon": -1}):
            with self.subTest(parameter=parameter), self.assertRaises(ValueError):
                EnergyParameters(**parameter)
        snapshot = dimer()
        for operation in (lambda: snapshot.translate(-1, [0, 0, 0]),
                          lambda: snapshot.translate(0, [0, math.nan, 0]),
                          lambda: snapshot.rotate(0, [0, 0, 0], 20),
                          lambda: pair_energy(snapshot, 0, 0, EnergyParameters())):
            with self.assertRaises(ValueError):
                operation()

    def test_overlaps_and_single_partner_constraint(self) -> None:
        base = dimer()
        snapshot = Snapshot(0, base.bounds, np.r_[base.centers, base.centers[1:]],
                            np.r_[base.patches, base.patches[1:]])
        report = calculate_energy(snapshot)
        self.assertTrue(math.isinf(report["total_energy_kBT"]))
        self.assertIn([0, 0], report["multiple_partner_arms"])
        self.assertFalse(report["single_partner_constraint_satisfied"])
        self.assertTrue(calculate_energy(snapshot, EnergyParameters(single_partner=False))[
            "single_partner_constraint_satisfied"])
        self.assertEqual(calculate_energy(base, EnergyParameters(bond_epsilon=0))["attraction_kBT"], 0)

    def test_cli_edits_and_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "input", Path(directory) / "new/output"
            write_snapshot(dimer(), source)
            command = [sys.executable, str(SCRIPT), str(source), "--bond_epsilon", "20",
                       "--scale-centers", "1.05", "--translate", "1", "0.1", "0", "0", "--rotate", "0", "0", "0", "1", "5",
                       "--output", str(output)]
            original = source.read_bytes()
            result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=True)
            report = json.loads(result.stdout)
            self.assertAlmostEqual(report["before"]["total_energy_kBT"], -20)
            self.assertIn("after", report)
            expected = load_snapshot(source)
            expected.scale_centers(1.05)
            expected.translate(1, [0.1, 0, 0]); expected.rotate(0, [0, 0, 1], 5)
            np.testing.assert_allclose(load_snapshot(output).patches, expected.patches, atol=1e-13)
            self.assertEqual(source.read_bytes(), original)
            rerun = subprocess.run(command, capture_output=True, text=True, timeout=15)
            self.assertNotEqual(rerun.returncode, 0)
            self.assertIn("Output exists", rerun.stderr)



if __name__ == "__main__":
    unittest.main()
