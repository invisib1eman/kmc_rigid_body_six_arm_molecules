"""Python checks for monotone, rigid, valence-preserving minimization."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from minimize_snapshot import PairCache, SearchSettings, load_fixed_bonds, minimize_snapshot
from snapshot_energy import EnergyParameters, Snapshot, calculate_energy, load_snapshot, write_snapshot
from test_snapshot_energy import dimer


class MinimizationTests(unittest.TestCase):
    """Validate the search independently with full energy evaluations."""

    def test_energy_decreases_and_bodies_remain_rigid(self) -> None:
        source = dimer(0.38)
        source.rotate(1, [0, 0, 1], 8)
        original = source.copy()
        parameters = EnergyParameters(bond_epsilon=20)
        result, history, status = minimize_snapshot(source, parameters, SearchSettings(max_sweeps=50))
        self.assertLess(calculate_energy(result, parameters)["total_energy_kBT"],
                        calculate_energy(source, parameters)["total_energy_kBT"] - 1)
        self.assertTrue(all(right["total_energy_kBT"] <= left["total_energy_kBT"] + 1e-10
                            for left, right in zip(history, history[1:])))
        for index in range(2):
            before = np.linalg.norm(source.patches[index, :, None] - source.patches[index, None], axis=-1)
            after = np.linalg.norm(result.patches[index, :, None] - result.patches[index, None], axis=-1)
            np.testing.assert_allclose(before, after, atol=3e-14, rtol=0)
        np.testing.assert_array_equal(source.centers, original.centers)
        np.testing.assert_array_equal(source.patches, original.patches)
        self.assertTrue(calculate_energy(result, parameters)["single_partner_constraint_satisfied"])
        self.assertIn(status["reason"], ("max_sweeps", "no_improvement_at_requested_steps"))

    def test_optimal_dimer_converges_at_requested_resolution(self) -> None:
        parameters = EnergyParameters(bond_epsilon=20)
        result, history, status = minimize_snapshot(dimer(), parameters,
            SearchSettings(translation_step=0.001, rotation_step=0.1,
                           translation_tolerance=0.001, rotation_tolerance=0.1, max_sweeps=2))
        self.assertTrue(status["converged"])
        self.assertEqual(status["reason"], "no_improvement_at_requested_steps")
        self.assertAlmostEqual(calculate_energy(result, parameters)["total_energy_kBT"], -20)
        self.assertEqual(history[-1]["accepted_moves"], 0)

    def test_pair_cache_matches_full_energy_after_trial(self) -> None:
        snapshot = dimer(0.4)
        parameters = EnergyParameters(bond_epsilon=20)
        cache = PairCache(snapshot, parameters)
        before = cache.energy
        snapshot.translate(1, [-0.04, 0, -0.0188])
        delta, records, occupancy = cache.trial(1)
        self.assertAlmostEqual(before + delta, calculate_energy(snapshot, parameters)["total_energy_kBT"])
        cache.accept(records, occupancy)
        self.assertAlmostEqual(cache.energy, calculate_energy(snapshot, parameters)["total_energy_kBT"])

    def test_single_partner_and_nonfinite_inputs_rejected(self) -> None:
        source = dimer()
        invalid = Snapshot(0, source.bounds, np.r_[source.centers, source.centers[1:]],
                           np.r_[source.patches, source.patches[1:]])
        with self.assertRaises(ValueError):
            minimize_snapshot(invalid, EnergyParameters(), SearchSettings())
        # A noncoincident third molecule can still compete for a currently occupied arm.
        trial = Snapshot(0, source.bounds,
                         np.r_[source.centers, source.centers[1:] + [1, 0, 0]],
                         np.r_[source.patches, source.patches[1:] + [1, 0, 0]])
        cache = PairCache(trial, EnergyParameters())
        trial.translate(2, [-0.98, 0, 0])
        unrestricted = calculate_energy(trial, EnergyParameters(single_partner=False))
        self.assertTrue(np.isfinite(unrestricted["total_energy_kBT"]))
        self.assertTrue(unrestricted["multiple_partner_arms"])
        delta, _, _ = cache.trial(2)
        self.assertTrue(np.isinf(delta))

    def test_cli_writes_new_compatible_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "input.lammpstrj", root / "minimized run"
            write_snapshot(dimer(0.4), source)
            original = source.read_bytes()
            command = [sys.executable, str(Path(__file__).with_name("minimize_snapshot.py")), str(source),
                       "--max-sweeps", "10", "--output-dir", str(output)]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=15)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            report = json.loads((output / "result.json").read_text())
            self.assertLess(report["final"]["total_energy_kBT"], report["initial"]["total_energy_kBT"])
            self.assertFalse(report["global_minimum_guaranteed"])
            self.assertTrue((output / "initial_state.kmc").exists())
            self.assertTrue((output / "energy_history.csv").exists())
            saved = load_snapshot(output / "minimized.lammpstrj")
            self.assertEqual(len(saved.centers), 2)
            self.assertAlmostEqual(calculate_energy(saved, EnergyParameters(bond_epsilon=20))["total_energy_kBT"],
                                   report["final"]["total_energy_kBT"], places=10)
            self.assertEqual(source.read_bytes(), original)
            repeated = subprocess.run(command, capture_output=True, text=True, timeout=15)
            self.assertNotEqual(repeated.returncode, 0)

    def test_fixed_arm_pair_cannot_break_or_switch(self) -> None:
        snapshot = dimer(0.4)
        parameters = EnergyParameters(bond_epsilon=20)
        fixed = {(0, 1): (0, 0)}
        cache = PairCache(snapshot, parameters, fixed)
        snapshot.translate(1, [2, 0, 0])
        self.assertTrue(np.isinf(cache.trial(1)[0]))
        with self.assertRaises(ValueError):
            PairCache(dimer(), parameters, {(0, 1): (1, 1)})
        minimized, _, _ = minimize_snapshot(dimer(0.4), parameters,
                                            SearchSettings(max_sweeps=10), fixed_bonds=fixed)
        PairCache(minimized, parameters, fixed)

    def test_fixed_topology_rejects_extra_attraction(self) -> None:
        first = dimer()
        second = dimer()
        for index in (0, 1):
            second.translate(index, [0, 6, 0])
        snapshot = Snapshot(0, first.bounds, np.r_[first.centers, second.centers],
                            np.r_[first.patches, second.patches])
        parameters = EnergyParameters(bond_epsilon=20)
        with self.assertRaises(ValueError):
            PairCache(snapshot, parameters, {(0, 1): (0, 0)})
        PairCache(snapshot, parameters, {(0, 1): (0, 0), (2, 3): (0, 0)})

    def test_bond_strength_preparation_reaches_native_threshold(self) -> None:
        snapshot = dimer(0.4)
        parameters = EnergyParameters(bond_epsilon=20)
        bonds = {(0, 1): (0, 0)}
        initial_cache = PairCache(snapshot, parameters, bonds, 0.51, 100)
        self.assertGreater(initial_cache.records[0, 1]["bond_penalty_kBT"], 0)
        minimized, history, _ = minimize_snapshot(snapshot, parameters,
            SearchSettings(max_sweeps=40, minimum_bond_fraction=0.51, bond_penalty=100), fixed_bonds=bonds)
        final_cache = PairCache(minimized, parameters, bonds, 0.51, 100)
        self.assertEqual(final_cache.records[0, 1]["bond_penalty_kBT"], 0)
        self.assertEqual(len(calculate_energy(minimized, parameters)["geometry_contacts"]), 1)
        self.assertTrue(all(right["objective_kBT"] <= left["objective_kBT"] + 1e-10
                            for left, right in zip(history, history[1:])))

    def test_cli_fixed_bond_audit_and_mapping_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, bonds, output = root / "input", root / "bonds.json", root / "fixed run"
            write_snapshot(dimer(0.4), source)
            bond = {"ends": [[1, 0], [0, 0]]}
            bonds.write_text(json.dumps({"visual_pairings": [bond]}))
            self.assertEqual(load_fixed_bonds(bonds, 2), {(0, 1): (0, 0)})
            command = [sys.executable, str(Path(__file__).with_name("minimize_snapshot.py")), str(source),
                       "--fixed-bonds", str(bonds), "--max-sweeps", "3", "--output-dir", str(output)]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=15)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            report = json.loads((output / "result.json").read_text())
            self.assertTrue(report["fixed_arm_pairings"][0]["retained"])
            self.assertFalse(report["topology_constraints_used_in_kmc"])
            self.assertTrue((output / "bond_audit.csv").exists())
            self.assertEqual(load_fixed_bonds(output / "fixed_bonds.json", 2), {(0, 1): (0, 0)})
            bonds.write_text(json.dumps({"visual_pairings": [bond, bond]}))
            with self.assertRaises(ValueError):
                load_fixed_bonds(bonds, 2)


if __name__ == "__main__":
    unittest.main()
