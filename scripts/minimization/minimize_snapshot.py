"""Locally minimize a snapshot with energy-lowering rigid-body pattern search.

Requires NumPy only. Uses snapshot_energy.py without changing the KMC potential.
Optional fixed arm pairings and preparation-only bond-strength penalties can
preserve a supplied topology; the physical and restrained energies are separate.
Each trial translates or rotates one whole molecule; the single-partner hard
constraint is enforced. This is a local minimum at a specified step resolution,
not a guarantee of the global ground state or an assembly topology constraint.
"""

import argparse
import csv
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
from typing import Callable

import numpy as np

from run_snapshot import DEFAULT_SNAPSHOT, PROJECT, rigid_orientations, write_initial_state
from snapshot_energy import (EnergyParameters, Snapshot, calculate_energy,
                             load_snapshot, pair_energy, write_snapshot)


@dataclass(frozen=True)
class SearchSettings:
    """Translation steps are nm and rotation steps are degrees."""

    translation_step: float = 0.03
    rotation_step: float = 2.0
    translation_tolerance: float = 1e-4
    rotation_tolerance: float = 0.01
    energy_tolerance: float = 1e-9
    max_sweeps: int = 500
    max_seconds: float | None = None
    seed: int = 10
    pattern_steps: int = 3
    minimum_bond_fraction: float = 0.0
    bond_penalty: float = 100.0

    def __post_init__(self) -> None:
        for name in ("translation_step", "rotation_step", "translation_tolerance",
                     "rotation_tolerance", "energy_tolerance"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.translation_step < self.translation_tolerance or self.rotation_step < self.rotation_tolerance:
            raise ValueError("Initial steps cannot be smaller than the requested tolerances")
        if self.max_sweeps < 1 or self.seed < 0:
            raise ValueError("max_sweeps must be positive and seed nonnegative")
        if not 0 <= self.pattern_steps <= 10:
            raise ValueError("pattern_steps must lie between 0 and 10")
        if not 0 <= self.minimum_bond_fraction < 1 or not math.isfinite(self.bond_penalty) or self.bond_penalty < 0:
            raise ValueError("Require 0 <= minimum_bond_fraction < 1 and a nonnegative finite bond_penalty")
        if self.max_seconds is not None and (not math.isfinite(self.max_seconds) or self.max_seconds <= 0):
            raise ValueError("max_seconds must be finite and positive")


class PairCache:
    """Cache unique pair energies and valence counts for single-body trials."""

    def __init__(self, snapshot: Snapshot, parameters: EnergyParameters,
                 fixed_bonds: dict | None = None, minimum_bond_fraction: float = 0.0,
                 bond_penalty: float = 100.0):
        self.snapshot = snapshot
        self.parameters = parameters
        self.fixed_bonds = fixed_bonds
        self.minimum_bond_fraction = minimum_bond_fraction
        self.bond_penalty = bond_penalty
        self.records = {}
        self.occupancy = np.zeros((len(snapshot.centers), 6), dtype=int)
        self.evaluations = 0
        for first in range(len(snapshot.centers)):
            for second in range(first + 1, len(snapshot.centers)):
                key = (first, second)
                self.records[key] = self.evaluate_pair(key)
                if not self.valid_topology(key, self.records[key]):
                    raise ValueError(f"Initial state does not preserve the specified arm pairing for molecules {key}")
                self.update_occupancy(self.occupancy, key, self.records[key], 1)
        self.energy = self.total()
        if not math.isfinite(self.energy):
            raise ValueError("Initial energy must be finite; separate exactly coincident sites first")
        if parameters.single_partner and np.any(self.occupancy > 1):
            raise ValueError("Initial state violates single_partner; correct its contacts before minimizing")

    def valid_topology(self, key: tuple, record: dict) -> bool:
        """Keep the specified attractive arm pairs and exclude new attractive pairs."""
        if self.fixed_bonds is None:
            return True
        if key in self.fixed_bonds:
            return (record["attraction_kBT"] < 0
                    and record["best_patch_pair"] == list(self.fixed_bonds[key])
                    and (self.bond_penalty > 0 or record["attraction_kBT"] <=
                         -self.minimum_bond_fraction * self.parameters.bond_epsilon))
        return record["attraction_kBT"] == 0

    def evaluate_pair(self, key: tuple) -> dict:
        """Add a preparation-only penalty for prescribed bonds that are too weak."""
        record = pair_energy(self.snapshot, *key, self.parameters)
        if self.fixed_bonds is not None and key in self.fixed_bonds:
            deficit = max(0.0, self.minimum_bond_fraction * self.parameters.bond_epsilon
                          + record["attraction_kBT"])
            record["bond_penalty_kBT"] = self.bond_penalty * deficit * deficit
        return record

    @staticmethod
    def pair_total(record: dict) -> float:
        """Return the active pair energy without separate bond-count terms."""
        return (record["core_repulsion_kBT"] + record["tip_repulsion_kBT"]
                + record["attraction_kBT"] + record.get("bond_penalty_kBT", 0.0))

    def total(self) -> float:
        """Recompute the cached total with compensated summation after each move."""
        return math.fsum(self.pair_total(record) for record in self.records.values())

    def update_occupancy(self, occupancy: np.ndarray, key: tuple,
                         record: dict, change: int) -> None:
        """Apply the program's strict half-well contact definition."""
        if record["attraction_kBT"] < -self.parameters.contact_fraction * self.parameters.bond_epsilon:
            arm1, arm2 = record["best_patch_pair"]
            occupancy[key[0], arm1] += change
            occupancy[key[1], arm2] += change

    def trial(self, molecule: int) -> tuple:
        """Evaluate the changed molecule's row and check the entire contact state."""
        records = {}
        occupancy = self.occupancy.copy()
        differences = []
        self.evaluations += 1
        for other in range(len(self.snapshot.centers)):
            if other == molecule:
                continue
            key = tuple(sorted((molecule, other)))
            old = self.records[key]
            new = self.evaluate_pair(key)
            if not self.valid_topology(key, new):
                return math.inf, {}, occupancy
            value = self.pair_total(new)
            if not math.isfinite(value):
                return math.inf, {}, occupancy
            differences.append(value - self.pair_total(old))
            records[key] = new
            self.update_occupancy(occupancy, key, old, -1)
            self.update_occupancy(occupancy, key, new, 1)
        if self.parameters.single_partner and np.any(occupancy > 1):
            return math.inf, {}, occupancy
        return math.fsum(differences), records, occupancy

    def accept(self, records: dict, occupancy: np.ndarray) -> None:
        """Commit a feasible move and avoid accumulating incremental energy drift."""
        self.records.update(records)
        self.occupancy = occupancy
        self.energy = self.total()

    def summary(self, sweep: int, translation_step: float,
                rotation_step: float, accepted: int) -> dict:
        """Summarize actual energy components at an accepted configuration."""
        core = math.fsum(r["core_repulsion_kBT"] for r in self.records.values())
        tips = math.fsum(r["tip_repulsion_kBT"] for r in self.records.values())
        attraction = math.fsum(r["attraction_kBT"] for r in self.records.values())
        return {"sweep": sweep, "total_energy_kBT": core + tips + attraction,
                "objective_kBT": self.energy,
                "bond_penalty_kBT": math.fsum(r.get("bond_penalty_kBT", 0.0) for r in self.records.values()),
                "core_repulsion_kBT": core, "tip_repulsion_kBT": tips, "attraction_kBT": attraction,
                "translation_step_nm": translation_step, "rotation_step_degrees": rotation_step,
                "accepted_moves": accepted, "evaluations": self.evaluations}


def minimize_snapshot(snapshot: Snapshot, parameters: EnergyParameters,
                      settings: SearchSettings,
                      callback: Callable[[dict], None] | None = None,
                      fixed_bonds: dict | None = None) -> tuple:
    """Return a monotonically improved copy, history, and honest stopping status.

    At each molecule/axis, test both signed translations or rotations and accept
    the better feasible energy decrease. Extrapolate successful sweeps as
    collective rigid-body pattern moves, also checking the full contact state.
    Halve the step sizes after a complete sweep without improvement. Stop when
    that test also fails at both requested step tolerances, or when the
    iteration/time budget is exhausted.
    """
    if snapshot.box_length <= 2 * parameters.interaction_cutoff:
        raise ValueError("Box length must exceed twice the interaction cutoff")
    current = snapshot.copy()
    cache = PairCache(current, parameters, fixed_bonds, settings.minimum_bond_fraction, settings.bond_penalty)
    translation_step = settings.translation_step
    rotation_step = settings.rotation_step
    rng = np.random.default_rng(settings.seed)
    history = []
    status = {"converged": False, "reason": "max_sweeps"}
    start = time.monotonic()
    axes = np.eye(3)

    def record(sweep: int, accepted: int) -> None:
        row = cache.summary(sweep, translation_step, rotation_step, accepted)
        history.append(row)
        if callback:
            callback(row)

    record(0, 0)
    try:
        for sweep in range(1, settings.max_sweeps + 1):
            before_sweep = current.copy()
            accepted = 0
            timed_out = False
            for molecule in rng.permutation(len(current.centers)):
                molecule = int(molecule)
                for coordinate in range(6):
                    if settings.max_seconds is not None and time.monotonic() - start >= settings.max_seconds:
                        timed_out = True
                        break
                    center = current.centers[molecule].copy()
                    patches = current.patches[molecule].copy()
                    best_delta = -settings.energy_tolerance
                    best = None
                    for sign in (1, -1):
                        current.centers[molecule] = center
                        current.patches[molecule] = patches
                        try:
                            if coordinate < 3:
                                current.translate(molecule, sign * translation_step * axes[coordinate])
                            else:
                                current.rotate(molecule, axes[coordinate - 3], sign * rotation_step)
                            delta, records, occupancy = cache.trial(molecule)
                            if delta < best_delta:
                                best_delta = delta
                                best = (current.centers[molecule].copy(), current.patches[molecule].copy(),
                                        records, occupancy)
                        finally:
                            current.centers[molecule] = center
                            current.patches[molecule] = patches
                    if best is not None:
                        current.centers[molecule], current.patches[molecule] = best[:2]
                        cache.accept(best[2], best[3])
                        accepted += 1
                if timed_out:
                    break
            # Extrapolate a successful sweep as a collective rigid-body move.
            # This accelerates coupled motions without scaling any arm vector.
            if accepted and settings.pattern_steps and not timed_out:
                base = current.copy()
                displacement = base.centers - before_sweep.centers
                old_vectors = before_sweep.patches - before_sweep.centers[:, None]
                new_vectors = base.patches - base.centers[:, None]
                rotations = []
                for old, new in zip(old_vectors, new_vectors):
                    left, _, right = np.linalg.svd(old.T @ new)
                    correction = np.eye(3)
                    correction[-1, -1] = np.linalg.det(left @ right)
                    rotations.append((left @ correction @ right).T)
                for exponent in range(settings.pattern_steps):
                    if settings.max_seconds is not None and time.monotonic() - start >= settings.max_seconds:
                        timed_out = True
                        break
                    factor = 2 ** exponent
                    trial = base.copy()
                    trial.centers += factor * displacement
                    for index, rotation in enumerate(rotations):
                        trial.patches[index] = (trial.centers[index]
                                                + new_vectors[index] @ np.linalg.matrix_power(rotation, factor).T)
                    cache.evaluations += 1
                    try:
                        trial_cache = PairCache(trial, parameters, fixed_bonds,
                                               settings.minimum_bond_fraction, settings.bond_penalty)
                    except ValueError:
                        break
                    if trial_cache.energy >= cache.energy - settings.energy_tolerance:
                        break
                    trial_cache.evaluations = cache.evaluations
                    current, cache = trial, trial_cache
                    accepted += 1
            record(sweep, accepted)
            if timed_out:
                status["reason"] = "time_limit"
                break
            if accepted == 0:
                if translation_step <= settings.translation_tolerance and rotation_step <= settings.rotation_tolerance:
                    status.update(converged=True, reason="no_improvement_at_requested_steps")
                    break
                translation_step = max(settings.translation_tolerance, translation_step / 2)
                rotation_step = max(settings.rotation_tolerance, rotation_step / 2)
    except KeyboardInterrupt:
        # Each trial restores its tentative pose in finally, leaving the best state.
        status["reason"] = "interrupted"
    status.update(elapsed_seconds=time.monotonic() - start, evaluations=cache.evaluations,
                  sweeps=history[-1]["sweep"], translation_step_nm=translation_step,
                  rotation_step_degrees=rotation_step)
    return current, history, status


def detached_molecules(snapshot: Snapshot, parameters: EnergyParameters) -> list[int]:
    """Identify molecules with no appreciable attractive interaction (<1e-8 kBT)."""
    attraction = np.zeros(len(snapshot.centers))
    for first in range(len(snapshot.centers)):
        for second in range(first + 1, len(snapshot.centers)):
            value = pair_energy(snapshot, first, second, parameters)["attraction_kBT"]
            attraction[first] += value
            attraction[second] += value
    return np.flatnonzero(attraction >= -1e-8).tolist()


def load_fixed_bonds(path: Path, molecules: int) -> dict:
    """Read the exact zero-based arm pairings recovered from the source HTML."""
    pairs = {}
    used_arms = set()
    for bond in json.loads(path.read_text())["visual_pairings"]:
        (first, arm1), (second, arm2) = bond["ends"]
        if not all(isinstance(value, int) for value in (first, arm1, second, arm2)):
            raise ValueError("Bond molecule and arm IDs must be integers")
        if not (0 <= first < molecules and 0 <= second < molecules and first != second
                and 0 <= arm1 < 6 and 0 <= arm2 < 6):
            raise ValueError("Bond IDs are outside the snapshot's molecule/arm ranges")
        if first > second:
            first, second, arm1, arm2 = second, first, arm2, arm1
        if (first, second) in pairs or (first, arm1) in used_arms or (second, arm2) in used_arms:
            raise ValueError("A fixed bond list cannot reuse an arm or a molecule pair")
        pairs[first, second] = (arm1, arm2)
        used_arms.update(((first, arm1), (second, arm2)))
    if not pairs:
        raise ValueError("The fixed bond list is empty")
    return pairs


def audit_bonds(snapshot: Snapshot, parameters: EnergyParameters, bonds: dict) -> list[dict]:
    """Report each prescribed arm pair, its separation, and its actual attraction."""
    from snapshot_energy import minimum_image
    rows = []
    for (first, second), (arm1, arm2) in sorted(bonds.items()):
        pair = pair_energy(snapshot, first, second, parameters)
        separation = minimum_image(snapshot.patches[second, arm2] - snapshot.patches[first, arm1], snapshot.box_length)
        rows.append({"molecule1": first, "arm1": arm1, "molecule2": second, "arm2": arm2,
                     "tip_distance_nm": float(np.linalg.norm(separation)),
                     "attraction_kBT": pair["attraction_kBT"],
                     "meets_kmc_contact_threshold": pair["attraction_kBT"] < -parameters.contact_fraction * parameters.bond_epsilon,
                     "retained": pair["attraction_kBT"] < 0 and pair["best_patch_pair"] == [arm1, arm2]})
    return rows


def main() -> None:
    """Save a minimized snapshot, native initial state, report, and energy history."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("snapshot", type=Path, nargs="?", default=DEFAULT_SNAPSHOT)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, help="New directory; existing paths are rejected")
    parser.add_argument("--fixed-bonds", type=Path, help="Preserve exactly the arm pairs in a snapshot JSON's visual_pairings")
    parser.add_argument("--scale-centers", type=float, default=1.0, help="Initial center-only scale factor; arms stay unchanged")
    for field in fields(SearchSettings):
        parser.add_argument("--" + field.name.replace("_", "-"), dest=field.name,
                            type=int if field.name in ("max_sweeps", "seed", "pattern_steps") else float, default=field.default)
    for field in fields(EnergyParameters):
        options = ["--" + field.name.replace("_", "-")]
        if "_" in field.name:
            options.append("--" + field.name)
        parser.add_argument(*options, dest=field.name, type=int if field.name == "single_partner" else float,
                            choices=(0, 1) if field.name == "single_partner" else None,
                            default=20.0 if field.name == "bond_epsilon" else field.default)
    args = parser.parse_args()
    try:
        settings = SearchSettings(**{field.name: getattr(args, field.name) for field in fields(SearchSettings)})
        parameters = EnergyParameters(**{field.name: getattr(args, field.name) for field in fields(EnergyParameters)})
        source = args.snapshot.resolve()
        snapshot = load_snapshot(source, args.frame)
        snapshot.scale_centers(args.scale_centers)
        rigid_orientations(snapshot)  # Reject deformed inputs before producing artifacts.
        bonds = load_fixed_bonds(args.fixed_bonds, len(snapshot.centers)) if args.fixed_bonds else None
        if settings.minimum_bond_fraction and bonds is None:
            raise ValueError("minimum_bond_fraction requires --fixed-bonds")
        if bonds is not None:
            PairCache(snapshot, parameters, bonds, settings.minimum_bond_fraction, settings.bond_penalty)
        initial = calculate_energy(snapshot, parameters)
        if not math.isfinite(initial["total_energy_kBT"]) or not initial["single_partner_constraint_satisfied"]:
            raise ValueError("Minimization requires a finite, single-partner-admissible initial state")
        if args.output_dir:
            output = args.output_dir.resolve()
            output.mkdir(parents=True, exist_ok=False)
        else:
            parent = PROJECT / "artifacts/minimization"
            parent.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            output = Path(tempfile.mkdtemp(prefix=f"minimize_{timestamp}_", dir=parent))
        metadata = {"source_snapshot": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "frame": args.frame, "parameters": asdict(parameters), "settings": asdict(settings),
                    "initial_center_scale": args.scale_centers,
                    "fixed_bonds_source": str(args.fixed_bonds.resolve()) if args.fixed_bonds else None}
        (output / "input.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Output: {output}", flush=True)
        print(f"Initial energy: {initial['total_energy_kBT']:.12g} kBT", flush=True)
        if bonds is not None:
            print(f"Keeping {len(bonds)} exact HTML arm pairings; no new attractive pairs allowed.", flush=True)
        initial_detached = detached_molecules(snapshot, parameters)
        if initial_detached:
            print(f"Initially detached molecule IDs: {initial_detached}; core indices: "
                  f"{[7 * index + 1 for index in initial_detached]}. Local minimization cannot guarantee recapture.", flush=True)
        with (output / "energy_history.csv").open("x", newline="") as stream:
            writer = None

            def progress(row: dict) -> None:
                nonlocal writer
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                stream.flush()
                if row["sweep"] % 10 == 0 or row["accepted_moves"] == 0:
                    print(f"Sweep {row['sweep']}: E={row['total_energy_kBT']:.12g}; "
                          f"objective={row['objective_kBT']:.12g}; "
                          f"accepted={row['accepted_moves']}; translation step={row['translation_step_nm']:.4g} nm; "
                          f"rotation step={row['rotation_step_degrees']:.4g} degrees", flush=True)

            minimized, history, status = minimize_snapshot(snapshot, parameters, settings, progress, bonds)
        final = calculate_energy(minimized, parameters)
        initial_objective = PairCache(snapshot, parameters, bonds, settings.minimum_bond_fraction, settings.bond_penalty).energy
        final_cache = PairCache(minimized, parameters, bonds, settings.minimum_bond_fraction, settings.bond_penalty)
        if final_cache.energy > initial_objective + 1e-7 or not final["single_partner_constraint_satisfied"]:
            raise RuntimeError("Final verification failed; no minimized snapshot was written")
        if not math.isclose(final_cache.energy, history[-1]["objective_kBT"], rel_tol=1e-10, abs_tol=1e-7):
            raise RuntimeError("Pair cache differs from full final energy")
        if bonds is not None:
            audit = audit_bonds(minimized, parameters, bonds)
            with (output / "bond_audit.csv").open("x", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(audit[0]))
                writer.writeheader()
                writer.writerows(audit)
            (output / "fixed_bonds.json").write_text(json.dumps({"visual_pairings": [
                {"ends": [[first, arm1], [second, arm2]]}
                for (first, second), (arm1, arm2) in sorted(bonds.items())]}, indent=2) + "\n")
        write_snapshot(minimized, output / "minimized.lammpstrj")
        orientations, fit_error = rigid_orientations(minimized)
        write_initial_state(minimized, orientations, output / "initial_state.kmc")
        detached = detached_molecules(minimized, parameters)
        report = {"initial": initial, "final": final, "optimization": status,
                  "initial_detached_molecules": initial_detached, "final_detached_molecules": detached,
                  "final_detached_core_indices": [7 * index + 1 for index in detached],
                  "max_rigid_fit_error_nm": fit_error,
                  "method": "Rigid-body pattern search of physical energy plus optional bond-strength penalties.",
                  "fixed_arm_pairings": audit if bonds is not None else None,
                  "all_fixed_bonds_meet_kmc_contact_threshold": all(row["meets_kmc_contact_threshold"] for row in audit) if bonds is not None else None,
                  "initial_objective_kBT": initial_objective, "final_objective_kBT": final_cache.energy,
                  "final_bond_penalty_kBT": history[-1]["bond_penalty_kBT"],
                  "topology_constraints_used_in_kmc": False,
                  "global_minimum_guaranteed": False}
        (output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Final energy: {final['total_energy_kBT']:.12g} kBT; {status['reason']}", flush=True)
        print(f"Snapshot: {output / 'minimized.lammpstrj'}", flush=True)
        if bonds is not None and settings.minimum_bond_fraction > 0:
            bonded = sum(row["meets_kmc_contact_threshold"] for row in audit)
            print(f"Original arm pairings meeting the KMC bond threshold: {bonded}/{len(bonds)}", flush=True)
            if bonded != len(bonds):
                parser.exit(2, "Bond-strength preparation is incomplete; the saved partial result needs further relaxation.\n")
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
