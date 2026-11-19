"""Convert a KMC LAMMPS frame to rigid-body input, then launch a simulation.

Run from any directory. Requires Python 3.10+ and NumPy. By default the model is
built with make and a fresh output folder is created. Additional NANOROD options
can be supplied after -- using their native underscore spelling.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

import numpy as np

from snapshot_energy import BODY_VECTORS, Snapshot, load_snapshot


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = PROJECT / "artifacts/snapshots/rotated_pair_axis_arrays/snapshot_centers_x1p20.lammpstrj"


def read_body_vectors(path: Path) -> np.ndarray:
    """Read the exact body vectors used by the current Molecule::UpdateVertices."""
    rows = re.findall(r"aorigin\.push_back\(XYZ\(([^)]+)\)\)", path.read_text())
    body = np.array([[float(value) for value in row.split(",")] for row in rows])
    if body.shape != (6, 3) or not np.all(np.isfinite(body)):
        raise ValueError("Cannot read six canonical aorigin vectors from molecule.h")
    return body


def rotation_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper active rotation to the KMC scalar-first convention."""
    r = rotation
    matrix = np.array([
        [r[0, 0]-r[1, 1]-r[2, 2], r[0, 1]+r[1, 0], r[0, 2]+r[2, 0], r[2, 1]-r[1, 2]],
        [r[0, 1]+r[1, 0], r[1, 1]-r[0, 0]-r[2, 2], r[1, 2]+r[2, 1], r[0, 2]-r[2, 0]],
        [r[0, 2]+r[2, 0], r[1, 2]+r[2, 1], r[2, 2]-r[0, 0]-r[1, 1], r[1, 0]-r[0, 1]],
        [r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1], np.trace(r)],
    ]) / 3
    _, vectors = np.linalg.eigh(matrix)
    xyzw = vectors[:, -1]
    result = np.r_[xyzw[3], xyzw[:3]]
    result /= np.linalg.norm(result)
    return result if result[0] >= 0 else -result


def rigid_orientations(snapshot: Snapshot, body: np.ndarray = BODY_VECTORS,
                       tolerance: float = 1e-3) -> tuple[np.ndarray, float]:
    """Fit orientations without permuting arm IDs; reject deformed molecules."""
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Rigid tolerance must be finite and positive")
    orientations = []
    largest_error = 0.0
    for index in range(len(snapshot.centers)):
        target = snapshot.patches[index] - snapshot.centers[index]
        left, _, right = np.linalg.svd(body.T @ target)
        correction = np.eye(3)
        correction[-1, -1] = np.linalg.det(left @ right)
        rotation = (left @ correction @ right).T
        error = float(np.max(np.linalg.norm(body @ rotation.T - target, axis=1)))
        if error > tolerance:
            raise ValueError(f"Molecule {index} is not a KMC rigid body: maximum patch error "
                             f"{error:.6g} nm exceeds {tolerance:.6g} nm")
        largest_error = max(largest_error, error)
        orientations.append(rotation_quaternion(rotation))
    return np.array(orientations), largest_error


def write_initial_state(snapshot: Snapshot, orientations: np.ndarray, path: Path) -> None:
    """Write the versioned center/quaternion format accepted by --initial_state."""
    if orientations.shape != (len(snapshot.centers), 4) or not np.all(np.isfinite(orientations)):
        raise ValueError("Need one finite scalar-first quaternion per molecule")
    with path.open("x") as stream:
        stream.write(f"KMC_RIGID_SNAPSHOT 1\n{len(snapshot.centers)} {snapshot.box_length:.17g}\n")
        for index, (center, orientation) in enumerate(zip(snapshot.centers, orientations)):
            values = np.r_[center, orientation]
            stream.write(str(index) + " " + " ".join(f"{value:.17g}" for value in values) + "\n")


def main() -> None:
    """Build, prepare a fresh run directory, and start MC from a selected frame."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("snapshot", type=Path, nargs="?", default=DEFAULT_SNAPSHOT)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--time", type=float, default=100.0, help="Additional MC time; restarts at time zero")
    parser.add_argument("--bond-epsilon", "--bond_epsilon", type=float, default=20.0)
    parser.add_argument("--sigma-angle", "--sigma_angle", type=float, default=0.3)
    parser.add_argument("--sigma-torsion", "--sigma_torsion", type=float, default=0.6)
    parser.add_argument("--mc-step", "--MCstep", type=float, default=0.1)
    parser.add_argument("--rotation-step", "--rotation_step", type=float, default=0.1)
    parser.add_argument("--seed", "--GSL_SEED", type=int, default=10)
    parser.add_argument("--grid", "--NGRID", type=int, default=10)
    parser.add_argument("--rigid-tolerance", type=float, default=1e-3, help="Maximum fitting error in nm")
    parser.add_argument("--output-dir", type=Path, help="New directory; existing paths are rejected")
    parser.add_argument("--binary", type=Path, default=PROJECT / "NANOROD")
    parser.add_argument("--skip-build", action="store_true", help="Use an already rebuilt binary")
    parser.add_argument("--prepare-only", action="store_true", help="Write state and command without running")
    args, extra = parser.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    try:
        source = args.snapshot.resolve()
        snapshot = load_snapshot(source, args.frame)
        body = read_body_vectors(PROJECT / "molecule.h")
        orientations, error = rigid_orientations(snapshot, body, args.rigid_tolerance)
        if not math.isfinite(args.time) or args.time < 0:
            raise ValueError("time must be finite and nonnegative")
        binary = args.binary.resolve()
        if args.output_dir and args.output_dir.exists():
            raise FileExistsError(f"Output directory already exists: {args.output_dir}")
        if not args.skip_build:
            jobs = min(int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)), 5)
            subprocess.run(["make", "-C", str(PROJECT), "-j", str(max(1, jobs))], check=True)
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError(f"Executable is missing or not executable: {binary}")
        if args.output_dir:
            output = args.output_dir.resolve()
            output.mkdir(parents=True, exist_ok=False)
        else:
            parent = PROJECT / "artifacts/snapshot_runs"
            parent.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            output = Path(tempfile.mkdtemp(prefix=f"snapshot_{timestamp}_", dir=parent))
        state = output / "initial_state.kmc"
        write_initial_state(snapshot, orientations, state)
        command = [str(binary), "--initial_state", str(state), "--time", str(args.time),
                   "--bond_epsilon", str(args.bond_epsilon), "--sigma_angle", str(args.sigma_angle),
                   "--sigma_torsion", str(args.sigma_torsion), "--MCstep", str(args.mc_step),
                   "--rotation_step", str(args.rotation_step), "--GSL_SEED", str(args.seed),
                   "--NGRID", str(args.grid), *extra]
        metadata = {"source_snapshot": str(source), "frame_index": args.frame,
                    "source_timestep": snapshot.timestep, "NMOL": len(snapshot.centers),
                    "box_length_nm": snapshot.box_length, "max_rigid_fit_error_nm": error,
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "command": command, "status": "prepared",
                    "restart_note": "Positions and orientations only; fresh time zero and selected RNG seed."}
        metadata_path = output / "input.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Run directory: {output}", flush=True)
        print(f"Loaded frame {args.frame}: {len(snapshot.centers)} molecules; box {snapshot.box_length:g} nm", flush=True)
        print("Command: " + shlex.join(command), flush=True)
        if args.prepare_only:
            return
        metadata["status"] = "running"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        try:
            with (output / "simulation.log").open("w") as log:
                result = subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT, check=False)
        except KeyboardInterrupt:
            metadata.update(status="interrupted", returncode=130)
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            parser.exit(130, f"Simulation interrupted. Outputs: {output}\n")
        except OSError:
            metadata.update(status="failed_to_start", returncode=None)
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            raise
        metadata["status"] = "finished" if result.returncode == 0 else "failed"
        metadata["returncode"] = result.returncode
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        if result.returncode:
            parser.exit(1, f"Simulation failed (exit {result.returncode}); see {output / 'simulation.log'}\n")
        print(f"Simulation finished. Outputs: {output}")
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
