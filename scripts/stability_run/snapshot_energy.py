"""Load, inspect, rigidly edit, and save seven-site KMC LAMMPS snapshots.

Requires Python 3.10+ and NumPy. The energy implementation mirrors the active
nonbonded functions in energies.h and the signed dihedral in utils.cpp; it does
not call the C++ header at runtime. Molecule IDs are zero-based, while dump site
IDs are 1..7*N. Energies are in kBT, lengths in nm, and potential angles in radians.
"""

import argparse
from collections import deque
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
from typing import Iterator

import numpy as np


BODY_VECTORS = np.array([
    [1.0, 0.0, 0.47], [1.0, 0.0, -0.47],
    [-0.5, 0.866, 0.47], [-0.5, 0.866, -0.47],
    [-0.5, -0.866, 0.47], [-0.5, -0.866, -0.47],
])


def minimum_image(vector: np.ndarray, length: float) -> np.ndarray:
    """Use the same [-L/2,L/2) displacement convention as real_vector()."""
    return (np.asarray(vector) + length / 2) % length - length / 2


@dataclass(frozen=True)
class EnergyParameters:
    """Defaults match System::ReadInput(), not any particular research run."""

    core_sigma: float = 1.8
    core_epsilon: float = 20.0
    bond_epsilon: float = 10.0
    bond_r0: float = 0.305638339862
    bond_cutoff: float = 0.5
    arm_epsilon: float = 1.0
    sigma_angle: float = 0.3
    sigma_torsion: float = 0.6
    angle0: float = 0.0
    torsion0: float = 0.0
    contact_fraction: float = 0.5
    single_partner: bool = True

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name in ("core_sigma", "bond_r0", "arm_epsilon", "sigma_angle", "sigma_torsion"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if min(self.core_epsilon, self.bond_epsilon) < 0:
            raise ValueError("Energy strengths must be nonnegative")
        if self.bond_cutoff <= self.bond_r0:
            raise ValueError("bond_cutoff must exceed bond_r0")
        if not 0 < self.contact_fraction < 1 or not 0 <= self.angle0 <= math.pi:
            raise ValueError("Require 0 < contact_fraction < 1 and 0 <= angle0 <= pi")

    @property
    def interaction_cutoff(self) -> float:
        """Return the center cutoff computed from the KMC canonical arm length."""
        return max(2 ** (1 / 6) * self.core_sigma,
                   2 * float(np.max(np.linalg.norm(BODY_VECTORS, axis=1))) + self.bond_cutoff)


@dataclass
class Snapshot:
    """One cubic, fully periodic frame; patches are unwrapped about each center."""

    timestep: int
    bounds: np.ndarray
    centers: np.ndarray
    patches: np.ndarray

    def __post_init__(self) -> None:
        self.bounds = np.asarray(self.bounds, dtype=float).copy()
        self.centers = np.asarray(self.centers, dtype=float).copy()
        self.patches = np.asarray(self.patches, dtype=float).copy()
        if self.bounds.shape != (3, 2) or self.centers.ndim != 2 or self.centers.shape[1] != 3:
            raise ValueError("Expected bounds (3,2) and centers (N,3)")
        if len(self.centers) == 0 or self.patches.shape != (len(self.centers), 6, 3):
            raise ValueError("Expected six ordered patches per molecule")
        if not all(np.all(np.isfinite(array)) for array in (self.bounds, self.centers, self.patches)):
            raise ValueError("Coordinates and bounds must be finite")
        lengths = np.diff(self.bounds, axis=1).ravel()
        if np.any(lengths <= 0) or not np.allclose(lengths, lengths[0], rtol=1e-12, atol=1e-12):
            raise ValueError("The KMC model requires a cubic box")

    @property
    def box_length(self) -> float:
        """Return the cubic box length in nm."""
        return float(self.bounds[0, 1] - self.bounds[0, 0])

    def copy(self) -> "Snapshot":
        """Make an independent frame for trial edits."""
        return Snapshot(self.timestep, self.bounds, self.centers, self.patches)

    def translate(self, molecule_id: int, displacement: np.ndarray) -> None:
        """Translate a complete molecule by a three-component displacement in nm."""
        self._check_id(molecule_id)
        displacement = np.asarray(displacement, dtype=float)
        if displacement.shape != (3,) or not np.all(np.isfinite(displacement)):
            raise ValueError("Translation needs three finite components")
        self.centers[molecule_id] += displacement
        self.patches[molecule_id] += displacement

    def rotate(self, molecule_id: int, axis: np.ndarray, degrees: float) -> None:
        """Rotate a whole molecule about its center and a world-coordinate axis."""
        self._check_id(molecule_id)
        axis = np.asarray(axis, dtype=float)
        if axis.shape != (3,) or not np.all(np.isfinite(axis)) or not math.isfinite(degrees):
            raise ValueError("Rotation needs a finite three-component axis and angle")
        norm = float(np.linalg.norm(axis))
        if not norm > 0 or not math.isfinite(norm):
            raise ValueError("Rotation axis must have a finite nonzero length")
        axis = axis / norm
        angle = math.radians(degrees)
        vectors = self.patches[molecule_id] - self.centers[molecule_id]
        rotated = (vectors * math.cos(angle) + np.cross(axis, vectors) * math.sin(angle)
                   + (vectors @ axis)[:, None] * axis * (1 - math.cos(angle)))
        self.patches[molecule_id] = self.centers[molecule_id] + rotated

    def scale_centers(self, factor: float, origin: np.ndarray | None = None) -> None:
        """Scale center separations, translating intact molecules without stretching.

        The default origin is the mean of the stored centers. For a cluster
        split across periodic boundaries, first put its centers into one
        continuous image. This edit retains the original box dimensions.
        """
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("Center scale factor must be finite and positive")
        origin = self.centers.mean(axis=0) if origin is None else np.asarray(origin, dtype=float)
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise ValueError("Scaling origin needs three finite coordinates")
        if factor == 1:
            return
        new_centers = origin + factor * (self.centers - origin)
        if not np.all(np.isfinite(new_centers)):
            raise ValueError("Scaled centers are not finite")
        self.patches += (new_centers - self.centers)[:, None]
        self.centers = new_centers

    def _check_id(self, molecule_id: int) -> None:
        if not isinstance(molecule_id, (int, np.integer)) or not 0 <= molecule_id < len(self.centers):
            raise ValueError(f"Molecule ID must be an integer in 0..{len(self.centers) - 1}")


def iter_snapshots(path: str | Path) -> Iterator[Snapshot]:
    """Stream KMC dumps with index/id, type, x,y,z columns; reject other layouts."""
    with Path(path).open() as stream:
        while True:
            first = stream.readline()
            if not first:
                return
            if not first.strip():
                continue
            if first.strip() != "ITEM: TIMESTEP":
                raise ValueError("Expected ITEM: TIMESTEP")
            timestep = int(stream.readline())
            if stream.readline().strip() != "ITEM: NUMBER OF ATOMS":
                raise ValueError("Missing atom count header")
            count = int(stream.readline())
            if count <= 0 or count % 7:
                raise ValueError("KMC snapshots need exactly seven sites per molecule")
            if stream.readline().split() != ["ITEM:", "BOX", "BOUNDS", "pp", "pp", "pp"]:
                raise ValueError("Only orthogonal, fully periodic boxes are supported")
            bounds = np.array([[float(value) for value in stream.readline().split()] for _ in range(3)])
            header = stream.readline().split()
            if header[:2] != ["ITEM:", "ATOMS"]:
                raise ValueError("Missing atom columns")
            columns = header[2:]
            identifier = "index" if "index" in columns else "id"
            required = [identifier, "type", "x", "y", "z"]
            if not all(name in columns for name in required):
                raise ValueError("Expected index (or id), type, and unscaled x,y,z columns")
            indexes = [columns.index(name) for name in required]
            positions = np.empty((count, 3))
            types = np.empty(count, dtype=int)
            seen = set()
            for _ in range(count):
                row = stream.readline().split()
                if len(row) != len(columns):
                    raise ValueError("Truncated frame or wrong number of atom fields")
                atom_id = int(row[indexes[0]])
                if not 1 <= atom_id <= count or atom_id in seen:
                    raise ValueError("Site IDs must be unique and cover 1..7*N")
                seen.add(atom_id)
                types[atom_id - 1] = int(row[indexes[1]])
                positions[atom_id - 1] = [float(row[index]) for index in indexes[2:]]
            expected_types = np.tile([1, 2, 2, 2, 2, 2, 2], count // 7)
            if not np.array_equal(types, expected_types):
                raise ValueError("Expected center then six arms in each seven-site ID group")
            centers = positions.reshape(-1, 7, 3)[:, 0]
            patches = positions.reshape(-1, 7, 3)[:, 1:]
            snapshot = Snapshot(timestep, bounds, centers, patches)
            snapshot.patches = centers[:, None] + minimum_image(patches - centers[:, None], snapshot.box_length)
            yield snapshot


def load_snapshot(path: str | Path, frame: int = 0) -> Snapshot:
    """Load a zero-based frame; negative indexes count from the trajectory end."""
    iterator = iter_snapshots(path)
    try:
        if frame >= 0:
            for index, snapshot in enumerate(iterator):
                if index == frame:
                    return snapshot
        else:
            tail = deque(iterator, maxlen=-frame)
            if len(tail) == -frame:
                return tail[0]
    finally:
        iterator.close()
    raise ValueError(f"Frame {frame} does not exist")


def write_snapshot(snapshot: Snapshot, path: str | Path, overwrite: bool = False) -> None:
    """Save one frame, wrapping every site; existing files require explicit consent."""
    path = Path(path)
    positions = np.concatenate((snapshot.centers[:, None], snapshot.patches), axis=1)
    positions = snapshot.bounds[:, 0] + (positions - snapshot.bounds[:, 0]) % snapshot.box_length
    if not np.all(np.isfinite(positions)):
        raise ValueError("Cannot write non-finite coordinates")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if overwrite else "x") as stream:
        stream.write(f"ITEM: TIMESTEP\n{snapshot.timestep}\nITEM: NUMBER OF ATOMS\n{positions.size // 3}\n")
        stream.write("ITEM: BOX BOUNDS pp pp pp\n")
        for lower, upper in snapshot.bounds:
            stream.write(f"{lower:.17g} {upper:.17g}\n")
        stream.write("ITEM: ATOMS index type x y z\n")
        for index, position in enumerate(positions.reshape(-1, 3)):
            coordinates = " ".join(f"{value:.17g}" for value in position)
            stream.write(f"{index + 1} {1 if index % 7 == 0 else 2} {coordinates}\n")


def _angles(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine = np.sum(first * second, axis=-1) / denominator
    return np.arccos(np.clip(cosine, -1, 1))


def _dihedrals(first: np.ndarray, axis: np.ndarray, second: np.ndarray) -> np.ndarray:
    # Keep the signed normal convention from dihedral_vectors(), not a replacement.
    normal1 = np.cross(first, axis)
    normal2 = np.cross(axis, second)
    result = np.arctan2(np.linalg.norm(axis) * np.sum(first * normal2, axis=-1),
                        np.sum(normal1 * normal2, axis=-1))
    undefined = (np.linalg.norm(axis) == 0) | (np.linalg.norm(normal1, axis=-1) == 0) | (np.linalg.norm(normal2, axis=-1) == 0)
    return np.where(undefined, np.nan, result)


def pair_energy(snapshot: Snapshot, first: int, second: int,
                parameters: EnergyParameters) -> dict:
    """Return core/tip repulsion and the strongest attractive patch pair."""
    snapshot._check_id(first)
    snapshot._check_id(second)
    if first == second:
        raise ValueError("A molecule cannot interact with itself")
    first, second = sorted((first, second))
    length = snapshot.box_length
    axis = minimum_image(snapshot.centers[second] - snapshot.centers[first], length)
    distance2 = float(axis @ axis)
    result = {"molecule_ids": [first, second], "core_repulsion_kBT": 0.0,
              "tip_repulsion_kBT": 0.0, "attraction_kBT": 0.0, "best_patch_pair": None}
    if distance2 >= parameters.interaction_cutoff ** 2:
        return result
    if parameters.core_epsilon and distance2 < 2 ** (1 / 3) * parameters.core_sigma ** 2:
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            q6 = (parameters.core_sigma ** 2 / np.float64(distance2)) ** 3
            result["core_repulsion_kBT"] = float(4 * parameters.core_epsilon * (q6 - 0.5) ** 2)
    separation = minimum_image(snapshot.patches[second][None] - snapshot.patches[first][:, None], length)
    radii2 = np.sum(separation ** 2, axis=-1)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        tip = parameters.arm_epsilon * ((parameters.bond_r0 ** 2 / radii2) ** 3 - 1) ** 2
    result["tip_repulsion_kBT"] = float(np.sum(np.where(radii2 < parameters.bond_r0 ** 2, tip, 0)))
    if parameters.bond_epsilon == 0:
        return result
    vectors1 = snapshot.patches[first] - snapshot.centers[first]
    vectors2 = snapshot.patches[second] - snapshot.centers[second]
    alpha = _angles(vectors1[:, None], separation)
    beta = _angles(vectors2[None], -separation)
    chi = _dihedrals(vectors1[np.arange(6) ^ 1, None], axis,
                     vectors2[None, np.arange(6) ^ 1])
    wrapped = np.array([math.remainder(float(value) - parameters.torsion0, 2 * math.pi)
                        if math.isfinite(value) else math.nan for value in chi.ravel()]).reshape(6, 6)
    radius = np.sqrt(radii2)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore", under="ignore"):
        factor = np.exp(-0.5 * (((alpha - parameters.angle0) / parameters.sigma_angle) ** 2
                                + ((beta - parameters.angle0) / parameters.sigma_angle) ** 2
                                + (wrapped / parameters.sigma_torsion) ** 2))
        q6 = (parameters.bond_r0 / radius) ** 6
        t = (radius - parameters.bond_r0) / (parameters.bond_cutoff - parameters.bond_r0)
        radial = -parameters.bond_epsilon * (2 * q6 - q6 ** 2) * (1 - t) ** 3 * (1 + 3 * t + 6 * t ** 2)
        radial = np.where(radius <= parameters.bond_r0, -parameters.bond_epsilon, radial)
        attraction = radial * factor
    attraction = np.where((radius > 0) & (radius < parameters.bond_cutoff)
                          & np.isfinite(attraction), attraction, 0)
    best = int(np.argmin(attraction))
    if attraction.ravel()[best] < 0:
        result["attraction_kBT"] = float(attraction.ravel()[best])
        result["best_patch_pair"] = list(divmod(best, 6))
    return result


def calculate_energy(snapshot: Snapshot, parameters: EnergyParameters | None = None,
                     top_pairs: int = 10) -> dict:
    """Sum unique pairs and report hard single-partner violations separately.

    Center-distance screening uses O(N) working memory; the search is O(N**2).
    This is an inspection tool, not a replacement for the simulation's cell list.
    """
    parameters = parameters or EnergyParameters()
    if snapshot.box_length <= 2 * parameters.interaction_cutoff:
        raise ValueError("Box length must exceed twice the model's center cutoff")
    if top_pairs < 0:
        raise ValueError("top_pairs cannot be negative")
    totals = np.zeros(3)
    occupancy = np.zeros((len(snapshot.centers), 6), dtype=int)
    largest = []
    contacts = []
    for first in range(len(snapshot.centers)):
        displacements = minimum_image(snapshot.centers[first + 1:] - snapshot.centers[first], snapshot.box_length)
        candidates = np.flatnonzero(np.sum(displacements ** 2, axis=1) < parameters.interaction_cutoff ** 2)
        for offset in candidates:
            second = first + 1 + int(offset)
            pair = pair_energy(snapshot, first, second, parameters)
            totals += [pair["core_repulsion_kBT"], pair["tip_repulsion_kBT"], pair["attraction_kBT"]]
            if pair["attraction_kBT"] < -parameters.contact_fraction * parameters.bond_epsilon:
                arm1, arm2 = pair["best_patch_pair"]
                occupancy[first, arm1] += 1
                occupancy[second, arm2] += 1
                contacts.append([first, arm1, second, arm2])
            if top_pairs:
                pair["repulsion_kBT"] = pair["core_repulsion_kBT"] + pair["tip_repulsion_kBT"]
                largest.append(pair)
                largest.sort(key=lambda item: item["repulsion_kBT"], reverse=True)
                del largest[top_pairs:]
    violations = np.argwhere(occupancy > 1).tolist()
    return {"timestep": snapshot.timestep, "NMOL": len(snapshot.centers),
            "box_length_nm": snapshot.box_length, "parameters": asdict(parameters),
            "core_repulsion_kBT": float(totals[0]), "tip_repulsion_kBT": float(totals[1]),
            "attraction_kBT": float(totals[2]), "total_energy_kBT": float(np.sum(totals)),
            "geometry_contacts": contacts, "multiple_partner_arms": violations,
            "single_partner_constraint_satisfied": not parameters.single_partner or not violations,
            "largest_repulsive_pairs": largest}


def _json_safe(value):
    """Represent divergent energies as strings instead of invalid JSON numbers."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def main() -> None:
    """Inspect a frame, optionally perform rigid edits, and write a new frame."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("--frame", type=int, default=0, help="Frame index; -1 means last")
    parser.add_argument("--output", type=Path, help="Save the selected, possibly edited frame")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output")
    parser.add_argument("--scale-centers", type=float, default=1.0,
                        help="Scale center separations about their mean; keep arm lengths and box unchanged")
    parser.add_argument("--scale-origin", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        help="Optional origin for center scaling; default is mean center")
    parser.add_argument("--translate", nargs=4, type=float, action="append", default=[],
                        metavar=("MOL", "DX", "DY", "DZ"), help="Rigid translation in nm; repeatable")
    parser.add_argument("--rotate", nargs=5, type=float, action="append", default=[],
                        metavar=("MOL", "AX", "AY", "AZ", "DEGREES"), help="World-axis rotation about center; repeatable")
    parser.add_argument("--top-pairs", type=int, default=10)
    for field in fields(EnergyParameters):
        options = ["--" + field.name.replace("_", "-")]
        if "_" in field.name:
            options.append("--" + field.name)
        parser.add_argument(*options, dest=field.name,
                            type=int if field.name == "single_partner" else float,
                            choices=(0, 1) if field.name == "single_partner" else None,
                            default=field.default)
    args = parser.parse_args()
    try:
        parameters = EnergyParameters(**{field.name: getattr(args, field.name) for field in fields(EnergyParameters)})
        snapshot = load_snapshot(args.input, args.frame)
        if args.output and args.output.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {args.output}; choose another path or use --overwrite")
        result = {"before": calculate_energy(snapshot, parameters, args.top_pairs)}
        # Expand first, then apply translations, followed by rotations.
        snapshot.scale_centers(args.scale_centers, args.scale_origin)
        for edit in args.translate:
            if not edit[0].is_integer():
                raise ValueError("Molecule IDs must be integers")
            snapshot.translate(int(edit[0]), edit[1:])
        for edit in args.rotate:
            if not edit[0].is_integer():
                raise ValueError("Molecule IDs must be integers")
            snapshot.rotate(int(edit[0]), edit[1:4], edit[4])
        if args.scale_centers != 1 or args.translate or args.rotate:
            result["after"] = calculate_energy(snapshot, parameters, args.top_pairs)
        if args.output:
            write_snapshot(snapshot, args.output, overwrite=args.overwrite)
            result["output"] = str(args.output)
        print(json.dumps(_json_safe(result), indent=2, allow_nan=False))
    except (OSError, ValueError, OverflowError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
