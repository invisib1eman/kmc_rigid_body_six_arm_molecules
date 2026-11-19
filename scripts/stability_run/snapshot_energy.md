# Load, calculate energy, edit, and save a snapshot

`snapshot_energy.py` is a standalone Python tool requiring NumPy. Run the
examples below from `01_kmc_rigid_body`. It does not modify the C++ program.
Its Python equations mirror the active nonbonded model in `energies.h`, including
the signed torsion convention in `utils.cpp`. Energy parameters are explicit
options; they are not inferred from the trajectory. Defaults match the program,
including **bond_epsilon=10**. Use `--bond-epsilon 20` for the snapshot below.

## Inspect energy

```sh
python tests/snapshot_energy.py \
  artifacts/snapshots/rotated_pair_axis_arrays/snapshot.lammpstrj \
  --bond-epsilon 20
```

The JSON report includes core repulsion, tip repulsion, attraction, total energy,
parameters, the most repulsive molecule pairs, and geometric contacts. The
single-partner constraint is checked separately from the pair-energy sum.
Infinite energies are written as the JSON string `"inf"`.

Use `--frame 3` for the fourth frame or `--frame -1` for the last frame.
All potential angles are radians; `--sigma-angle`/`--sigma_angle` and the other
hyphen/underscore parameter spellings are both accepted.

## Expand center positions without stretching arms

```sh
python tests/snapshot_energy.py \
  artifacts/snapshots/rotated_pair_axis_arrays/snapshot.lammpstrj \
  --bond-epsilon 20 --scale-centers 1.2 \
  --output artifacts/edited/expanded.lammpstrj
```

For scaling factor `s`, origin `O`, center `C`, and tip `P`:

```text
C_new = O + s * (C - O)
P_new = P + (C_new - C)
```

Thus each molecule is only translated: `P_new-C_new = P-C`. The default origin
is the mean of the centers. Use `--scale-origin X Y Z` to choose another origin.
The box dimensions stay fixed. For a cluster split across periodic boundaries,
first place its centers in one continuous image before scaling.

The delivered `snapshot_centers_x1p20.lammpstrj` has 18 molecules and, with
`bond_epsilon=20`, `sigma_angle=0.3`, `sigma_torsion=0.6`, and other defaults:

```text
Core repulsion:   0 kBT
Tip repulsion:    0 kBT
Attraction:     -73.3291752612628 kBT
Total:          -73.3291752612628 kBT
```

This is an expanded geometry, not an energy-minimized structure. The original
snapshot is preserved. A negative total alone does not guarantee zero repulsion;
the two repulsive components were checked separately.

## Translate or rotate selected molecules

```sh
python tests/snapshot_energy.py \
  artifacts/snapshots/rotated_pair_axis_arrays/snapshot_centers_x1p20.lammpstrj \
  --bond-epsilon 20 \
  --translate 0 0.1 0 0 \
  --rotate 1 0 0 1 5 \
  --output artifacts/edited/translated_rotated.lammpstrj
```

Molecule IDs are zero-based. `--translate MOL DX DY DZ` uses nm.
`--rotate MOL AX AY AZ DEGREES` rotates about that molecule's center and the
specified world-coordinate axis. Both options can be repeated. Edits are applied
in this order: center scaling, all translations, all rotations. The report gives
energies before and after edits. Saving writes only the selected frame and wraps
all sites into the original periodic box.

Existing outputs are protected unless `--overwrite` is explicitly supplied.
Dump files must have cubic, fully periodic orthogonal boxes, unscaled `x y z`
coordinates, and the KMC seven-site ID layout. Shuffled rows and `id` instead
of `index` are supported; arbitrary LAMMPS molecule layouts are rejected.

## Use from Python or a notebook

```python
from pathlib import Path
import sys
sys.path.insert(0, str(Path("tests").resolve()))

from snapshot_energy import (
    EnergyParameters, calculate_energy, load_snapshot, write_snapshot,
)

snapshot = load_snapshot("artifacts/snapshots/rotated_pair_axis_arrays/snapshot.lammpstrj")
parameters = EnergyParameters(bond_epsilon=20, sigma_angle=0.3, sigma_torsion=0.6)
trial = snapshot.copy()
trial.scale_centers(1.2)
trial.translate(0, [0.1, 0, 0])
trial.rotate(1, [0, 0, 1], 5)
print(calculate_energy(trial, parameters))
write_snapshot(trial, "artifacts/edited/trial.lammpstrj")
```

Direct arrays `snapshot.centers` and `snapshot.patches` are also available;
editing them individually can distort a molecule. Prefer the rigid-edit methods.
The energy search has O(N²) center screening with O(N) working memory and is
intended for snapshot inspection. It does not replace the simulation's cell list.

Python-only checks:

```sh
python tests/test_snapshot_energy.py
```
