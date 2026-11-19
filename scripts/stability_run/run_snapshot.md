# Start KMC from a LAMMPS snapshot

From `01_kmc_rigid_body`, run:

```sh
python tests/run_snapshot.py --time 10000
```

This defaults to the 18-molecule expanded snapshot
`artifacts/snapshots/rotated_pair_axis_arrays/snapshot_centers_x1p20.lammpstrj`.
It builds `NANOROD`, converts the chosen frame into centers and unit quaternions,
and starts a new simulation in a fresh `artifacts/snapshot_runs/snapshot_*/`
folder. The script requires Python 3.10+ with NumPy, plus the usual compiler,
Boost and GSL dependencies for building the model.

The molecule count and cubic box length come from the snapshot. The default
potential settings are `bond_epsilon=20`, `sigma_angle=0.3`, and
`sigma_torsion=0.6` radians; other potential parameters use the program defaults.
Translation and maximum rotation steps are both 0.1, seed is 10, and NGRID is 10.
The launcher's default time is 100 if `--time` is omitted.

## Select a file, frame, parameters, or output directory

```sh
python tests/run_snapshot.py path/to/input.lammpstrj \
  --frame -1 --time 10000 \
  --bond-epsilon 20 --sigma-angle 0.3 --sigma-torsion 0.6 \
  --mc-step 0.1 --rotation-step 0.1 --seed 10 \
  --output-dir artifacts/my_snapshot_simulation
```

`--frame -1` selects the last frame; the default is frame zero. Output directories
must be new, so simulation files cannot append to an earlier run accidentally.
Source snapshots are read-only. On later runs `--skip-build` can reuse the rebuilt
binary. To prepare inputs without starting a simulation:

```sh
python tests/run_snapshot.py --prepare-only --skip-build
```

Additional native NANOROD options can be passed after `--`, for example:

```sh
python tests/run_snapshot.py --time 10000 -- --core_epsilon 20 --single_partner 0
```

Use the launcher's documented flags for time, seed, grid, steps, bond epsilon,
and sigma widths; do not repeat those through native options.

## What is loaded

The script accepts the seven-site cubic, fully periodic dump layout supported by
`snapshot_energy.py`. It preserves center positions and arm labels and fits one
proper rigid rotation to each molecule. The maximum allowed patch fitting error
is 0.001 nm (`--rigid-tolerance`); this accommodates the program's rounded dump
coordinates. Deformed or incorrectly labeled molecules are rejected. All six arms
are then regenerated from the exact `aorigin` vectors in the current `molecule.h`.

The C++ program now accepts `--initial_state FILE`. The state format is:

```text
KMC_RIGID_SNAPSHOT 1
NMOL BOX_LENGTH_NM
0 cx cy cz qw qx qy qz
1 cx cy cz qw qx qy qz
...
```

IDs must uniquely cover `0..NMOL-1`. Quaternions use scalar-first `w,x,y,z` order
and must have unit norm within 1e-6. The reader wraps centers, regenerates vertices,
populates the existing cell list, and lets the existing MC initialization rebuild
contacts from geometry and the selected potential. It does not import the HTML's
visual pairings as bonds.

Explicit `--NMOL` or `--box_length` values that conflict with the state header
are rejected. Without `--initial_state`, the program retains random initialization.

This starts a **new trajectory from the saved geometry**, with time zero and a
fresh chosen RNG seed. It does not restore the earlier RNG state, elapsed time,
or a bitwise continuation of a prior run.

## Output files

Each run folder contains `initial_state.kmc`, `input.json` with the source frame,
command and completion status, `simulation.log`, and the normal KMC trajectory,
energy, contact and grid outputs. A prepared state can also be run directly:

```sh
# Run from a fresh output directory, using absolute paths for these files.
/absolute/path/to/NANOROD --initial_state /absolute/path/to/initial_state.kmc \
  --time 10000 --bond_epsilon 20 --sigma_angle 0.3 --sigma_torsion 0.6
```

Validation:

```sh
make
python tests/test_run_snapshot.py
```

These checks use temporary directories and short native MC runs. They cover
loaded coordinates, subsequent motion, malformed states, parameter conflicts,
deformed input molecules, shuffled input rows, and output-directory protection.
