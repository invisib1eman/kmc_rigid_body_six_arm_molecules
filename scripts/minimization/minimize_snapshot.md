# Rigid-body energy minimization

## Preserve the original HTML arm pairings

From `01_kmc_rigid_body`:

```sh
bash minimize_html_bonds.sh --max-sweeps 1000 \
  --translation-tolerance 0.001 --rotation-tolerance 0.1
bash run_minimized.sh
```

The first command starts from the original converted HTML geometry, expands only
the centers by 16%, and reads all 31 exact molecule/arm pairings from its
`snapshot.json`. During minimization, a prescribed pair must remain the selected
attractive arm pair, and other molecule pairs cannot acquire attraction. The
existing single-partner condition also remains active.

A preparation-only penalty strengthens weak prescribed bonds:

```text
objective = KMC energy
          + 100 * sum(max(0, 0.51*bond_epsilon + bond_attraction)**2)
```

The physical KMC energy and this penalty are reported separately. The 0.51 target
provides a margin above the program's default 0.5 contact threshold. The script
checks that every prescribed pair actually meets the KMC bond criterion; if not,
it saves the partial relaxation and exits with status 2 rather than claiming
successful preparation. `bond_audit.csv` lists every arm pairing, tip distance,
attraction, and contact-threshold result. `fixed_bonds.json` preserves the exact
list for future minimizations.

The delivered constrained result is
`artifacts/minimization/minimize_20261119T043700Z_6ba2vjmm/`:
all 31 original pairings qualify as KMC contacts, the physical energy is
**-515.579149320 kBT**, and the remaining preparation penalty is **zero**.
The minimum/maximum prescribed tip separations are approximately 0.29262 and
0.30658 nm. It passed the local stopping test at 0.001 nm / 0.1 degrees.

`run_minimized.sh` defaults to this corrected result. It builds KMC and saves a
copy of the initial state, the simulation log, trajectory, energies, and contacts
in a new `artifacts/run/html_bonds_*/` folder. Default simulation time is 10000;
the optional second argument changes it. If you repeat the minimization, pass
the new run's `initial_state.kmc` path printed by the minimizer:

```sh
bash run_minimized.sh /absolute/path/to/initial_state.kmc 100
```

These topology constraints and penalties apply **only during minimization**.
Subsequent KMC uses the ordinary nonbonded model, so contacts may break and reform.
The C++ energy and MC code were not changed for this preparation.

## Unrestricted minimization

Run from `01_kmc_rigid_body`:

```sh
python tests/minimize_snapshot.py --max-sweeps 2000
```

By default this minimizes the expanded 18-molecule snapshot, with
`bond_epsilon=20`, `sigma_angle=0.3`, and `sigma_torsion=0.6` radians. NumPy is
the only dependency beyond the standard library; SciPy and compilation are not
required. The C++ program and potential are not changed.

For another snapshot or the last frame of a simulation:

```sh
python tests/minimize_snapshot.py path/to/trajectory.lammpstrj --frame -1 \
  --bond-epsilon 20 --sigma-angle 0.3 --sigma-torsion 0.6 \
  --max-sweeps 2000 --output-dir artifacts/my_minimized_snapshot
```

## Method and stopping criterion

This is deterministic, seeded rigid-body pattern search. Each trial translates
or rotates a complete molecule, preserving its arm vectors and internal angles.
For each Cartesian translation or rotation axis, the algorithm tests both signs
and keeps the better feasible energy decrease. It also attempts collective
extrapolations of successful sweeps, using proper rotation matrices so the arms
are never stretched. Every accepted move lowers the same nonbonded energy used
by `snapshot_energy.py` and the current `energies.h` model.

The hard `single_partner` condition is checked for the entire candidate state,
including other molecules whose arms receive new contacts. It is enabled by
default. The energy definition, periodic box, and molecule count are preserved;
no positional restraints or fixed bond graph are added in this unrestricted mode.

After a sweep without improvement, the translation and rotation steps are halved.
Defaults are 0.03 nm and 2 degrees initially, decreasing to 0.0001 nm and 0.01
degrees. `converged=true` means no feasible energy decrease larger than
`1e-9 kBT` was found in either direction of any tested coordinate at these final
step sizes. This is a **local, resolution-dependent** stopping condition, not a
global ground-state guarantee or a continuous-gradient proof.

Relevant controls:

```sh
python tests/minimize_snapshot.py \
  --translation-step 0.03 --rotation-step 2 \
  --translation-tolerance 0.0001 --rotation-tolerance 0.01 \
  --energy-tolerance 1e-9 --max-sweeps 2000 --max-seconds 300
```

Rotation step/tolerance options use degrees; sigma and preferred potential-angle
options use radians. `--seed` controls reproducible sweep ordering.
`--pattern-steps 0` disables collective extrapolation; the default is three
attempts with factors 1, 2, and 4. All energy parameters can be supplied as in
`snapshot_energy.py`, including `--single-partner 0` if that is intentionally the
model you want.

## Output and continuation

A fresh `artifacts/minimization/minimize_*/` folder contains:

- `minimized.lammpstrj`: lowest accepted configuration at termination.
- `initial_state.kmc`: the same rigid-body state in native KMC input format.
- `energy_history.csv`: component energies, steps, and accepted moves per sweep.
- `result.json`: initial/final energies, constraint validity, stopping reason,
  convergence status, and detached-molecule diagnostics.
- `input.json`: source file/hash, model parameters, and search settings.

The input snapshot is preserved. Existing output directories are rejected.
Iteration limits, time limits, and interruption save the best accepted snapshot
with `converged=false`; they are not reported as convergence. Resume by passing
that output snapshot as the next input. The previous final step sizes are in
`result.json` and can be reused through `--translation-step` and `--rotation-step`.

To run MC afterward:

```sh
python tests/run_snapshot.py \
  artifacts/minimization/YOUR_RUN/minimized.lammpstrj --time 10000
```

Supply the same energy parameters to MC if they differ from the launcher's
defaults. Minimization lowers potential energy; finite-temperature MC can still
accept energy-increasing moves afterward. A minimum can also have nonzero
repulsion where it is balanced by stronger attraction.

## Escaped molecules and core index 71

Dump core index 71 is molecule ID 10, since the seven-site layout uses
`core_index = 7*mol_id + 1`. The inspected simulation frame had its nearest tip
about 4.67 nm away, outside the 0.5 nm attractive cutoff. Such a molecule has no
restoring attraction, so a local minimizer cannot be expected to recapture it.

The default expanded starting snapshot places that molecule in the assembly.
When minimizing an escaped frame, reposition it with `snapshot_energy.py` first
or choose an earlier frame. The minimizer reports molecules with summed
attraction weaker than `1e-8 kBT`, including their dump core indices. It does
not silently tether them or change the physical model to preserve an assembly.

Python-only checks:

```sh
python tests/test_minimize_snapshot.py
```
