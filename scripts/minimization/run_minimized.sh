#!/usr/bin/env bash
# Usage: bash run_minimized.sh [initial_state.kmc] [simulation_time]
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Minimized snapshot retaining all 31 original HTML arm pairings.
snapshot="${1:-$project_dir/artifacts/minimization/minimize_20261119T043700Z_6ba2vjmm/initial_state.kmc}"
simulation_time="${2:-10000}"

[[ -f "$snapshot" ]] || { printf 'Snapshot not found: %s\n' "$snapshot" >&2; exit 1; }
snapshot="$(cd -- "$(dirname -- "$snapshot")" && pwd)/$(basename -- "$snapshot")"
make -C "$project_dir"

mkdir -p "$project_dir/artifacts/run"
run_dir="$(mktemp -d "$project_dir/artifacts/run/html_bonds_XXXXXX")"
cp "$snapshot" "$run_dir/initial_state.kmc"
printf 'Snapshot: %s\nOutput: %s\nLog: %s/simulation.log\n' "$snapshot" "$run_dir" "$run_dir"
cd -- "$run_dir"

exec "$project_dir/NANOROD" --initial_state "$run_dir/initial_state.kmc" --time "$simulation_time" \
    --bond_epsilon 20 --bond_r0 0.305638339862 --bond_cutoff 0.5 \
    --sigma_angle 0.3 --sigma_torsion 0.6 \
    --MCstep 0.1 --rotation_step 0.1 --GSL_SEED 10 > simulation.log 2>&1
