#!/usr/bin/env bash
# Keep the original HTML arm pairings during minimization only.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$project_dir/artifacts/snapshots/rotated_pair_axis_arrays"
exec python "$project_dir/tests/minimize_snapshot.py" "$source_dir/snapshot.lammpstrj" \
    --scale-centers 1.16 --fixed-bonds "$source_dir/snapshot.json" \
    --minimum-bond-fraction 0.51 --bond-penalty 100 "$@"
