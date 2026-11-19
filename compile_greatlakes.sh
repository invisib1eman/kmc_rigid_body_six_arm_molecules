#!/bin/bash -l
# Source this file from the batch job so its loaded modules also apply at runtime.
set -euo pipefail

project_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "$project_directory/Makefile" ]]; then
    printf 'Place compile_greatlakes.sh beside the program Makefile. Missing: %s/Makefile\n' "$project_directory" >&2
    exit 2
fi
printf 'Source directory: %s\n' "$project_directory"
build_jobs="${SLURM_CPUS_PER_TASK:-2}"
if [[ ! "$build_jobs" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Invalid CPU count: %s\n' "$build_jobs" >&2
    exit 2
fi
if ! type module >/dev/null 2>&1; then
    printf 'The module command is unavailable. Run this on Great Lakes in a login shell or Slurm job.\n' >&2
    exit 1
fi

# These compiler/library modules are listed in the U-M ARC software catalog.
# Set GL_MODULES to a compatible alternative module list if needed.
read -r -a greatlakes_modules <<< "${GL_MODULES:-gcc/13.2.0 boost gsl}"
module purge
module load "${greatlakes_modules[@]}"
module list

for program in make g++ gsl-config bash awk; do
    if ! command -v "$program" >/dev/null 2>&1; then
        printf 'Required program missing after loading modules: %s\n' "$program" >&2
        exit 1
    fi
done
gsl_compile_flags="$(gsl-config --cflags)"
gsl_link_flags="$(gsl-config --libs)"

# Rebuild copied macOS objects and remove their Homebrew dependency files.
# The clean target removes only object/dependency files, retaining run outputs.
make -C "$project_directory" clean
make -C "$project_directory" -j "$build_jobs" CXX=g++ \
    GSL_CFLAGS="$gsl_compile_flags" GSL_LIBS="$gsl_link_flags"
"$project_directory/NANOROD" --help >/dev/null
export NANOROD_BINARY="$project_directory/NANOROD"
printf 'Built %s\n' "$project_directory/NANOROD"
