#!/bin/bash -p
# shellcheck shell=bash
if [[ $- != *p* ]]; then
    printf '%s\n' "error: invoke this AutoDL entry directly; an explicit Bash must use -p" >&2
    exit 2
fi
while IFS= read -r sg_imported_function; do
    builtin unset -f -- "${sg_imported_function}"
done < <(builtin compgen -A function)
builtin unset sg_imported_function
builtin set +x +v
set -Eeuo pipefail

sg_entry_source="${BASH_SOURCE[0]}"
sg_here="${sg_entry_source%/*}"
if [[ "${sg_here}" == "${sg_entry_source}" ]]; then
    sg_here="."
fi
if [[ "${sg_here}" != /* ]]; then
    sg_here="${PWD}/${sg_here}"
fi
exec "${sg_here}/_run_scaleguard.sh" smoke "$@"
