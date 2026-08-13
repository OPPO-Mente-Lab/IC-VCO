#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="${SCRIPT_DIR}/patches"
TARGET_DIR="${1:-$(pwd)}"
MODE="${2:-apply}"

EXPECTED_COMMIT="58fdeb6b980bda22096d912d70d1c858dedc84fd"

usage() {
    cat >&2 <<'EOF'
Usage:
  bash third_party/vlmevalkit/apply_patches.sh <VLMEvalKit checkout> [apply|check]

The patches are tested against VLMEvalKit commit:
  58fdeb6b980bda22096d912d70d1c858dedc84fd
EOF
}

if [[ "${TARGET_DIR}" == "-h" || "${TARGET_DIR}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ "${MODE}" != "apply" && "${MODE}" != "check" ]]; then
    echo "Unsupported mode: ${MODE}" >&2
    usage
    exit 1
fi

if [[ ! -e "${TARGET_DIR}/.git" || ! -d "${TARGET_DIR}/vlmeval" ]]; then
    echo "Target is not a VLMEvalKit git checkout: ${TARGET_DIR}" >&2
    exit 1
fi

mapfile -t PATCHES < <(find "${PATCH_DIR}" -maxdepth 1 -type f -name '*.patch' | sort)
if ((${#PATCHES[@]} == 0)); then
    echo "No patches found under ${PATCH_DIR}" >&2
    exit 1
fi

current_commit="$(git -C "${TARGET_DIR}" rev-parse HEAD)"
if [[ "${current_commit}" != "${EXPECTED_COMMIT}" ]]; then
    cat >&2 <<EOF
Warning: VLMEvalKit HEAD is ${current_commit}, but these patches were tested
against ${EXPECTED_COMMIT}. Continuing with git apply --check first.
EOF
fi

git -C "${TARGET_DIR}" apply --check --whitespace=error "${PATCHES[@]}"

if [[ "${MODE}" == "check" ]]; then
    echo "Patch check passed for ${TARGET_DIR}"
    exit 0
fi

git -C "${TARGET_DIR}" apply --whitespace=error "${PATCHES[@]}"
echo "Applied ${#PATCHES[@]} VLMEvalKit patch(es) to ${TARGET_DIR}"
