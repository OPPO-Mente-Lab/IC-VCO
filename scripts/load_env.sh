#!/usr/bin/env bash

load_icvco_env() {
    local env_path="${1:-${ICVCO_ENV_FILE:-}}"
    local had_nounset=0

    if [[ -z "${env_path}" ]]; then
        return 0
    fi
    if [[ ! -f "${env_path}" ]]; then
        return 0
    fi

    case "$-" in
        *u*) had_nounset=1; set +u ;;
    esac
    set -a
    # shellcheck source=/dev/null
    source "${env_path}"
    set +a
    if (( had_nounset == 1 )); then
        set -u
    fi

    echo "[env] loaded ${env_path}"
}
