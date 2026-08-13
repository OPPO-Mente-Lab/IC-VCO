#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/load_env.sh"
load_icvco_env "${REPO_ROOT}/.env"

GPU_COUNT="${ICVCO_GPU_COUNT:-8}"
DEFAULT_YN_DATA="HallusionBench AMBER"
DEFAULT_MCQ_DATA="CRPE_EXIST CRPE_RELATION R-Bench-Dis R-Bench-Ref BLINK"

usage() {
    cat >&2 <<'EOF'
Usage:
  bash scripts/eval.sh --model-path <path> --work-dir <dir> [options]

Options:
  --model-path <path>             Model or checkpoint path to evaluate.
  --model-family <name>           llava-interleave or llava-onevision. Default: llava-interleave
  --work-dir <dir>                Work directory for VLMEvalKit outputs.
  --vlmevalkit-root <path>        VLMEvalKit checkout. Defaults to VLMEVALKIT_ROOT.
  --lmu-data <path>               Benchmark data root. Sets LMUData.
  --yn-data "<benchmarks>"        Default: HallusionBench AMBER
  --mcq-data "<benchmarks>"       Default: CRPE_EXIST CRPE_RELATION R-Bench-Dis R-Bench-Ref BLINK
  --max-new-tokens <n>            Default: 128
  --yn-judge <name>               Default: qwen-flash
  --mcq-judge <name>              Default: exact_matching
  --api-nproc <n>                 Default: 4
  --mode <all|infer|eval>         Default: all
  --reuse-aux <all|infer|none>    Forward VLMEvalKit --reuse-aux.
  --use-vllm                      Forward --use-vllm to VLMEvalKit.
  --no-reuse                      Disable VLMEvalKit reuse mode.
  --dashscope-compatible-judge    Map DASHSCOPE_API_KEY to OPENAI_API_* for qwen-flash. Default.
  --no-dashscope-compatible-judge Do not rewrite OpenAI-compatible judge env.

Environment overrides:
  ICVCO_FORCE_ATTN_IMPLEMENTATION  Default: eager
  ICVCO_USE_DASHSCOPE_COMPAT_JUDGE Default: 1
  API_BASE / API_KEY                OpenAI-compatible judge endpoint and key.
  DASHSCOPE_OPENAI_API_BASE        Default: https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
EOF
}

model_path=""
model_family="llava-interleave"
work_dir=""
vlmevalkit_root=""
lmu_data=""
yn_data="${DEFAULT_YN_DATA}"
mcq_data="${DEFAULT_MCQ_DATA}"
max_new_tokens="128"
yn_judge="qwen-flash"
mcq_judge="exact_matching"
api_nproc="4"
mode="all"
reuse_aux=""
use_vllm=0
reuse=1
dashscope_compatible_judge="${ICVCO_USE_DASHSCOPE_COMPAT_JUDGE:-1}"

while (($#)); do
    case "$1" in
        --model-path)
            model_path="${2:-}"
            shift 2
            ;;
        --model-family)
            model_family="${2:-}"
            shift 2
            ;;
        --work-dir)
            work_dir="${2:-}"
            shift 2
            ;;
        --vlmevalkit-root)
            vlmevalkit_root="${2:-}"
            shift 2
            ;;
        --lmu-data)
            lmu_data="${2:-}"
            shift 2
            ;;
        --yn-data)
            yn_data="${2:-}"
            shift 2
            ;;
        --mcq-data)
            mcq_data="${2:-}"
            shift 2
            ;;
        --max-new-tokens)
            max_new_tokens="${2:-}"
            shift 2
            ;;
        --yn-judge)
            yn_judge="${2:-}"
            shift 2
            ;;
        --mcq-judge)
            mcq_judge="${2:-}"
            shift 2
            ;;
        --api-nproc)
            api_nproc="${2:-}"
            shift 2
            ;;
        --mode)
            mode="${2:-}"
            shift 2
            ;;
        --reuse-aux)
            reuse_aux="${2:-}"
            shift 2
            ;;
        --use-vllm)
            use_vllm=1
            shift
            ;;
        --no-reuse)
            reuse=0
            shift
            ;;
        --dashscope-compatible-judge)
            dashscope_compatible_judge=1
            shift
            ;;
        --no-dashscope-compatible-judge)
            dashscope_compatible_judge=0
            shift
            ;;
        -h|--help|help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "${model_path}" || -z "${work_dir}" ]]; then
    echo "--model-path and --work-dir are required." >&2
    usage
    exit 1
fi
case "${reuse_aux}" in
    ""|all|infer|none)
        ;;
    *)
        echo "--reuse-aux must be one of: all, infer, none" >&2
        exit 1
        ;;
esac

case "${model_family}" in
    llava-interleave)
        model_alias="llava_next_interleave_7b"
        model_class="LLaVA_Next"
        ;;
    llava-onevision)
        model_alias="llava_onevision_custom"
        model_class="LLaVA_OneVision_HF"
        ;;
    *)
        echo "Unsupported --model-family: ${model_family}" >&2
        echo "Supported values: llava-interleave, llava-onevision" >&2
        exit 1
        ;;
esac

if [[ -n "${vlmevalkit_root}" ]]; then
    export VLMEVALKIT_ROOT="${vlmevalkit_root}"
fi
if [[ -n "${lmu_data}" ]]; then
    export LMUData="${lmu_data}"
fi

VLMEVALKIT_ROOT="${VLMEVALKIT_ROOT:-${VLMEValKit_ROOT:-}}"
if [[ -z "${VLMEVALKIT_ROOT}" || ! -d "${VLMEVALKIT_ROOT}" ]]; then
    echo "Set --vlmevalkit-root or VLMEVALKIT_ROOT to a VLMEvalKit checkout." >&2
    exit 1
fi
if [[ ! -d "${VLMEVALKIT_ROOT}/vlmeval" ]]; then
    echo "VLMEVALKIT_ROOT is not a VLMEvalKit checkout: ${VLMEVALKIT_ROOT}" >&2
    exit 1
fi

cd "${REPO_ROOT}"
if [[ "${work_dir}" != /* ]]; then
    work_dir="${REPO_ROOT}/${work_dir}"
fi
if [[ "${model_path}" != /* && -e "${model_path}" ]]; then
    model_path="$(cd "$(dirname "${model_path}")" && pwd)/$(basename "${model_path}")"
fi
mkdir -p "${work_dir}/_vlmeval_configs"

export PYTHONPATH="${VLMEVALKIT_ROOT}:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export ICVCO_FORCE_ATTN_IMPLEMENTATION="${ICVCO_FORCE_ATTN_IMPLEMENTATION:-eager}"

configure_yn_judge_env() {
    local api_key
    local api_base

    if [[ "${dashscope_compatible_judge}" != "1" ]]; then
        return 0
    fi
    api_key="${API_KEY:-${DASHSCOPE_API_KEY:-}}"
    if [[ -z "${api_key}" ]]; then
        return 0
    fi

    api_base="${API_BASE:-${DASHSCOPE_OPENAI_API_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions}}"
    export OPENAI_API_KEY="${api_key}"
    export OPENAI_API_BASE="${api_base}"
    echo "[eval-env] qwen-flash judge endpoint=${api_base}"
}

write_eval_config() {
    local output_path="$1"
    shift
    python3 - "$output_path" "$model_alias" "$model_class" "$model_path" "$max_new_tokens" "$@" <<'PY'
import json
import sys

output_path, model_alias, model_class, model_path, max_new_tokens, *datasets = sys.argv[1:]
dataset_classes = {
    "HallusionBench": "ImageYORNDataset",
    "AMBER": "ImageYORNDataset",
    "CRPE_EXIST": "CRPE",
    "CRPE_RELATION": "CRPE",
    "R-Bench-Dis": "ImageMCQDataset",
    "R-Bench-Ref": "ImageMCQDataset",
    "BLINK": "ImageMCQDataset",
}
missing = [dataset for dataset in datasets if dataset not in dataset_classes]
if missing:
    raise SystemExit(f"Unsupported benchmark config mapping: {missing}")
config = {
    "model": {
        model_alias: {
            "class": model_class,
            "model_path": model_path,
            "do_sample": False,
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": int(max_new_tokens),
        }
    },
    "data": {
        dataset: {"class": dataset_classes[dataset], "dataset": dataset}
        for dataset in datasets
    },
}
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
    f.write("\n")
PY
}

torchrun_clean_env=(
    env
    -u RANK
    -u LOCAL_RANK
    -u WORLD_SIZE
    -u LOCAL_WORLD_SIZE
    -u GROUP_RANK
    -u GROUP_WORLD_SIZE
    -u ROLE_RANK
    -u ROLE_WORLD_SIZE
    -u NODE_RANK
    -u MASTER_ADDR
    -u MASTER_PORT
    -u PET_NNODES
    -u PET_NODE_RANK
    -u PET_MASTER_ADDR
    -u PET_MASTER_PORT
    -u TORCH_NPROC_PER_NODE
    -u TORCHELASTIC_ERROR_FILE
    -u TORCHELASTIC_MAX_RESTARTS
    -u TORCHELASTIC_RESTART_COUNT
    -u TORCHELASTIC_RUN_ID
    -u TORCHELASTIC_USE_AGENT_STORE
)

run_config() {
    local config_path="$1"
    local judge="$2"
    local args=(
        --config "${config_path}"
        --work-dir "${work_dir}"
        --mode "${mode}"
        --judge "${judge}"
        --api-nproc "${api_nproc}"
        --verbose
    )
    if (( reuse )); then
        args+=(--reuse)
    fi
    if [[ -n "${reuse_aux}" ]]; then
        args+=(--reuse-aux "${reuse_aux}")
    fi
    if (( use_vllm )); then
        args+=(--use-vllm)
    fi
    echo "[vlmeval] model_family=${model_family}"
    echo "[vlmeval] model_alias=${model_alias}"
    echo "[vlmeval] model_class=${model_class}"
    echo "[vlmeval] model_path=${model_path}"
    echo "[vlmeval] root=${VLMEVALKIT_ROOT}"
    echo "[vlmeval] config=${config_path}"
    echo "[vlmeval] work_dir=${work_dir}"
    echo "[vlmeval] nproc_per_node=${GPU_COUNT}"
    echo "[vlmeval] reuse_aux=${reuse_aux:-<vlmeval-default>}"
    echo "[vlmeval] force_attn_implementation=${ICVCO_FORCE_ATTN_IMPLEMENTATION}"
    export MODEL_PATH="${model_path}"
    cd "${VLMEVALKIT_ROOT}"
    "${torchrun_clean_env[@]}" torchrun --standalone --nproc-per-node="${GPU_COUNT}" \
        -m icvco.vlmeval_entry "${args[@]}"
    cd "${REPO_ROOT}"
}

read -r -a yn_benchmarks <<< "${yn_data}"
read -r -a mcq_benchmarks <<< "${mcq_data}"

if ((${#yn_benchmarks[@]})); then
    configure_yn_judge_env
    yn_config="${work_dir}/_vlmeval_configs/${model_alias}_yn_max_new_tokens_${max_new_tokens}.json"
    write_eval_config "${yn_config}" "${yn_benchmarks[@]}"
    run_config "${yn_config}" "${yn_judge}"
fi

if ((${#mcq_benchmarks[@]})); then
    mcq_config="${work_dir}/_vlmeval_configs/${model_alias}_mcq_max_new_tokens_${max_new_tokens}.json"
    write_eval_config "${mcq_config}" "${mcq_benchmarks[@]}"
    run_config "${mcq_config}" "${mcq_judge}"
fi
