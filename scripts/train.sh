#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_DATASET="IC-VCO-Dataset"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/load_env.sh"
load_icvco_env "${REPO_ROOT}/.env"

GPU_COUNT="${ICVCO_GPU_COUNT:-8}"

usage() {
    cat >&2 <<'EOF'
Usage:
  bash scripts/train.sh --base-model <path-or-hf-id> [options]

Options:
  --base-model <path-or-hf-id>       Base model path or Hugging Face id.
  --model-family <name>              llava-interleave or llava-onevision. Default: llava-interleave
  --dataset <hf-id-or-path>          IC-VCO dataset package. Default: IC-VCO-Dataset
  --output-root <dir>                Output root. Default: outputs/<model_tag>
  --run-id <name>                    Run id. Default: rYYYYMMDD-HHMM_<model_tag>_single8

Environment overrides:
  ICVCO_DATASET                      Default: IC-VCO-Dataset
  ICVCO_ATTN_IMPLEMENTATION          Optional; unset by default to match the r20 runtime.
  LORA_TARGET_MODULES                Default: q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
  DATASET_NUM_PROC                   Default: 16
  SFT_NUM_TRAIN_EPOCHS               Default: 1
  SFT_PER_DEVICE_TRAIN_BATCH_SIZE    Default: 1
  SFT_GRADIENT_ACCUMULATION_STEPS    Default: 16 (8 GPUs -> global batch 128)
  SFT_LEARNING_RATE                  Default: 2e-5
  SFT_WARMUP_RATIO                   Default: 0.1
  SFT_LR_SCHEDULER_TYPE              Default: cosine
  SFT_GRADIENT_CHECKPOINTING         Default: false
  SFT_LORA_R                         Default: 128
  SFT_LORA_ALPHA                     Default: 256
  SFT_LOGGING_STEPS                  Default: 5
  SFT_REPORT_TO                      Default: none
  VCO_NUM_TRAIN_EPOCHS               Default: 1
  VCO_PER_DEVICE_TRAIN_BATCH_SIZE    Default: 1
  VCO_GRADIENT_ACCUMULATION_STEPS    Default: 8 (8 GPUs -> global batch 64)
  VCO_LEARNING_RATE                  Default: 5e-6
  VCO_WARMUP_RATIO                   Default: 0.1
  VCO_LR_SCHEDULER_TYPE              Default: cosine
  VCO_GRADIENT_CHECKPOINTING         Default: true
  VCO_LORA_R                         Default: 128
  VCO_LORA_ALPHA                     Default: 256
  VCO_SAVE_STRATEGY                  Default: steps
  VCO_SAVE_STEPS                     Default: 100
  VCO_SAVE_TOTAL_LIMIT               Default: 3
  VCO_LOGGING_STEPS                  Default: 5
  VCO_REPORT_TO                      Default: none
  ICVCO_SINGLE_WEIGHT                Default: 1.75
  ICVCO_MULTI_WEIGHT                 Default: 0.75
  ICVCO_ANCHOR_WEIGHT                Default: 1.0
  ICVCO_VCDIST_WEIGHT                Default: 0.4
  ICVCO_VCDIST_THRESHOLD             Default: 0.5
  ICVCO_BETA                         Default: 0.1
  ICVCO_SEED                         Default: 42
  SFT_MAX_SAMPLES                    Optional training subset size.
  SFT_MAX_STEPS                      Optional max optimizer steps.
  VCO_MAX_SAMPLES                    Optional training subset size.
  VCO_MAX_STEPS                      Optional max optimizer steps.
EOF
}

base_model="${ICVCO_BASE_MODEL:-models/llava-interleave-qwen-7b-hf}"
model_family="llava-interleave"
dataset="${ICVCO_DATASET:-${DEFAULT_DATASET}}"
output_root=""
run_id=""

while (($#)); do
    case "$1" in
        --base-model)
            base_model="${2:-}"
            shift 2
            ;;
        --model-family)
            model_family="${2:-}"
            shift 2
            ;;
        --dataset)
            dataset="${2:-}"
            shift 2
            ;;
        --output-root)
            output_root="${2:-}"
            shift 2
            ;;
        --run-id)
            run_id="${2:-}"
            shift 2
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

if [[ -z "${base_model}" ]]; then
    echo "--base-model is required." >&2
    usage
    exit 1
fi
if [[ -z "${dataset}" ]]; then
    echo "--dataset is required." >&2
    usage
    exit 1
fi
if [[ "${dataset}" == "${DEFAULT_DATASET}" && ! -e "${dataset}" && ! -e "${REPO_ROOT}/${dataset}" ]]; then
    echo "Default dataset path not found: ${REPO_ROOT}/${DEFAULT_DATASET}" >&2
    echo "Download OPPOer/IC-VCO-Dataset to the repository root as ${DEFAULT_DATASET}, or pass --dataset <hf-id-or-path>." >&2
    exit 1
fi

case "${model_family}" in
    llava-interleave)
        model_tag="llava_interleave"
        ;;
    llava-onevision)
        model_tag="llava_onevision"
        ;;
    *)
        echo "Unsupported --model-family: ${model_family}" >&2
        echo "Supported values: llava-interleave, llava-onevision" >&2
        exit 1
        ;;
esac

if [[ -z "${output_root}" ]]; then
    output_root="outputs/${model_tag}"
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${output_root}" != /* ]]; then
    output_root="${REPO_ROOT}/${output_root}"
fi
if [[ -z "${run_id}" ]]; then
    run_id="$(date +r%Y%m%d-%H%M)_${model_tag}_single8"
fi
if [[ "${base_model}" != /* && -e "${base_model}" ]]; then
    base_model="$(cd "$(dirname "${base_model}")" && pwd)/$(basename "${base_model}")"
fi
if [[ "${dataset}" != /* ]]; then
    if [[ -e "${dataset}" ]]; then
        dataset="$(cd "$(dirname "${dataset}")" && pwd)/$(basename "${dataset}")"
    elif [[ -e "${REPO_ROOT}/${dataset}" ]]; then
        dataset="${REPO_ROOT}/${dataset}"
    fi
fi

run_dir="${output_root}/${run_id}"
sft_output_dir="${run_dir}/sft"
icvco_output_dir="${run_dir}/icvco"
mkdir -p "${run_dir}"

attn_implementation="${ICVCO_ATTN_IMPLEMENTATION:-}"
lora_target_modules="${LORA_TARGET_MODULES:-q_proj k_proj v_proj o_proj gate_proj up_proj down_proj}"
read -r -a lora_target_modules_array <<< "${lora_target_modules}"
lora_target_args=()
if ((${#lora_target_modules_array[@]})); then
    lora_target_args=(--lora_target_modules "${lora_target_modules_array[@]}")
fi
attn_args=()
if [[ -n "${attn_implementation}" ]]; then
    attn_args=(--attn_implementation "${attn_implementation}")
fi
sft_smoke_args=()
if [[ -n "${SFT_MAX_SAMPLES:-}" ]]; then
    sft_smoke_args+=(--max_samples "${SFT_MAX_SAMPLES}")
fi
if [[ -n "${SFT_MAX_STEPS:-}" ]]; then
    sft_smoke_args+=(--max_steps "${SFT_MAX_STEPS}")
fi
vco_smoke_args=()
if [[ -n "${VCO_MAX_SAMPLES:-}" ]]; then
    vco_smoke_args+=(--max_samples "${VCO_MAX_SAMPLES}")
fi
if [[ -n "${VCO_MAX_STEPS:-}" ]]; then
    vco_smoke_args+=(--max_steps "${VCO_MAX_STEPS}")
fi

sft_per_device_train_batch_size="${SFT_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
sft_gradient_accumulation_steps="${SFT_GRADIENT_ACCUMULATION_STEPS:-16}"
vco_per_device_train_batch_size="${VCO_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
vco_gradient_accumulation_steps="${VCO_GRADIENT_ACCUMULATION_STEPS:-8}"
sft_global_batch_size=$((GPU_COUNT * sft_per_device_train_batch_size * sft_gradient_accumulation_steps))
vco_global_batch_size=$((GPU_COUNT * vco_per_device_train_batch_size * vco_gradient_accumulation_steps))

accelerate_config="${REPO_ROOT}/configs/accelerate/deepspeed_zero2.yaml"
accelerate_args=(
    --config_file "${accelerate_config}"
    --num_processes "${GPU_COUNT}"
    --num_machines 1
)

common_model_args=(
    --model_family_id "${model_family}"
    --trust_remote_code true
    --dtype bfloat16
    "${attn_args[@]}"
    --use_peft true
)

echo "[icvco] model_family=${model_family}"
echo "[icvco] base_model=${base_model}"
echo "[icvco] dataset=${dataset}"
echo "[icvco] run_dir=${run_dir}"
echo "[icvco] gpu_count=${GPU_COUNT}"
echo "[icvco] sft_config=sft"
echo "[icvco] sft_global_batch_size=${sft_global_batch_size}"
echo "[icvco] preference_config=preference"
echo "[icvco] vco_global_batch_size=${vco_global_batch_size}"

accelerate launch "${accelerate_args[@]}" -m icvco.cli.train_sft \
    --model_name_or_path "${base_model}" \
    "${common_model_args[@]}" \
    "${sft_smoke_args[@]}" \
    --lora_r "${SFT_LORA_R:-128}" \
    --lora_alpha "${SFT_LORA_ALPHA:-256}" \
    --lora_dropout "${SFT_LORA_DROPOUT:-0.05}" \
    "${lora_target_args[@]}" \
    --dataset_name "${dataset}" \
    --output_dir "${sft_output_dir}" \
    --logging_dir "${run_dir}/logs/sft" \
    --bf16 true \
    --gradient_checkpointing "${SFT_GRADIENT_CHECKPOINTING:-false}" \
    --dataset_num_proc "${DATASET_NUM_PROC:-16}" \
    --per_device_train_batch_size "${sft_per_device_train_batch_size}" \
    --gradient_accumulation_steps "${sft_gradient_accumulation_steps}" \
    --num_train_epochs "${SFT_NUM_TRAIN_EPOCHS:-1}" \
    --learning_rate "${SFT_LEARNING_RATE:-2e-5}" \
    --warmup_ratio "${SFT_WARMUP_RATIO:-0.1}" \
    --lr_scheduler_type "${SFT_LR_SCHEDULER_TYPE:-cosine}" \
    --logging_steps "${SFT_LOGGING_STEPS:-5}" \
    --save_strategy "${SFT_SAVE_STRATEGY:-epoch}" \
    --eval_strategy no \
    --report_to "${SFT_REPORT_TO:-none}" \
    --use_single_image true \
    --use_multi_image true

python3 -m icvco.cli.merge_ckpt --path "${sft_output_dir}"

accelerate launch "${accelerate_args[@]}" -m icvco.cli.train_vco \
    --model_name_or_path "${sft_output_dir}" \
    "${common_model_args[@]}" \
    "${vco_smoke_args[@]}" \
    --lora_r "${VCO_LORA_R:-128}" \
    --lora_alpha "${VCO_LORA_ALPHA:-256}" \
    --lora_dropout "${VCO_LORA_DROPOUT:-0.05}" \
    "${lora_target_args[@]}" \
    --dataset_name "${dataset}" \
    --output_dir "${icvco_output_dir}" \
    --logging_dir "${run_dir}/logs/icvco" \
    --bf16 true \
    --gradient_checkpointing "${VCO_GRADIENT_CHECKPOINTING:-true}" \
    --dataset_num_proc "${DATASET_NUM_PROC:-16}" \
    --per_device_train_batch_size "${vco_per_device_train_batch_size}" \
    --gradient_accumulation_steps "${vco_gradient_accumulation_steps}" \
    --num_train_epochs "${VCO_NUM_TRAIN_EPOCHS:-1}" \
    --learning_rate "${VCO_LEARNING_RATE:-5e-6}" \
    --warmup_ratio "${VCO_WARMUP_RATIO:-0.1}" \
    --lr_scheduler_type "${VCO_LR_SCHEDULER_TYPE:-cosine}" \
    --logging_steps "${VCO_LOGGING_STEPS:-5}" \
    --save_strategy "${VCO_SAVE_STRATEGY:-steps}" \
    --save_steps "${VCO_SAVE_STEPS:-100}" \
    --save_total_limit "${VCO_SAVE_TOTAL_LIMIT:-3}" \
    --eval_strategy no \
    --report_to "${VCO_REPORT_TO:-none}" \
    --use_single_image true \
    --use_multi_image true \
    --use_single_branch_token_mask true \
    --single_weight "${ICVCO_SINGLE_WEIGHT:-1.75}" \
    --multi_weight "${ICVCO_MULTI_WEIGHT:-0.75}" \
    --anchor_weight "${ICVCO_ANCHOR_WEIGHT:-1.0}" \
    --single_anchor_weight "${ICVCO_SINGLE_ANCHOR_WEIGHT:-1.75}" \
    --multi_anchor_weight "${ICVCO_MULTI_ANCHOR_WEIGHT:-0.75}" \
    --vcdist_weight "${ICVCO_VCDIST_WEIGHT:-0.4}" \
    --vcdist_threshold "${ICVCO_VCDIST_THRESHOLD:-0.5}" \
    --beta "${ICVCO_BETA:-0.1}" \
    --seed "${ICVCO_SEED:-42}" \
    --data_seed "${ICVCO_SEED:-42}"

echo "[icvco] finished"
echo "[icvco] sft_checkpoint=${sft_output_dir}"
echo "[icvco] icvco_checkpoint=${icvco_output_dir}"
