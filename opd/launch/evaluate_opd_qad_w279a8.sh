#!/usr/bin/env bash
set -euo pipefail

# Path roots default to this file's location so a QAOPD checkout stays self-contained.
QAOPD_ROOT=${QAOPD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
ROOT=${ROOT:-$QAOPD_ROOT}
VERL_ROOT=${VERL_ROOT:-$QAOPD_ROOT/third_party/verl}

EDGE_ROOT=${EDGE_ROOT:-$ROOT/third_party/edgerazor}
EVAL_ENV=${EVAL_ENV:-}
PY=${PY:-${EVAL_ENV:+$EVAL_ENV/bin/}python}
ACCELERATE=${ACCELERATE:-${EVAL_ENV:+$EVAL_ENV/bin/}accelerate}

HF_CHECKPOINT=${HF_CHECKPOINT:-${1:-}}
RUN_NAME=${RUN_NAME:-${2:-}}
QAT_CONFIG=${QAT_CONFIG:-$QAOPD_ROOT/configs/opd/edgerazor_w2_79a8_qwen3.yaml}
EVAL_TOKENIZER_SOURCE=${EVAL_TOKENIZER_SOURCE:-${MODELS_DIR:-$QAOPD_ROOT/models}/w279_step10000}

if [[ -z "$HF_CHECKPOINT" || -z "$RUN_NAME" ]]; then
    echo "Usage: bash $0 <actor/huggingface> <run-name>" >&2
    exit 2
fi

OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/experiments/opd_qad/evaluation/$RUN_NAME}
MODEL_LABEL=${MODEL_LABEL:-w279a8kv16}
MODEL_DIR=${MODEL_DIR:-$OUTPUT_ROOT/model_$MODEL_LABEL}
RESULT_ROOT=${RESULT_ROOT:-$OUTPUT_ROOT/results/$MODEL_LABEL}
LOG_ROOT=${LOG_ROOT:-$OUTPUT_ROOT/logs/$MODEL_LABEL}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
NUM_PROCESSES=${NUM_PROCESSES:-1}
ACCELERATE_MAIN_PROCESS_PORT=${ACCELERATE_MAIN_PROCESS_PORT:-}
MAIN_BATCH=${MAIN_BATCH:-16}
CHAT_BATCH=${CHAT_BATCH:-4}
GSM_BATCH=${GSM_BATCH:-$MAIN_BATCH}
HUMANEVAL_BATCH=${HUMANEVAL_BATCH:-$CHAT_BATCH}
IFEVAL_BATCH=${IFEVAL_BATCH:-$CHAT_BATCH}
LIMIT=${LIMIT:-}
RUN_CONVERT=${RUN_CONVERT:-1}
RUN_LOAD_SMOKE=${RUN_LOAD_SMOKE:-1}
RUN_GSM8K=${RUN_GSM8K:-1}
RUN_FULL=${RUN_FULL:-0}
RUN_CHAT=${RUN_CHAT:-0}
RUN_HUMANEVAL=${RUN_HUMANEVAL:-0}
RUN_IFEVAL=${RUN_IFEVAL:-0}
RESTORE_EVAL_TOKENIZER=${RESTORE_EVAL_TOKENIZER:-1}
FULL_TASKS=${FULL_TASKS:-EdgeRazor_Eval_QLLM}

CONVERT_SCRIPT=$EDGE_ROOT/example/edgerazor-llm/src/convert/convert_qweight.py
MODEL_WRAPPER=$EDGE_ROOT/src/edgerazor/templates/modeling_edgerazor.py
INCLUDE_PATH=${LM_EVAL_INCLUDE_PATH:-$QAOPD_ROOT/eval_tasks}
CHECKPOINT_VALIDATOR=$QAOPD_ROOT/opd/tools/validate_qat_latent_checkpoint.py

for path in "$HF_CHECKPOINT" "$QAT_CONFIG" "$PY" "$ACCELERATE" "$CONVERT_SCRIPT" "$MODEL_WRAPPER" "$CHECKPOINT_VALIDATOR"; do
    if [[ ! -e "$path" ]]; then
        echo "Required path does not exist: $path" >&2
        exit 3
    fi
done
if [[ "$RUN_CONVERT" != "1" && ! -f "$MODEL_DIR/config.json" ]]; then
    echo "Evaluation model is missing config.json: $MODEL_DIR" >&2
    echo "Set RUN_CONVERT=1 or point MODEL_DIR at an existing export." >&2
    exit 3
fi
CHAT_TEMPLATE_SRC="$EVAL_TOKENIZER_SOURCE/chat_template.jinja"
if [[ "$RESTORE_EVAL_TOKENIZER" == "1" && ! -f "$CHAT_TEMPLATE_SRC" ]]; then
    # Stock HF releases (models/Qwen3-1.7B, say) still carry the template inside
    # tokenizer_config.json; only transformers>=4.51 writes it out as a separate
    # file. Requiring the file means a base model cannot be used as the tokenizer
    # source at all, which blocks exporting an un-QAD'd reference. Extract it.
    CHAT_TEMPLATE_SRC="$OUTPUT_ROOT/.chat_template.jinja"
    mkdir -p "$OUTPUT_ROOT"
    "$PY" - "$EVAL_TOKENIZER_SOURCE" "$CHAT_TEMPLATE_SRC" <<'PY' || exit 3
import json, sys
src, dst = sys.argv[1:3]
tpl = json.load(open(f"{src}/tokenizer_config.json")).get("chat_template")
if not tpl:
    sys.exit(f"no chat_template.jinja and no chat_template in {src}/tokenizer_config.json")
open(dst, "w").write(tpl)
print(f"extracted chat template from {src}/tokenizer_config.json ({len(tpl)} chars)")
PY
fi

mkdir -p "$OUTPUT_ROOT" "$RESULT_ROOT" "$LOG_ROOT"
export PYTHONPATH="$EDGE_ROOT/src:${PYTHONPATH:-}"
export HF_ALLOW_CODE_EVAL=1
export HF_EVALUATE_OFFLINE=${HF_EVALUATE_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES

if [[ "$RUN_CONVERT" == "1" ]]; then
    "$PY" "$CHECKPOINT_VALIDATOR" --model "$HF_CHECKPOINT" --purpose "deployment export"

    rm -rf "$MODEL_DIR"
    mkdir -p "$MODEL_DIR"
    for source in "$HF_CHECKPOINT"/*; do
        if [[ -f "$source" || -L "$source" ]]; then
            cp -a "$source" "$MODEL_DIR/"
        fi
    done
    rm -f "$MODEL_DIR"/model*.safetensors "$MODEL_DIR/model.safetensors.index.json"

    "$PY" "$CONVERT_SCRIPT" \
        --quant_config "$QAT_CONFIG" \
        --unquantized_model "$HF_CHECKPOINT" \
        --quantized_model "$MODEL_DIR" \
        --dtype bfloat16 \
        --device cuda:0 2>&1 | tee -a "$LOG_ROOT/convert.log"

    cp "$MODEL_WRAPPER" "$MODEL_DIR/modeling_edgerazor.py"
    "$PY" - "$MODEL_DIR/config.json" "$QAT_CONFIG" <<'PY'
import json
import sys

import yaml

config_path, qat_path = sys.argv[1:]
with open(config_path, encoding="utf-8") as handle:
    config = json.load(handle)
with open(qat_path, encoding="utf-8") as handle:
    qat = yaml.safe_load(handle)["qat_configuration"]

qat.pop("method", None)
qat["function"]["is_w_quantized"] = True
config.setdefault("auto_map", {})["AutoModelForCausalLM"] = (
    "modeling_edgerazor.EdgeRazorForCausalLM"
)
config["edgerazor_config"] = {"qat_configuration": qat}
config.pop("quant_mode", None)
config.pop("quantization_config", None)

with open(config_path, "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2)
    handle.write("\n")
PY

    if cmp -s "$MODEL_DIR/model.safetensors" "$HF_CHECKPOINT/model.safetensors"; then
        echo "Converted weights are byte-identical to the latent checkpoint" >&2
        exit 4
    fi
fi

if [[ "$RESTORE_EVAL_TOKENIZER" == "1" ]]; then
    # Training intentionally uses a content-only template for distillation. The
    # standard Qwen protocol must be restored before chat/instruct evaluation.
    # Vocabulary-bearing files must be byte-identical -- a resized or swapped
    # vocabulary silently invalidates every score.
    tokenizer_files=(
        added_tokens.json
        merges.txt
        special_tokens_map.json
        tokenizer.json
        vocab.json
    )
    for filename in "${tokenizer_files[@]}"; do
        source_file="$EVAL_TOKENIZER_SOURCE/$filename"
        model_file="$MODEL_DIR/$filename"
        if [[ -f "$source_file" && ! -f "$model_file" ]]; then
            echo "Evaluation model is missing tokenizer asset: $model_file" >&2
            exit 4
        fi
        if [[ -f "$source_file" ]] && ! cmp -s "$source_file" "$model_file"; then
            echo "Tokenizer asset differs from evaluation source: $filename" >&2
            exit 4
        fi
    done

    # tokenizer_config.json cannot be compared byte-wise: any trainer that calls
    # save_pretrained under transformers>=4.53 re-serializes it, moving
    # chat_template out to chat_template.jinja and adding defaults the source file
    # never had. EdgeRazor's QAD trainer does exactly that, so byte comparison
    # rejects a perfectly good checkpoint. Compare the fields that would actually
    # change tokenization instead; the template itself is overwritten below and
    # then smoke-tested against the source.
    "$PY" - "$EVAL_TOKENIZER_SOURCE/tokenizer_config.json" \
            "$MODEL_DIR/tokenizer_config.json" <<'PY' || exit 4
import json
import sys

src, mod = (json.load(open(p, encoding="utf-8")) for p in sys.argv[1:])
# chat_template: restored from CHAT_TEMPLATE_SRC a few lines down.
# extra_special_tokens/use_cache: serialization defaults, no effect on encoding.
ignore = {"chat_template", "extra_special_tokens", "use_cache"}
diff = sorted(k for k in (set(src) | set(mod)) - ignore
              if src.get(k) != mod.get(k))
if diff:
    sys.exit("tokenizer_config.json differs from evaluation source in "
             f"{diff} -- refusing to evaluate with a mismatched tokenizer")
PY
    cp -f "$CHAT_TEMPLATE_SRC" "$MODEL_DIR/chat_template.jinja"

    "$PY" - "$EVAL_TOKENIZER_SOURCE" "$MODEL_DIR" <<'PY' \
        2>&1 | tee -a "$LOG_ROOT/tokenizer_restore.log"
import sys

from transformers import AutoTokenizer

source_path, model_path = sys.argv[1:]
messages = [{"role": "user", "content": "Template restoration smoke test."}]

source = AutoTokenizer.from_pretrained(source_path, trust_remote_code=True)
restored = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
source_prompt = source.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
restored_prompt = restored.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

if restored_prompt != source_prompt:
    raise RuntimeError("Restored evaluation chat template does not match its source")
# This check used to look for Qwen's ChatML markers only. Its purpose -- confirm
# that the restored template is a chat template and not the plain-text training
# template ({{ message.content }}, which renders without any role markers) -- is
# architecture independent, but the markers themselves are family specific.
# Other families use their own (header-id style, or plain <|user|>/<|assistant|>).
# So match per family and fail only when none match; Qwen behaves exactly as before.
PROTOCOLS = {
    "Qwen ChatML": ("<|im_start|>user", "<|im_start|>assistant"),
    "Llama-3": ("<|start_header_id|>user", "<|start_header_id|>assistant"),
    "Falcon3": ("<|user|>", "<|assistant|>"),
}
matched = [n for n, marks in PROTOCOLS.items() if all(m in restored_prompt for m in marks)]
if not matched:
    raise RuntimeError(
        "Restored evaluation chat template matches no known chat protocol "
        f"(tried {', '.join(PROTOCOLS)}); rendered prompt starts with: {restored_prompt[:120]!r}"
    )
print(f"Chat protocol detected: {matched[0]}")

print(f"Restored evaluation tokenizer from {source_path}")
print("Evaluation chat-template rendering matches the QAD warm start")
PY
fi

if [[ "$RUN_LOAD_SMOKE" == "1" ]]; then
    "$PY" - "$MODEL_DIR" <<'PY' 2>&1 | tee -a "$LOG_ROOT/load_smoke.log"
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from edgerazor.qat.block.qkv_cache import QuantizedKVState

path = sys.argv[1]
with (Path(path) / "config.json").open(encoding="utf-8") as handle:
    model_config = json.load(handle)
target_types = (
    model_config.get("edgerazor_config", {})
    .get("qat_configuration", {})
    .get("select", {})
    .get("target_types", [])
)
expects_kv_quant = "kv_cache" in target_types
kv_updates = 0
original_update = QuantizedKVState.update

def counting_update(self, *args, **kwargs):
    global kv_updates
    kv_updates += 1
    return original_update(self, *args, **kwargs)

QuantizedKVState.update = counting_update
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
)
inputs = tokenizer("Question: What is 2 + 3?\nAnswer:", return_tensors="pt").to("cuda:0")
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=8, do_sample=False)
print(tokenizer.decode(output[0], skip_special_tokens=True))
if expects_kv_quant != (kv_updates > 0):
    raise RuntimeError(
        f"KV quantization runtime mismatch: expected={expects_kv_quant}, "
        f"QuantizedKVState.update calls={kv_updates}"
    )
print(f"KV quantization verified: enabled={expects_kv_quant}, updates={kv_updates}")
PY
fi

common_args=(
    --model hf
    --model_args "pretrained=$MODEL_DIR,trust_remote_code=True,dtype=bfloat16"
    --include_path "$INCLUDE_PATH"
    --confirm_run_unsafe_code
    --trust_remote_code
    --log_samples
)
# -- Evaluation protocol: chat format vs plain-text continuation ---------------
# Flipping only this switch on one set of weights moved a 3B instruct model at
# W2.79 from 17.74 to 49.43 on GSM8K and from 0.00 to 18.40 on MATH-500. Under a
# non-chat prompt the quantized model falls into repetition loops and never
# reaches an answer; its BF16 counterpart does not (64.90 plain / 33.00
# MATH-500). So this is format brittleness introduced by quantization, not lost
# capability. The Qwen line is fine under both protocols and all of its
# historical numbers were measured plain, so switching would make old and new
# numbers incomparable. Hence a per-family split rather than one global choice.
#
# The important part is that it can no longer be got wrong by forgetting an
# argument: the default is auto, the family is detected from the model's own chat
# template, and whichever branch is taken is written to the log so every number
# carries its protocol with it.
EVAL_PROTOCOL=${EVAL_PROTOCOL:-auto}
if [[ "$EVAL_PROTOCOL" == "auto" ]]; then
    EVAL_FAMILY=$("$PY" - "$MODEL_DIR" <<'PY'
import sys
from transformers import AutoTokenizer
try:
    tk = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True)
    p = tk.apply_chat_template([{"role": "user", "content": "x"}],
                               tokenize=False, add_generation_prompt=True)
except Exception:
    print("unknown")
    raise SystemExit(0)
# The criterion is which protocol a family's historical numbers were measured
# under, not what the template looks like:
#   qwen  -> plain. Every historical Qwen3 number is plain; switching would break
#            comparability. Measured, Qwen is also not affected by the format
#            brittleness after quantization (W1.88: 8.33 plain / 5.00 chat).
#   other chat models -> chat. On another family the same weights differed by
#            31.7 points (17.74 vs 49.43): quantization sends the model into
#            repetition loops under a non-chat prompt. New families take this branch.
# NOTE: some non-Qwen families also use ChatML markers, so <|im_start|> alone
#   cannot identify Qwen; matching on it would route them to plain and repeat the
#   underestimate above. Exclude by model name first.
# NOTE: _name_or_path and the directory name are not enough. What is evaluated is
#   the *export*, whose config.json has no _name_or_path, and the export directory
#   is named unieval_<tag>/model_w279a8kv16 -- the tag does not necessarily
#   contain "qwen" (the 4B line used the tag w4bqad). Qwen3-4B was therefore
#   classified as chatml_other, ran under chat, and silently scored 11.0 too high
#   on MATH-500 and 6.5 too low on GSM8K. architectures / model_type are always
#   present and authoritative in an export, so they are part of the criterion too.
import json, os
name = ""
arch = ""
mtype = ""
try:
    cfg = json.load(open(os.path.join(sys.argv[1], "config.json")))
    name = cfg.get("_name_or_path", "") or ""
    arch = " ".join(cfg.get("architectures") or [])
    mtype = cfg.get("model_type", "") or ""
except Exception:
    pass
low = (name + " " + arch + " " + mtype + " " + sys.argv[1]).lower()
if "qwen" in low:
    print("qwen")
elif "<|start_header_id|>assistant" in p:
    print("llama3")
elif "<|im_start|>assistant" in p:
    print("chatml_other")
elif "<|assistant|>" in p:
    print("falcon3")
else:
    print("unknown")
PY
)
    case "$EVAL_FAMILY" in
        qwen)                              EVAL_PROTOCOL=plain ;;
        llama3|chatml_other|falcon3)       EVAL_PROTOCOL=chat ;;
        # Prefer to stop when unsure: defaulting to plain silently underestimates
        # by two to three times.
        *)  echo "!! cannot determine the chat protocol (family=$EVAL_FAMILY). Set EVAL_PROTOCOL=plain or chat explicitly." >&2
            exit 2 ;;
    esac
    echo "Eval protocol: $EVAL_PROTOCOL (auto, family=$EVAL_FAMILY)"
else
    echo "Eval protocol: $EVAL_PROTOCOL (explicit)"
fi
if [[ "$EVAL_PROTOCOL" == "chat" ]]; then
    common_args+=(--apply_chat_template --fewshot_as_multiturn)
fi

# The port is picked here rather than by the caller because everything above
# this line -- the export, the quantization, the load smoke test -- takes
# minutes, and a port reserved before all that is no longer a reservation.
# An explicit ACCELERATE_MAIN_PROCESS_PORT still wins, for callers that need a
# known port.
if [[ -z "$ACCELERATE_MAIN_PROCESS_PORT" ]]; then
    source "$QAOPD_ROOT/scripts/lib/port.sh"
    ACCELERATE_MAIN_PROCESS_PORT=$(free_port) || {
        echo "could not find a free port for accelerate" >&2; exit 3; }
    echo "accelerate rendezvous port: $ACCELERATE_MAIN_PROCESS_PORT (auto)"
fi
accelerate_launch_args=(launch --num_processes="$NUM_PROCESSES")
accelerate_launch_args+=(--main_process_port "$ACCELERATE_MAIN_PROCESS_PORT")
if [[ -n "$LIMIT" ]]; then
    common_args+=(--limit "$LIMIT")
fi

if [[ "$RUN_GSM8K" == "1" ]]; then
    "$ACCELERATE" "${accelerate_launch_args[@]}" -m lm_eval \
        "${common_args[@]}" \
        --tasks gsm8k \
        --batch_size "$GSM_BATCH" \
        --output_path "$RESULT_ROOT/gsm8k" \
        2>&1 | tee -a "$LOG_ROOT/gsm8k.log"
fi

RUN_MATH500=${RUN_MATH500:-0}
if [[ "$RUN_MATH500" == "1" ]]; then
    "$ACCELERATE" "${accelerate_launch_args[@]}" -m lm_eval \
        "${common_args[@]}" \
        --tasks math500 \
        --batch_size "${MATH500_BATCH:-16}" \
        --output_path "$RESULT_ROOT/math500" \
        2>&1 | tee -a "$LOG_ROOT/math500.log"
fi

RUN_AMC23=${RUN_AMC23:-0}
if [[ "$RUN_AMC23" == "1" ]]; then
    # 32k generation budget (fits: max_position_embeddings 40960 leaves 8k for
    # the 4-shot prompt); batch kept small because the KV cache at this length
    # is ~4.6GB per sequence.
    "$ACCELERATE" "${accelerate_launch_args[@]}" -m lm_eval \
        "${common_args[@]}" \
        --tasks amc23 \
        --batch_size "${AMC23_BATCH:-8}" \
        --output_path "$RESULT_ROOT/amc23" \
        2>&1 | tee -a "$LOG_ROOT/amc23.log"
fi

if [[ "$RUN_FULL" == "1" ]]; then
    "$ACCELERATE" "${accelerate_launch_args[@]}" -m lm_eval \
        "${common_args[@]}" \
        --tasks "$FULL_TASKS" \
        --batch_size "$MAIN_BATCH" \
        --output_path "$RESULT_ROOT/EdgeRazor_Eval_QLLM_full" \
        2>&1 | tee -a "$LOG_ROOT/main_eval.log"
fi

if [[ "$RUN_CHAT" == "1" ]]; then
    "$ACCELERATE" "${accelerate_launch_args[@]}" -m lm_eval \
        "${common_args[@]}" \
        --apply_chat_template \
        --tasks EdgeRazor_Eval_QLLM_Chat \
        --batch_size "$CHAT_BATCH" \
        --output_path "$RESULT_ROOT/EdgeRazor_Eval_QLLM_Chat_full" \
        2>&1 | tee -a "$LOG_ROOT/chat_eval.log"
fi

if [[ "$RUN_HUMANEVAL" == "1" ]]; then
    "$ACCELERATE" "${accelerate_launch_args[@]}" -m lm_eval \
        "${common_args[@]}" \
        --apply_chat_template \
        --tasks humaneval_instruct \
        --batch_size "$HUMANEVAL_BATCH" \
        --output_path "$RESULT_ROOT/humaneval_instruct" \
        2>&1 | tee -a "$LOG_ROOT/humaneval_instruct.log"
fi

if [[ "$RUN_IFEVAL" == "1" ]]; then
    "$ACCELERATE" "${accelerate_launch_args[@]}" -m lm_eval \
        "${common_args[@]}" \
        --apply_chat_template \
        --tasks ifeval_qwen3_instruct \
        --batch_size "$IFEVAL_BATCH" \
        --output_path "$RESULT_ROOT/ifeval_qwen3_instruct" \
        2>&1 | tee -a "$LOG_ROOT/ifeval_qwen3_instruct.log"
fi

echo "Evaluation output: $OUTPUT_ROOT"
