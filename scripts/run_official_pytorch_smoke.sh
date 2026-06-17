#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
SELECTED_ENV="experiments/full_repro/pytorch_official/env_matrix/selected_env.sh"
if [[ -f "$SELECTED_ENV" ]]; then source "$SELECTED_ENV"; fi
OFFICIAL_REPO="${OFFICIAL_REPO:-/mnt/data/sj/projects/TADSR_official_pytorch}"
export OFFICIAL_PYTORCH_REPO="$OFFICIAL_REPO"
export PYTHONPATH="$OFFICIAL_REPO:${PYTHONPATH:-}"
VENV_DIR="${VENV_DIR:-${OFFICIAL_PYTORCH_VENV:-/mnt/data/sj/venvs/tadsr_official_pytorch}}"
WEIGHTS="${WEIGHTS:-/mnt/data/sj/checkpoints/TADSR/preset/weights}"
INPUT="${INPUT:-/mnt/data/sj/datasets/TADSR/smoke/input}"
OUT="${OUT_DIR:-experiments/full_repro/pytorch_official/smoke}"
mkdir -p "$OUT"
STATUS_JSON="$OUT/smoke_status.json"
STATUS_MD="$OUT/smoke_status.md"
required=(tadsr.pkl DAPE.pth ram_swin_large_14m.pth time_vae unet vae text_encoder tokenizer scheduler feature_extractor bert-base-uncased)
missing=()
for item in "${required[@]}"; do [[ -e "$WEIGHTS/$item" ]] || missing+=("$item"); done
write_status() {
  local status="$1"; local reason="$2"; local runtime="${3:-0}"
  local input_count=0 output_count=0 gpu_before="" gpu_after=""
  [[ -d "$INPUT" ]] && input_count=$(find "$INPUT" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.bmp' \) | wc -l)
  [[ -d "$OUT/output" ]] && output_count=$(find "$OUT/output" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.bmp' \) | wc -l)
  [[ -f "$OUT/nvidia_smi_before.log" ]] && gpu_before=$(grep -Eo '[0-9]+MiB /[ ]*[0-9]+MiB' "$OUT/nvidia_smi_before.log" | tr '\n' ';' || true)
  [[ -f "$OUT/nvidia_smi_after.log" ]] && gpu_after=$(grep -Eo '[0-9]+MiB /[ ]*[0-9]+MiB' "$OUT/nvidia_smi_after.log" | tr '\n' ';' || true)
  python3 - <<PY
import json
data={
  "status": "$status",
  "reason": "$reason",
  "input_count": int("$input_count"),
  "output_count": int("$output_count"),
  "all_inputs_have_outputs": int("$input_count") > 0 and int("$output_count") >= int("$input_count"),
  "runtime_sec": float("$runtime"),
  "avg_time_sec": (float("$runtime") / int("$input_count")) if int("$input_count") else None,
  "gpu_memory_before": "$gpu_before",
  "gpu_memory_after": "$gpu_after",
}
json.dump(data, open("$STATUS_JSON", "w"), indent=2)
json.dump({"status": data["status"], "runtime_sec": data["runtime_sec"], "avg_time_sec": data["avg_time_sec"], "input_count": data["input_count"], "output_count": data["output_count"]}, open("$OUT/runtime.json", "w"), indent=2)
with open("$STATUS_MD", "w") as f:
    f.write("# Official PyTorch Smoke Status\\n\\n")
    f.write(f"Status: **{data['status']}**\\n\\n")
    f.write(f"Reason: {data['reason']}\\n\\n")
    f.write(f"Input images: {data['input_count']}\\n\\n")
    f.write(f"Output images: {data['output_count']}\\n\\n")
    f.write(f"All inputs have outputs: {data['all_inputs_have_outputs']}\\n\\n")
    f.write(f"Runtime sec: {data['runtime_sec']}\\n\\n")
    f.write(f"Average time sec: {data['avg_time_sec']}\\n")
PY
}

ensure_official_weights_link() {
  local link="$OFFICIAL_REPO/weights"
  if [[ -L "$link" ]]; then
    local target
    target=$(readlink "$link")
    if [[ "$target" != "$WEIGHTS" ]]; then
      rm -f "$link"
      ln -s "$WEIGHTS" "$link"
    fi
  elif [[ -e "$link" ]]; then
    echo "OFFICIAL_REPO_WEIGHTS_PATH_CONFLICT: $link exists and is not a symlink" >&2
    { echo "# Official Smoke Failure Analysis"; echo; echo "Failure type: path error"; echo; echo "Official repo already has a non-symlink weights path: $link"; } > "$OUT/failure_analysis.md"
    write_status "PYTORCH_OFFICIAL_PATH_FAIL" "official repo weights path conflict"
    exit 2
  else
    ln -s "$WEIGHTS" "$link"
  fi
}
if [[ ! -x "$VENV_DIR/bin/python" ]]; then write_status "PYTORCH_OFFICIAL_ENV_FAIL" "official venv missing"; exit 2; fi
if [[ ${#missing[@]} -gt 0 ]]; then echo "PYTORCH_OFFICIAL_ASSETS_MISSING: ${missing[*]}" | tee "$OUT/BLOCKED.txt"; { echo "# Official Smoke Failure Analysis"; echo; echo "Failure type: missing weight"; echo; echo "Missing: ${missing[*]}"; } > "$OUT/failure_analysis.md"; write_status "PYTORCH_OFFICIAL_ASSETS_MISSING" "missing: ${missing[*]}"; exit 2; fi
if [[ ! -d "$OFFICIAL_REPO" || ! -f "$OFFICIAL_REPO/test_tadsr.py" ]]; then write_status "PYTORCH_OFFICIAL_REPO_MISSING" "missing $OFFICIAL_REPO/test_tadsr.py"; exit 2; fi
ensure_official_weights_link
if [[ ! -d "$INPUT" ]] || ! find "$INPUT" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.bmp' \) | grep -q .; then
  python3 scripts/prepare_official_test_inputs.py --output "$INPUT" > "$OUT/input_prepare.log" 2>&1 || { write_status "PYTORCH_OFFICIAL_INPUT_MISSING" "no input images"; exit 2; }
fi
source "$VENV_DIR/bin/activate"
python - <<'PY' > "$OUT/import_test.log" 2>&1
import torch
import diffusers
import transformers
import peft
import cv2
import PIL
print("official imports ok")
PY
if [[ $? -ne 0 ]]; then write_status "PYTORCH_OFFICIAL_ENV_FAIL" "import test failed"; exit 2; fi
nvidia-smi > "$OUT/nvidia_smi_before.log" 2>&1 || true
env | sort > "$OUT/env.log"
cat > "$OUT/command.sh" <<EOF
cd "$OFFICIAL_REPO"
python test_tadsr.py \\
  --input_image "$INPUT" \\
  --output_dir "$PWD/$OUT/output" \\
  --tadsr_path "$WEIGHTS/tadsr.pkl" \\
  --pretrained_model_name_or_path "$WEIGHTS" \\
  --ram_ft_path "$WEIGHTS/DAPE.pth" \\
  --ram_path "$WEIGHTS/ram_swin_large_14m.pth"
EOF
chmod +x "$OUT/command.sh"
start=$(date +%s)
set +e
bash "$OUT/command.sh" > "$OUT/stdout.log" 2> "$OUT/stderr.log"
status=$?
set -e
end=$(date +%s)
runtime=$((end-start))
nvidia-smi > "$OUT/nvidia_smi_after.log" 2>&1 || true
if [[ $status -ne 0 ]]; then { echo "# Official Smoke Failure Analysis"; echo; echo "Failure type: other traceback or runtime error"; echo; echo "See stderr.log and stdout.log."; tail -n 80 "$OUT/stderr.log"; } > "$OUT/failure_analysis.md"; write_status "PYTORCH_OFFICIAL_SMOKE_FAIL" "test_tadsr.py crashed; see stderr.log" "$runtime"; exit $status; fi
python3 scripts/make_official_visual_grid.py --input_dir "$OUT/output" --output "$OUT/visual_grid.png" >> "$OUT/postprocess.log" 2>&1 || true
python3 scripts/evaluate_official_outputs.py --output_dir "$OUT/output" --metrics_json "$OUT/metrics.json" >> "$OUT/postprocess.log" 2>&1 || true
write_status "PASS" "official smoke completed" "$runtime"
echo "OFFICIAL_SMOKE_PASS: $OUT"
