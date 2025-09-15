#!/usr/bin/env bash
# run_latency_fees_sensitivity.sh
# Requirements: jq
set -euo pipefail

JSON_PATH="${1:-./sensitivity_latency_fees.json}"
RUN_DIR="${RUN_DIR:-../run}"
REPROD_DIR="${REPROD_DIR:-../evaluation_check}"
CONFIG="${RUN_DIR}/run_config.json"

RUN_SCRIPT="${RUN_DIR}/run_simulation.py"
EVAL_SCRIPT="${REPROD_DIR}/latency_fees_sensitivity.py"

if ! command -v jq >/dev/null 2>&1; then
  echo "[ERR] 'jq' not found. Install jq (e.g., 'brew install jq' or 'apt-get install jq')." >&2
  exit 1
fi
[[ -f "$JSON_PATH" ]] || { echo "[ERR] JSON not found: $JSON_PATH" >&2; exit 1; }
[[ -f "$CONFIG"   ]] || { echo "[ERR] run_config.json not found at: $CONFIG" >&2; exit 1; }

# Normalize JSON: letency → latency
JSON_BAK="${JSON_PATH}.bak.$(date +%s)"
cp -f "$JSON_PATH" "$JSON_BAK"
tmp_json="$(mktemp)"
jq 'to_entries
    | map(.value.latency = (.value.latency // .value.letency) | .value |= del(.letency))
    | from_entries' "$JSON_PATH" > "$tmp_json"
mv "$tmp_json" "$JSON_PATH"
echo "[INFO] Normalized JSON (letency→latency). Backup: $JSON_BAK"

# Backup run_config.json once
CONF_BAK="${CONFIG}.bak.$(date +%s)"
cp -f "$CONFIG" "$CONF_BAK"
echo "[INFO] Backed up run_config.json -> $CONF_BAK"

update_config() {
  local latency="$1" fees="$2" label_value="$3" agent_type="$4"
  local tmp; tmp="$(mktemp)"
  # NOTE: set eval_log_flag=true here
  jq \
    --arg LBL  "$label_value" \
    --arg TYPE "$agent_type" \
    --arg MODE "pretrained" \
    --arg SIM  "eval" \
    --argjson LAT "$latency" \
    --argjson FEE "$fees" \
    '
      .["latency_micros"]  = $LAT
      | .["cost_in_bps"]   = $FEE
      | .["label"]         = $LBL
      | .agent.type        = $TYPE
      | .agent.mode        = $MODE
      | .["simulator_mode"]= $SIM
      | .["eval_log_flag"] = true
    ' "$CONFIG" > "$tmp"
  mv "$tmp" "$CONFIG"
}

# Iterate keys (portable; no process substitution)
jq -r 'keys[]' "$JSON_PATH" | while IFS= read -r key; do
  fees="$(jq -r --arg k "$key" '.[$k].fees'    "$JSON_PATH")"
  latency="$(jq -r --arg k "$key" '.[$k].latency' "$JSON_PATH")"

  if [[ -z "$fees" || "$fees" == "null" || -z "$latency" || "$latency" == "null" ]]; then
    echo "[ERR] Missing 'fees' or 'latency' for run '$key' in $JSON_PATH" >&2
    exit 1
  fi

  fee_tag="${fees//./p}"
  suffix="_l_${latency}_c_${fee_tag}"

  echo
  echo "========== $key  (latency_micros=${latency}, cost_in_bps=${fees}) =========="

  # DQNc
  dqn_label="dqn${suffix}"
#   echo "[INFO] DQN label: $dqn_label"
  update_config "$latency" "$fees" "$dqn_label" "dqn"

    # cd "$RUN_DIR"
#   echo "[RUN] python $RUN_SCRIPT  (DQN)"
  python "$RUN_SCRIPT"


  # L1
  l1_label="l1${suffix}"
#   echo "[INFO] L1 label:  $l1_label"
  update_config "$latency" "$fees" "$l1_label" "symmetric_l1"

    # cd "$RUN_DIR"c
#   echo "[RUN] python $RUN_SCRIPT  (L1)"
  python "$RUN_SCRIPT"

  echo "[RUN] python $EVAL_SCRIPT  (L1)"
  python "$EVAL_SCRIPT" "$dqn_label" "$l1_label"

done

echo
echo "[DONE] All runs completed."
# echo "[INFO] Original JSON backup:       $JSON_BAK"
# echo "[INFO] Original run_config backup: $CONF_BAK"

# -------- Cleanup: remove all *.json.bak* in JSON dir and RUN_DIR --------
JSON_DIR="$(cd "$(dirname "$JSON_PATH")" && pwd)"
# echo "[CLEANUP] Deleting backup files (*.json.bak*) in:"
# echo "          - $JSON_DIR"
# echo "          - $RUN_DIR"
# Remove the explicit backups we created (in case find isn't available)
rm -f "$JSON_BAK" "$CONF_BAK" || true
# Remove any other *.json.bak* in those two locations (recursive, portable)
find "$JSON_DIR" "$RUN_DIR" -type f -name "*.json.bak*" -print -exec rm -f {} + || true
# echo "[CLEANUP] Done."

