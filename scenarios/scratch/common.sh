# Fungsi bersama skrip skenario SCRATCH (dipakai lewat `source`).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
SCRATCH_ROOT=scenarios/scratch

# Satu GPU tidak cukup untuk dua training: jangan sampai mengganggu training yang sedang berjalan.
refuse_if_training_running() {
  if pgrep -af "src\.training\.train_" >&2; then
    echo "[scratch] Ada training lain yang sedang berjalan (lihat di atas). Tunggu sampai selesai, lalu jalankan lagi." >&2
    exit 1
  fi
}

# run_experiment <modul> <task>/<config.yaml> <ID>
run_experiment() {
  local module=$1 config=$SCRATCH_ROOT/configs/$2 run_dir=$SCRATCH_ROOT/outputs/${2%%/*}/$3
  local resume=()
  if [[ -f $run_dir/training_summary.json ]]; then
    echo "[scratch] $3 sudah selesai, dilewati."
    return 0
  fi
  [[ -f $run_dir/weights/last.pt ]] && resume=(--resume)
  python -m "$module" --config "$config" ${resume[@]+"${resume[@]}"} ${SET[@]+--set "${SET[@]}"}
}
