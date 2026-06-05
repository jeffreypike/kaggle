#!/usr/bin/env bash
# Run a notebook headless on Kaggle GPU/TPU from the terminal, then pull outputs.
#
#   scripts/kaggle_run.sh notebooks/stellar-realmlp-hpo-tpu.ipynb tpu
#   scripts/kaggle_run.sh notebooks/stellar-realmlp-tpu.ipynb tpu --wait
#   scripts/kaggle_run.sh <notebook.ipynb> <gpu|tpu|cpu> [--wait] [slug]
#
# Stages the notebook + a generated kernel-metadata.json, pushes it (save & run all),
# and prints the status/output commands. With --wait it polls until done and pulls
# outputs into predictions/. Auth comes from ../.env (KAGGLE_USERNAME / KAGGLE_KEY).
set -euo pipefail

NB="${1:?usage: kaggle_run.sh <notebook.ipynb> <gpu|tpu|cpu> [--wait] [slug]}"
ACC="${2:-cpu}"
WAIT=false; SLUG=""
for a in "${@:3}"; do [ "$a" = "--wait" ] && WAIT=true || SLUG="$a"; done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
set -a; . "$ROOT/../.env"; set +a
export KAGGLE_API_TOKEN="${KAGGLE_KEY:-${KAGGLE_API_TOKEN:-}}"
USER="${KAGGLE_USERNAME:?KAGGLE_USERNAME not set in ../.env}"
KG() { uv run --no-project --with kaggle kaggle "$@"; }   # kaggle CLI via uv (no global install needed)

[ -f "$NB" ] || { echo "no such notebook: $NB" >&2; exit 1; }
base="$(basename "$NB" .ipynb)"; SLUG="${SLUG:-$base}"
gpu=false; tpu=false
case "$ACC" in gpu) gpu=true;; tpu) tpu=true;; cpu) ;; *) echo "accel must be gpu|tpu|cpu" >&2; exit 1;; esac

STAGE="$(mktemp -d)"; trap 'rm -rf "$STAGE"' EXIT
cp "$NB" "$STAGE/"
# headless push runs via papermill, which needs a kernelspec (interactive UI adds it; nbformat doesn't)
uv run --no-project python - "$STAGE/$(basename "$NB")" <<'PY'
import sys, nbformat as nbf
p = sys.argv[1]; nb = nbf.read(p, as_version=4)
nb.metadata.setdefault("kernelspec", {"name": "python3", "display_name": "Python 3", "language": "python"})
nb.metadata.setdefault("language_info", {"name": "python"})
nbf.write(nb, p)
PY
cat > "$STAGE/kernel-metadata.json" <<JSON
{
  "id": "$USER/$SLUG",
  "title": "$SLUG",
  "code_file": "$(basename "$NB")",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": $gpu,
  "enable_tpu": $tpu,
  "enable_internet": true,
  "competition_sources": ["playground-series-s6e6"],
  "dataset_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
JSON

echo "pushing $USER/$SLUG  (accelerator=$ACC)…"
KG kernels push -p "$STAGE"
echo
echo "  status : scripts/kaggle_run.sh ... (or) kaggle kernels status $USER/$SLUG"
echo "  output : kaggle kernels output $USER/$SLUG -p predictions/"

if $WAIT; then
  echo "waiting for completion (polling every 60s)…"
  while true; do
    sleep 60
    st="$(KG kernels status "$USER/$SLUG" 2>/dev/null | grep -oE 'complete|error|running|queued|cancelAcknowledged' | head -1 || true)"
    echo "  $(date +%H:%M:%S)  $st"
    case "$st" in
      complete) KG kernels output "$USER/$SLUG" -p "$ROOT/predictions/"; echo "outputs pulled to predictions/"; break;;
      error|cancelAcknowledged) echo "run did not complete cleanly — check logs on Kaggle"; exit 1;;
    esac
  done
fi
