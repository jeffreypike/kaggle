#!/usr/bin/env bash
# Set up + run the TabPFN-3 stacker on Kaggle GPU, headless from the terminal.
#
#   scripts/run_tabpfn_stacker.sh dataset   # create/refresh the s6e6-realmlp-distinct dataset
#   scripts/run_tabpfn_stacker.sh push      # push + run the notebook (after the dataset exists)
#   scripts/run_tabpfn_stacker.sh           # do both
#
# Uploads our two distinct RealMLP bases (bs128, ls0.08) as a PRIVATE dataset, then pushes the
# notebook wired to: competition data + the 6 public base kernels + that dataset. Auth from ../.env.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
set -a; . "$ROOT/../.env"; set +a
export KAGGLE_API_TOKEN="${KAGGLE_KEY:-${KAGGLE_API_TOKEN:-}}"
USER="${KAGGLE_USERNAME:?KAGGLE_USERNAME not set in ../.env}"
KG() { uv run --no-project --with kaggle kaggle "$@"; }

DS="$USER/s6e6-stack-bases"
STAGE_BASES="/tmp/stack_bases"

do_dataset() {
  # normalize all base predictions into /tmp/stack_bases, then create/version the dataset
  uv run --no-project python "$ROOT/scripts/build_stack_bases.py"
  if KG datasets files "$DS" >/dev/null 2>&1; then
    echo "dataset exists -> new version"; KG datasets version -p "$STAGE_BASES" -m "refresh stack bases" --dir-mode zip
  else
    echo "creating dataset $DS (private)"; KG datasets create -p "$STAGE_BASES" --dir-mode zip
  fi
}

do_push() {
  local d; d="$(mktemp -d)"; cp "$ROOT/notebooks/stellar-tabpfn-stacker.ipynb" "$d/"
  cat > "$d/kernel-metadata.json" <<JSON
{
  "id": "$USER/stellar-tabpfn-stacker",
  "title": "stellar-tabpfn-stacker",
  "code_file": "stellar-tabpfn-stacker.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "competition_sources": ["playground-series-s6e6"],
  "dataset_sources": ["$USER/s6e6-stack-bases"],
  "kernel_sources": [],
  "model_sources": ["prior-labsai/tabpfn-3/pyTorch/default/1"]
}
JSON
  echo "pushing $USER/stellar-tabpfn-stacker (GPU=${ACCELERATOR:-NvidiaTeslaT4})…"
  KG kernels push -p "$d" --accelerator "${ACCELERATOR:-NvidiaTeslaT4}"
  echo "  status : kaggle kernels status $USER/stellar-tabpfn-stacker"
  echo "  output : kaggle kernels output $USER/stellar-tabpfn-stacker -p predictions/"
  rm -rf "$d"
}

case "${1:-both}" in
  dataset) do_dataset;;
  push)    do_push;;
  both)    do_dataset; do_push;;
  *) echo "usage: $0 [dataset|push]"; exit 1;;
esac
