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

DS="$USER/s6e6-realmlp-distinct"
FILES=(oof_realmlp_bs128.npy test_realmlp_bs128.npy "oof_realmlp_ls0.08.npy" "test_realmlp_ls0.08.npy")

do_dataset() {
  local d; d="$(mktemp -d)"
  for f in "${FILES[@]}"; do cp "$ROOT/predictions/$f" "$d/"; done
  cat > "$d/dataset-metadata.json" <<JSON
{ "title": "s6e6-realmlp-distinct", "id": "$DS", "licenses": [{ "name": "CC0-1.0" }] }
JSON
  if KG datasets files "$DS" >/dev/null 2>&1; then
    echo "dataset exists -> new version"; KG datasets version -p "$d" -m "refresh distinct RealMLP bases" --dir-mode zip
  else
    echo "creating dataset $DS (private)"; KG datasets create -p "$d" --dir-mode zip
  fi
  rm -rf "$d"
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
  "dataset_sources": ["$DS"],
  "kernel_sources": [
    "cdeotte/xgb-v0-for-s6e6", "cdeotte/xgb-v1-for-s6e6",
    "yekenot/ps-s6-e6-realmlp-pytorch", "cdeotte/realmlp-v1-for-s6e6",
    "donmarch14/s6e6-tabm", "cdeotte/cat-v0-for-s6e6"
  ],
  "model_sources": []
}
JSON
  echo "pushing $USER/stellar-tabpfn-stacker (GPU)…"
  KG kernels push -p "$d"
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
