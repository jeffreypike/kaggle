"""Does *config* diversity decorrelate RealMLP errors more than *seeds* already do?

Our ensemble is 8× the same config with different seeds. Seed-ensembling works because
seeds decorrelate errors (+0.0017). Heterogeneous configs (one wider, one deeper, …) only
help if they decorrelate errors *beyond* what seeds give. This screen measures exactly that
at FULL scale, before committing to rebuilding the 8-member TPU notebook.

It trains, single-seed 5-fold OOF each:
  • the reference config across several SEEDS  → seed-vs-seed disagreement (the baseline)
  • several distinct CONFIGS at a fixed seed   → config-vs-config disagreement
then reports the two disagreement levels and whether a heterogeneous blend of K configs
beats a homogeneous blend of K seeds (same K). If config-disagreement ≲ seed-disagreement,
there is no extra variety to harvest — stop. Otherwise build the heterogeneous ensemble.

Run on a GPU/TPU box (full scale, ranking doesn't transfer from a subsample):
    python stellar_class/src/diversity_screen.py
Tiny CPU logic check:
    python stellar_class/src/diversity_screen.py --quick
"""
import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import train_realmlp as T
from validation import load_data_with_folds, tune_class_weights, DATA_DIR, PREDICTIONS_DIR
from sklearn.metrics import balanced_accuracy_score

# Distinct CONFIGS (overrides on the reference), one fixed seed — tests architectural +
# soft diversity. Each is a single-axis change so the source of any decorrelation is clear.
CONFIGS = {
    "ref":        {},
    "wide":       {"hidden_dims": [1024, 1024, 1024]},
    "deep":       {"hidden_dims": [512, 512, 512, 512]},
    "narrow":     {"hidden_dims": [256, 256, 256]},
    "dropout0.1": {"dropout": 0.10},
    "ls0.08":     {"ls_eps": 0.08},
}
# SEEDS for the reference config — establishes the seed-vs-seed decorrelation baseline.
SEEDS = [42, 1, 2, 3]


def run(overrides, seed, Xtr, y, folds, combos, cat_cols, num_cols, cat_dims, n_classes):
    """Full 5-fold single-seed OOF probabilities for one config+seed."""
    cfg = dict(T.CONFIG); cfg.update(overrides); cfg["seed"] = seed
    oof = np.zeros((len(Xtr), n_classes))
    for fold in range(5):
        tr = np.where(folds != fold)[0]; va = np.where(folds == fold)[0]
        enc = T.make_target_encoder(seed)
        tr_te = enc.fit_transform(Xtr.iloc[tr][combos], y[tr]).astype(np.float32)
        va_te = enc.transform(Xtr.iloc[va][combos]).astype(np.float32)
        pre = T.NumericalPreprocessor(cfg["tfms"]).fit(np.hstack([Xtr.iloc[tr][num_cols].values.astype(np.float32), tr_te]))
        Xtr_n = pre.transform(np.hstack([Xtr.iloc[tr][num_cols].values.astype(np.float32), tr_te]))
        Xva_n = pre.transform(np.hstack([Xtr.iloc[va][num_cols].values.astype(np.float32), va_te]))
        clip = np.array(cat_dims) - 1
        Xtr_c = np.clip(Xtr.iloc[tr][cat_cols].values.astype(np.int64), 0, clip)
        Xva_c = np.clip(Xtr.iloc[va][cat_cols].values.astype(np.int64), 0, clip)
        val_probs, _, _ = T.fit_fold(Xtr_n, Xtr_c, y[tr], Xva_n, Xva_c, y[va], cat_dims, n_classes, cfg)
        oof[va] = val_probs
    return oof


def disagreement(a, b):
    return (a.argmax(1) != b.argmax(1)).mean()


def best_blend(oofs, y, step=0.1):
    """Best equal-or-searched simplex blend tuned BA over a list of member OOFs."""
    n = len(oofs)
    k = round(1 / step)
    best = (-1.0, None)
    for combo in itertools.product(range(k + 1), repeat=n - 1):
        if sum(combo) <= k:
            w = np.array([c / k for c in combo] + [1 - sum(c / k for c in combo)])
            blend = sum(wi * oi for wi, oi in zip(w, oofs))
            t = tune_class_weights(y, blend)[1]
            if t > best[0]:
                best = (t, w)
    return best


def main(quick=False):
    train_df = load_data_with_folds().to_pandas()
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    classes = sorted(train_df["class"].unique()); cmap = {c: i for i, c in enumerate(classes)}
    y = train_df["class"].map(cmap).to_numpy(); folds = train_df["fold"].to_numpy()
    raw_cat = ["spectral_type", "galaxy_population"]; raw_num = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
    st = {}
    Xtr, new_cat, new_num, combos = T.feature_engineering(train_df[raw_cat + raw_num], raw_cat[:], raw_num[:], st, fit=True)
    Xte, *_ = T.feature_engineering(test_df[raw_cat + raw_num], raw_cat[:], raw_num[:], st, fit=False)
    cat_cols = sorted(raw_cat + new_cat); num_cols = raw_num + new_num
    Xtr = Xtr.reindex(sorted(Xtr.columns), axis=1); Xte = Xte.reindex(sorted(Xte.columns), axis=1)
    cat_dims = [int(max(Xtr[c].max(), Xte[c].max()) + 1) for c in cat_cols]
    n_classes = len(classes)

    configs, seeds = CONFIGS, SEEDS
    if quick:                                   # local CPU logic check only
        keep = np.random.RandomState(0).choice(len(Xtr), 8000, replace=False)
        Xtr, y, folds = Xtr.iloc[keep].reset_index(drop=True), y[keep], folds[keep]
        tiny = {"n_ens": 2, "hidden_dims": [64], "epochs": 1, "train_bs": 2048}
        configs = {"ref": tiny, "wide": {**tiny, "hidden_dims": [128]}, "deep": {**tiny, "hidden_dims": [64, 64]}}
        seeds = [42, 1, 2]

    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    print(f"=== RealMLP diversity screen ({'quick' if quick else 'full'}) ===", flush=True)

    # 1) reference config across seeds (seed decorrelation baseline)
    seed_oofs = {}
    for s in seeds:
        t0 = time.time()
        oof = run(configs["ref"], s, Xtr, y, folds, combos, cat_cols, num_cols, cat_dims, n_classes)
        seed_oofs[s] = oof
        np.save(PREDICTIONS_DIR / f"oof_div_seed{s}.npy", oof)
        print(f"  ref seed={s:<3d} tuned={tune_class_weights(y, oof)[1]:.5f}  ({time.time()-t0:.0f}s)", flush=True)

    # 2) distinct configs at the reference seed (config decorrelation)
    cfg_oofs = {}
    for name, ov in configs.items():
        if name == "ref":
            cfg_oofs[name] = seed_oofs[seeds[0]]; continue
        t0 = time.time()
        oof = run(ov, seeds[0], Xtr, y, folds, combos, cat_cols, num_cols, cat_dims, n_classes)
        cfg_oofs[name] = oof
        np.save(PREDICTIONS_DIR / f"oof_div_{name}.npy", oof)
        print(f"  config {name:11s} tuned={tune_class_weights(y, oof)[1]:.5f}  ({time.time()-t0:.0f}s)  {ov}", flush=True)

    # 3) decorrelation: mean pairwise OOF label-disagreement
    seed_pairs = [disagreement(seed_oofs[a], seed_oofs[b]) for a, b in itertools.combinations(seeds, 2)]
    cfg_names = list(cfg_oofs)
    cfg_pairs = [disagreement(cfg_oofs[a], cfg_oofs[b]) for a, b in itertools.combinations(cfg_names, 2)]
    print("\n--- decorrelation (mean pairwise OOF label-disagreement) ---")
    print(f"  seed vs seed   (ref config) : {np.mean(seed_pairs)*100:.3f}%  (n={len(seed_pairs)} pairs)")
    print(f"  config vs config (fixed seed): {np.mean(cfg_pairs)*100:.3f}%  (n={len(cfg_pairs)} pairs)")
    extra = np.mean(cfg_pairs) - np.mean(seed_pairs)
    print(f"  → configs decorrelate {extra*100:+.3f}% {'MORE' if extra > 0 else 'LESS'} than seeds")

    # 4) does a heterogeneous blend of K configs beat a homogeneous blend of K seeds?
    K = min(len(seeds), len(cfg_names))
    het_t, het_w = best_blend([cfg_oofs[n] for n in cfg_names[:K]], y)
    hom_t, hom_w = best_blend([seed_oofs[s] for s in seeds[:K]], y)
    print(f"\n--- {K}-member blend, tuned BA ---")
    print(f"  homogeneous ({K} ref seeds)   : {hom_t:.5f}  weights={hom_w.round(2)}")
    print(f"  heterogeneous ({K} configs)   : {het_t:.5f}  weights={het_w.round(2)}  members={cfg_names[:K]}")
    print(f"  → heterogeneous {het_t-hom_t:+.5f} vs homogeneous")
    print("\nVERDICT: build the 8-distinct-member ensemble only if configs decorrelate clearly "
          "more AND the heterogeneous blend wins. Otherwise the seed-ensemble already captures it.")
    print("DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny CPU logic check")
    args = ap.parse_args()
    main(quick=args.quick)
