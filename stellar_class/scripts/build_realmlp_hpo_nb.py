"""Builds notebooks/stellar-realmlp-hpo-tpu.ipynb (run from project root).

8-SEED-ENSEMBLE HPO. Our earlier HPO was single-seed and didn't transfer (the ensemble
absorbs single-seed variance), so it was the wrong evaluation level. The TPU notebook trains
an 8-seed ensemble in ~10 min, so we can search at the level that actually matters: each
config is trained as a full 8-seed × 5-fold ensemble and judged on its ensemble OOF.

Reuses the validated install/imports/model/trainer/data cells from the main TPU notebook
verbatim (no drift), then loops over CONFIGS, saving per-config OOF/test
(`oof_realmlp_<name>.npy`, `test_realmlp_<name>.npy`) so each can be ranked offline both on
its own tuned OOF *and* on its marginal contribution to the TabPFN/LogReg stack.

Scope: MODEL hyperparameters only (architecture/epochs/dropout/lr/…); feature engineering is
fixed (a separate search would recompute Xtr/Xte inside the loop). Build the base notebook
first: `python scripts/build_realmlp_tpu_nb.py`.
"""
import nbformat as nbf

base = nbf.read("notebooks/stellar-realmlp-tpu.ipynb", as_version=4)
# cells 0..10 = title, install, imports, model, trainer, data — reuse verbatim (cell 10 is the data code cell)
cells = list(base.cells[:11])

cells[0] = nbf.v4.new_markdown_cell("""# 🔬 RealMLP-TD 8-seed-ensemble HPO (Flax NNX, TPU)
Search RealMLP hyperparameters **at the ensemble level**. Each config trains a full
**8-seed × 5-fold** ensemble (≈10 min on v5e-8) and is scored on its out-of-fold balanced
accuracy — the level that actually matters, since a seed-ensemble absorbs the single-seed
variance that makes single-seed HPO misleading.

Per-config OOF/test probabilities are saved (`oof_realmlp_<name>.npy`,
`test_realmlp_<name>.npy`) so configs can also be ranked by their contribution to a stacking
ensemble. Results save **as each config finishes**, so a session timeout keeps completed runs.

**Kaggle setup:** accelerator **TPU VM v5e-8**, Internet **On**, add the
`playground-series-s6e6` competition data.""")

M = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
C = lambda s: cells.append(nbf.v4.new_code_cell(s))

M("## 6. HPO: train an 8-seed ensemble per config")
C('''import time

# Single-axis overrides on the reference CONFIG (so each config isolates one effect).
# ~10 min each on v5e-8; the default ~10 keep the run under ~2 h. Uncomment more as budget allows.
# Heavy ones (more compute): epochs10, nens16.
CONFIGS = {
    "ref":          {},
    "wide":         {"hidden_dims": [1024, 1024, 1024]},
    "xwide":        {"hidden_dims": [768, 768, 768]},
    "deep":         {"hidden_dims": [512, 512, 512, 512]},
    "pyramid":      {"hidden_dims": [768, 512, 256]},
    "epochs8":      {"epochs": 8},
    "epochs4":      {"epochs": 4},          # faster, more underfit -> possibly distinct stack member
    "dropout0.10":  {"dropout": 0.10},
    "ls0.08":       {"ls_eps": 0.08},
    "bs128":        {"train_bs": 128},
    # --- optional extras (add compute) ---
    # "epochs10":   {"epochs": 10},
    # "dropout0.15":{"dropout": 0.15},
    # "lr0.02":     {"lr": 0.02},
    # "nens16":     {"n_ens": 16},
    # "embed12":    {"embed_dim": 12},
}


def train_ensemble(cfg):
    """Full 8-seed x 5-fold ensemble OOF + averaged test probs for one config."""
    bs = cfg["train_bs"]
    oof = np.zeros((len(Xtr), n_classes))
    test_probs = np.zeros((len(Xte), n_classes))
    for fold in range(5):
        tr = np.where(folds != fold)[0]; va = np.where(folds == fold)[0]
        enc = make_target_encoder(42)
        tr_te = enc.fit_transform(Xtr.iloc[tr][combos], y[tr]).astype(np.float32)
        va_te = enc.transform(Xtr.iloc[va][combos]).astype(np.float32)
        te_te = enc.transform(Xte[combos]).astype(np.float32)
        pre = NumericalPreprocessor(cfg["tfms"]).fit(np.hstack([Xtr.iloc[tr][num_cols].values.astype(np.float32), tr_te]))
        Xtr_n = pre.transform(np.hstack([Xtr.iloc[tr][num_cols].values.astype(np.float32), tr_te]))
        Xva_n = pre.transform(np.hstack([Xtr.iloc[va][num_cols].values.astype(np.float32), va_te]))
        Xte_n = pre.transform(np.hstack([Xte[num_cols].values.astype(np.float32), te_te]))
        clip = np.array(cat_dims) - 1
        Xtr_c = np.clip(Xtr.iloc[tr][cat_cols].values.astype(np.int64), 0, clip)
        Xva_c = np.clip(Xtr.iloc[va][cat_cols].values.astype(np.int64), 0, clip)
        Xte_c = np.clip(Xte[cat_cols].values.astype(np.int64), 0, clip)
        ytr, yva = y[tr], y[va]
        cw = jnp.asarray(compute_class_weight("balanced", classes=np.arange(n_classes), y=ytr), jnp.float32)

        n_train = (len(ytr) // bs) * bs
        n_batches = n_train // bs

        @nnx.vmap(in_axes=0)
        def create(seed_key):
            return RealMLP(n_classes, cat_dims, Xtr_n.shape[1], cfg, rngs=nnx.Rngs(params=seed_key))
        graphdef, params8 = nnx.split(create(jax.random.split(jax.random.key(100 + fold), N_SEEDS)))
        params0 = jax.tree.map(lambda x: x[0], params8)
        tx = make_tx(params0, cfg, n_batches * cfg["epochs"])
        train_fn = build_train_fn(graphdef, tx, cfg, n_train, n_batches, n_classes)

        pm = jax.pmap(train_fn, in_axes=(0, 0) + (None,) * 9)
        tkeys = jax.random.split(jax.random.key(200 + fold), N_SEEDS)
        oof8, test8, scores = pm(params8, tkeys,
            jnp.asarray(Xtr_n), jnp.asarray(Xtr_c), jnp.asarray(ytr),
            jnp.asarray(Xva_n), jnp.asarray(Xva_c), jnp.asarray(yva),
            jnp.asarray(Xte_n), jnp.asarray(Xte_c), cw)
        oof[va] = np.asarray(oof8).mean(0)
        test_probs += np.asarray(test8).mean(0) / 5
    return oof, test_probs


results = {}
for name, ov in CONFIGS.items():
    cfg = dict(CONFIG); cfg.update(ov)
    t0 = time.time()
    print(f"\\n========== {name}  {ov} ==========", flush=True)
    oof, test_probs = train_ensemble(cfg)
    raw = balanced_accuracy_score(y, oof.argmax(1))
    weights, tuned = tune_class_weights(y, oof)
    results[name] = tuned
    np.save(OUT / f"oof_realmlp_{name}.npy", oof)
    np.save(OUT / f"test_realmlp_{name}.npy", test_probs)
    print(f"{name}: raw={raw:.5f}  tuned={tuned:.5f}  ({time.time()-t0:.0f}s)", flush=True)
    print("  running leaderboard:", flush=True)
    for n in sorted(results, key=results.get, reverse=True):
        print(f"    {n:14s} {results[n]:.5f}{'  <- ref' if n == 'ref' else ''}", flush=True)

import pandas as pd
pd.DataFrame(sorted(results.items(), key=lambda kv: -kv[1]), columns=["config", "tuned_oof"]).to_csv(OUT / "hpo_results.csv", index=False)
print("\\nwrote hpo_results.csv + per-config oof_realmlp_<name>.npy / test_realmlp_<name>.npy")''')

M("""## Outputs
- **`hpo_results.csv`** — each config's 8-seed-ensemble tuned OOF balanced accuracy, ranked.
- **`oof_realmlp_<name>.npy`** / **`test_realmlp_<name>.npy`** — per-config OOF + test
  probabilities (train/test row order, columns sorted GALAXY/QSO/STAR). Rank these offline by
  tuned OOF *and* by marginal contribution to the stacker.""")

base.cells = cells
nbf.write(base, "notebooks/stellar-realmlp-hpo-tpu.ipynb")
print(f"wrote notebooks/stellar-realmlp-hpo-tpu.ipynb ({len(cells)} cells)")
