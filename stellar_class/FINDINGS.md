# Stellar Classification (Playground Series S6E6) — Findings & Experiment Log

Competition: predict stellar `class` ∈ {GALAXY, QSO, STAR} from photometric/spectroscopic
features. Metric: **balanced accuracy** (imbalanced classes). Synthetic data generated from
the [Stellar Classification Dataset – SDSS17](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17).

## Best result
- **LB 0.97024 (~4th, early)** — blend of RealMLP-8seed ensemble (0.65) + AutoGluon (0.35), class-weight tuned.
- RealMLP 8-seed ensemble **solo: LB 0.96998** (≈0.969 OOF).

## Methodology (what makes our numbers trustworthy)
- **Standardized 5-fold OOF** (`StratifiedKFold(5, shuffle, seed=42)`, `validation.load_data_with_folds`) shared by every model → OOFs are directly comparable and stackable.
- **CV↔LB is tight and trustworthy.** Tuned OOF tracks the public LB; LB has landed **~+0.0007–0.0009 above tuned CV** every time. Adversarial-validation AUC ≈ 0.50 (no train/test drift) predicted this.
- **Trust the full-577k OOF for ranking;** the public LB is a *subset* of test, so treat ±0.0002–0.0003 LB moves as noise. Don't optimize the public LB.
- **Class-weight tuning** (`validation.tune_class_weights`: per-class probability multipliers maximizing balanced accuracy) generalizes (~+0.009, held on LB).

## What worked ✅
| Lever | Effect |
|---|---|
| **RealMLP-TD** (Flax NNX port of `yekenot/ps-s6-e6-realmlp-pytorch`) | The top single model for this data |
| **Seed-ensembling** (8 seeds, parallel on TPU v5e-8 via `pmap`, ~12 min) | +0.0017 over single seed |
| **FE: 6 color pairs** (u-g, u-r, g-r, r-i, i-z, g-i) | +~0.0006 LB |
| Class-weight tuning | +~0.009 |

## What didn't (tested and ruled out) ❌
| Lever | Result |
|---|---|
| **HPO** (wide/deep/epochs/dropout/n_ens) | Helped single-seed (+0.0011) but **tied at the 8-seed ensemble** (−0.0001). Single-seed HPO doesn't transfer — the ensemble already handles variance. |
| More colors (10 pairs), redshift ratios, abs-mag/distance-modulus proxies | Flat or hurt (redshift's near-zero tail makes ratios noisy) |
| **External SDSS17 data** as extra training rows | Hurts (−0.0012) — distribution shift vs the synthetic test |
| FE for GBDTs | Flat — trees already recover colors/interactions |
| **Cross-family blending** (AutoGluon / TabPFN / LGB as blend members) | Saturated; TabPFN 3rd member +0.00007. As RealMLP improved, AG blend stopped helping (solo ≈ blend). |
| **Full AutoGluon vs memory-trimmed AG** in the blend | Wash. The local AG run had skipped some models (XGB/CatBoost) under memory pressure; re-ran the full Kaggle AG and saved OOF/test. Full AG solo is −0.00012 (0.96534 vs 0.96546) and the RealMLP blend ties (0.96931 vs 0.96930, +0.00001; blend down-weights AG 0.40→0.35). AG OOF labels differ only 0.40%, final blend submissions differ on 0.15% of test rows — pure noise. Confirms the banked 0.97024 was *not* a memory-constrained compromise. |
| **Pseudo-labeling** (confident test rows, leak-free CV) | No effect (−0.00002). 577k same-distribution labels + ~60% of test confidently labeled = redundant easy rows. |

## Single-model OOF (tuned balanced accuracy)
| Model | tuned OOF | LB |
|---|---|---|
| Manual LightGBM | 0.96437 | — |
| FLAML (lgbm) | 0.96488 | — |
| AutoGluon best_quality | 0.96546 | 0.966 |
| TabPFN-3 | 0.96305 | ~0.964 |
| RealMLP single-seed | 0.96761 | — |
| RealMLP 8-seed ensemble | 0.96908 | 0.96998 |
| **RealMLP-8seed + AutoGluon blend** | **0.96945** | **0.97024** |

## Reproduce
- **RealMLP (main model):** `notebooks/stellar-realmlp-tpu.ipynb` (Kaggle TPU v5e-8, 8 seeds in parallel) → `oof_realmlp_ens.npy`, `test_realmlp_ens.npy`, `submission.csv`. Local single-device: `src/train_realmlp.py`.
- **AutoGluon:** `src/train_autogluon.py --preset best_quality` (or reload the saved predictor; reloading a bagged predictor deadlocks on ray — init `ray.init(local_mode=True)` first).
- **Blend:** `python src/blend.py realmlp_ens autogluon_best_quality` → writes the blended submission from saved OOF + test probs.
- **Screens:** `src/hpo_realmlp.py` (config), and the FE/pseudo screens. All judge candidates on the standardized OOF.
- Other models: `src/train_lgb.py`, `src/train_automl.py` (FLAML), `src/train_tabpfn.py` (needs a CUDA GPU).

## Open / in progress
- **8-seed-ensemble HPO (redo of the flawed single-seed search).** Our earlier "HPO doesn't
  transfer" verdict came from single-seed runs — the *wrong* level, since a seed-ensemble absorbs
  single-seed variance (that's literally why the gains vanished). The TPU trains an 8-seed ensemble
  in ~10 min, so searching at the ensemble level is ~2 h, not prohibitive. `scripts/build_realmlp_hpo_nb.py`
  → `notebooks/stellar-realmlp-hpo-tpu.ipynb`: loops ~10 single-axis configs (width/depth/epochs/
  dropout/ls/bs), trains each as a full 8-seed×5-fold ensemble, saves per-config `oof_realmlp_<name>.npy`
  /`test_realmlp_<name>.npy` + `hpo_results.csv` (saves as each finishes → timeout-safe). Reuses the
  validated trainer/data cells verbatim. Model-HP scope only (FE fixed). Rank offline by tuned OOF
  *and* by marginal stack contribution (the metric that matters now — a config that's *more distinct*
  beats one that's marginally better solo, since our RealMLP is redundant with the pool's two RealMLPs).
  - **RESULT (2026-06-03, 10 configs, 8-seed ensembles on TPU):** SOLO is exhausted — `ref` [512]³ wins
    (0.96908), every variant within 0.0001 and ≤ ref; the 8-seed ensemble fully absorbs config differences.
    BUT by **marginal stack contribution** the order flips: `bs128` (worse solo, 0.96895) adds **+0.00012**
    to the public-base LogReg stack vs `ref` +0.00006 and our ens +0.00002 — small batch → more distinct →
    less redundant. Best pair **`bs128`+`ls0.08` = 0.96983** (public-only 0.96966; public+our-ens 0.96968).
    **Our contributed stack bases are `bs128`+`ls0.08`, not the ref ensemble.** Gain small (+0.00017 CV),
    matters only through the TabPFN meta, but it's non-redundant signal others lack.
- **TabPFN-3 stacking (the current S6E6 meta — re-engaged after dropping 4th→28th overnight).**
  Top solutions now use TabPFN-3 as a *stacker* (meta-model over diverse base OOFs), not as a
  blend member (which we'd ruled out). Read the actual notebook `philippsinger/tabpfn-3-stacker`
  (builds on `cdeotte/gpu-logistic-regression-stacker`): base OOFs → **logits** (`log(p/(1-p))`,
  ±30) → append **raw original features** → `TabPFNClassifier(n_estimators=2, balance_probabilities=True)`
  fit on all 577k, predict test. Bases: XGB×2, CatBoost, RealMLP×2 (yekenot + Deotte), **TabM**.
  - **Key finding: our bottleneck was base *diversity*, not the meta-model.** Their exact recipe
    (logits, +raw feats) on *our* OOFs goes nowhere (LogReg-logit 0.96926 ≤ our 0.96946 ceiling) —
    because our bases are correlated (AutoGluon's OOF is itself an XGB/Cat/LGB/NN stack, so
    AG+LGB+FLAML ≈ one GBDT signal). A stacker can only exploit diversity present in its inputs.
  - **Offline LogReg-logit on the 6 public diverse bases (alignment SHA/label-verified):**
    public-only **0.96966**, + our RealMLP 0.96968, + our RealMLP + AG + raw feats **0.96975** —
    vs our 0.96946 ceiling. Projects to ~0.9705–0.9706 LB (offset +0.0007–0.0009; but borrowed-base
    fold misalignment could add slight CV optimism — **LB is the arbiter**). Our RealMLP-8seed adds
    only +0.00002 (redundant with their two RealMLPs).
  - `src/stack_tabpfn.py`: fetches the 6 public base OOFs via Kaggle API, adds our RealMLP, builds
    [logits + raw feats], runs LogReg-logit ref (CPU) + TabPFN-3 meta (CUDA, `--subsample` for 10GB
    VRAM), writes submission + saves stacker OOF/test. `--no-tabpfn` already produced
    `submission_stack_logreg.csv` (0.96975 OOF) — submittable now without a GPU.
  - **LB RESULT (2026-06-03): LogReg stack = 0.96990 public LB** — *below* our banked 0.97024 blend.
    Offset was only +0.00015 (CV 0.96975), vs our own-model history of +0.0007–0.0009. Test alignment
    ruled out as the cause (all bases agree 0.978–0.995 on test argmax). The CV gain didn't translate —
    LogReg likely fit OOF-specific quirks in the borrowed base preds (CV up, LB flat). **Lesson: don't
    project the +0.0008 offset onto borrowed-base stacks; their offset is ~+0.0002.**
  - **Leaderboard target (2026-06-03): public LB top = 0.97076 (cluster of 6+ teams), Deotte 0.97070.**
    That identical-score cluster = everyone running the public TabPFN-3 stacker. So the **meta-model is
    the lever**: same bases, LogReg 0.96990 vs TabPFN ~0.97076 ≈ +0.0008 from the meta alone.
  - NEXT: (1) run the **TabPFN-3 meta** (`src/stack_tabpfn.py`, 3080) — expect ~0.9707, +0.0005 over our
    0.97024, joins the top cluster. (2) To *beat* the cluster (TabPFN-on-public-bases is what everyone
    has), we need a base they lack → the ensemble-HPO search for a *distinct* RealMLP. TabM already in pool.
- **Heterogeneous ensemble** (config diversity instead of 8× same-config seeds): does varying
  width/depth/dropout across members decorrelate errors *beyond* what seeds already do?
  Screen: `src/diversity_screen.py` (full scale, GPU/TPU) measures config-vs-config OOF
  disagreement against the seed-vs-seed baseline and compares a heterogeneous K-member blend
  to a homogeneous one. Build the 8-distinct-member notebook only if configs decorrelate
  clearly more *and* the heterogeneous blend wins.
  - NOTE: `predictions/oof_realmlp_wide.npy` / `test_realmlp_wide.npy` are **byte-identical
    copies of `realmlp_ens`** (verified by SHA), not a real wide-config ensemble — earlier
    notes calling them a "diverse member candidate" were wrong. No saved second-config
    ensemble exists; the screen has to train distinct configs from scratch.

## Status / next
Search space is **mapped and mostly exhausted**; the model is well-optimized at ~0.970.
Suggested 2 final submissions: the **0.97024 blend** + the **0.96998 RealMLP-solo** (most
diverse strong pair). Re-engage when there's new signal (rising LB to defend, or a new
strong public notebook — the discussion board is the best source; it's how RealMLP and
TabPFN were found). Remaining ideas are marginal (more seeds; blending diverse RealMLP
*config* variants).

## Environment gotchas
- `setuptools >= 81` removed `pkg_resources` → AutoGluon silently drops XGBoost/CatBoost. Pin `setuptools < 81`.
- Kaggle sklearn ≥ 1.9 deprecates `TargetEncoder(shuffle=, random_state=)` → pass a CV splitter (`make_target_encoder` handles both).
- 3080 box: needs `polars-lts-cpu` (no AVX2) and Python < 3.12 (f-string backslash); TabPFN test prediction must be batched (cuBLAS limit at 247k rows).
- ray has no Python 3.13 wheel → AutoGluon+ray runs in a separate 3.12 env (`.venv-ag`).
