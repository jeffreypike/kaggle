"""Builds notebooks/stellar-tabpfn-stacker.ipynb — the TabPFN-3 stacker, runnable headless
on Kaggle GPU (T4x2). Reproduces philippsinger/tabpfn-3-stacker (base OOFs -> logits + raw
features -> TabPFN-3) and adds our two *distinct* RealMLP bases (bs128, ls0.08), which the
offline screen showed contribute more to the stack than our redundant ref ensemble.

Inputs are mounted by Kaggle from the kernel-metadata.json sources (see scripts/run_tabpfn_stacker.sh):
  competition_sources : playground-series-s6e6           -> train/test/sample_submission
  kernel_sources      : the 6 public base notebooks       -> their oof_/test_ outputs
  dataset_sources     : jeffreypike/s6e6-realmlp-distinct -> our oof/test_realmlp_{bs128,ls0.08}.npy
TabPFN-3 weights download from HuggingFace at runtime (enable_internet=True).
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
M = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
C = lambda s: cells.append(nbf.v4.new_code_cell(s))

M("""# 🌌 TabPFN-3 Stacker — diverse bases → logits + raw features → TabPFN-3
Reproduces the current S6E6 meta (after Chris Deotte's stacker + philippsinger's TabPFN-3
version): convert each base model's OOF/test probabilities to **logits**, append the **raw
features**, and let **TabPFN-3** be the meta-model. Adds two *distinct* RealMLP bases of our
own (small-batch + high-label-smoothing) that an offline screen showed are less redundant
with the pool than a standard RealMLP.

**Setup:** GPU **T4 × 2**, Internet **On**; inputs added via kernel-metadata.json.""")

M("## 1. Install TabPFN")
C('''import sys, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tabpfn"], check=True)
print("installed tabpfn")''')

M("## 2. Imports + device")
C('''import os, glob, numpy as np, pandas as pd, torch
from sklearn.metrics import balanced_accuracy_score
ndev = torch.cuda.device_count()
DEVICE = [f"cuda:{i}" for i in range(ndev)] if ndev > 1 else ("cuda" if ndev == 1 else "cpu")
print("CUDA devices:", ndev, "->", DEVICE)
ROOT = "/kaggle/input"; OUT = "/kaggle/working"
EPS, CLIP = 1e-15, 30.0
TMAP = {"GALAXY": 0, "QSO": 1, "STAR": 2}; CLASSES = ["GALAXY", "QSO", "STAR"]
def logit(p):
    p = np.clip(p, EPS, 1 - EPS).astype(np.float64)
    return np.clip(np.log(p / (1 - p)), -CLIP, CLIP)
def find(slug, fname):
    direct = f"{ROOT}/{slug}/{fname}"
    if os.path.exists(direct): return direct
    hits = glob.glob(f"{ROOT}/{slug}/**/{fname}", recursive=True) or glob.glob(f"{ROOT}/**/{fname}", recursive=True)
    if not hits: raise FileNotFoundError(f"{slug}/{fname} not found under {ROOT} — check kernel-metadata sources")
    return hits[0]''')

M("## 3. Load data + base predictions")
C('''comp = glob.glob(f"{ROOT}/**/train.csv", recursive=True)[0].rsplit("/", 1)[0]
train = pd.read_csv(f"{comp}/train.csv"); test = pd.read_csv(f"{comp}/test.csv")
ids = train["id"].to_numpy(); test_ids = test["id"]
y = train["class"].map(TMAP).to_numpy(); N, M_ = len(train), len(test)
print("train", N, "test", M_)

# (name, kaggle kernel/dataset slug, oof file, test file, loader kind)
BASES = [
    ("xgb0",     "xgb-v0-for-s6e6",          "oof_xgb_cv.csv",               "test_xgb_preds.csv",            "firstn"),
    ("xgb1",     "xgb-v1-for-s6e6",          "oof_preds.npy",                "test_preds.npy",               "npy"),
    ("realmlp0", "ps-s6-e6-realmlp-pytorch", "oof_preds.csv",                "test_preds.csv",               "id:GALAXY,QSO,STAR"),
    ("realmlp1", "realmlp-v1-for-s6e6",      "oof_preds.npy",                "test_preds.npy",               "npy"),
    ("tabm",     "s6e6-tabm",                "oof_preds.csv",                "test_preds.csv",               "flat"),
    ("cat",      "cat-v0-for-s6e6",          "catboost_oof_predictions.csv", "catboost_test_predictions.csv","id:prob_GALAXY,prob_QSO,prob_STAR"),
    # our distinct RealMLP bases (uploaded dataset); win the marginal-stack-contribution screen
    ("our_bs128","s6e6-realmlp-distinct",    "oof_realmlp_bs128.npy",        "test_realmlp_bs128.npy",       "npy"),
    ("our_ls008","s6e6-realmlp-distinct",    "oof_realmlp_ls0.08.npy",       "test_realmlp_ls0.08.npy",      "npy"),
]

def load(kind, path, index_ids, n):
    if kind == "npy":    return np.load(path).astype(np.float64)[:n]
    if kind == "firstn": return pd.read_csv(path).iloc[:n, -3:].to_numpy(np.float64)
    if kind == "flat":   return pd.read_csv(path).iloc[:, 0].to_numpy().reshape(-1, 3)[:n]
    if kind.startswith("id:"):
        cols = kind[3:].split(","); return pd.read_csv(path).set_index("id").loc[index_ids, cols].to_numpy(np.float64)
    raise ValueError(kind)

oof_parts, test_parts, names = [], [], []
for name, slug, oof_f, test_f, kind in BASES:
    o = load(kind, find(slug, oof_f), ids, N)
    t = load(kind, find(slug, test_f), test_ids, M_)
    ba = balanced_accuracy_score(y, o.argmax(1))
    assert ba > 0.90, f"{name}: solo BA {ba:.3f} too low — misaligned/class-order mismatch"
    print(f"  {name:10s} solo BA={ba:.4f}")
    oof_parts.append(logit(o)); test_parts.append(logit(t)); names.append(name)

feat = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
Xtr = np.hstack(oof_parts + [train[feat].to_numpy(np.float64)]).astype(np.float32)
Xte = np.hstack(test_parts + [test[feat].to_numpy(np.float64)]).astype(np.float32)
print(f"\\nstack matrix: {len(names)} bases x3 logits + {len(feat)} raw = {Xtr.shape[1]} cols")''')

M("## 4. TabPFN-3 meta (fit on all train, predict test)")
C('''from tabpfn import TabPFNClassifier

def fit_predict(subsample=None):
    idx = np.arange(N) if not subsample else np.random.RandomState(42).choice(N, subsample, replace=False)
    clf = TabPFNClassifier(device=DEVICE, n_estimators=2, balance_probabilities=True, ignore_pretraining_limits=True)
    clf.fit(Xtr[idx], y[idx])
    return clf.predict_proba(Xte)

try:
    test_prob = fit_predict()
except RuntimeError as e:                      # OOM on a single small GPU -> cap context
    print("retrying with subsampled context:", e)
    torch.cuda.empty_cache(); test_prob = fit_predict(subsample=200_000)
print("test_prob", test_prob.shape)''')

M("## 5. Submit")
C('''preds = np.asarray(CLASSES)[test_prob.argmax(1)]
pd.DataFrame({"id": test_ids, "class": preds}).to_csv(f"{OUT}/submission.csv", index=False)
np.save(f"{OUT}/test_stack_tabpfn.npy", test_prob)
print("wrote submission.csv + test_stack_tabpfn.npy")
print("pred class dist:", dict(zip(*np.unique(preds, return_counts=True))))''')

nb.cells = cells
# headless `kaggle kernels push` runs via papermill, which requires a kernelspec
# (the interactive UI fills this in automatically; nbformat does not).
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
nbf.write(nb, "notebooks/stellar-tabpfn-stacker.ipynb")
print(f"wrote notebooks/stellar-tabpfn-stacker.ipynb ({len(cells)} cells)")
