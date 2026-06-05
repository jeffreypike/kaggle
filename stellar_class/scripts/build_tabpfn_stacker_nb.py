"""Builds notebooks/stellar-tabpfn-stacker.ipynb — the TabPFN-3 stacker, runnable headless
on Kaggle GPU. Reproduces philippsinger/tabpfn-3-stacker (base OOFs -> logits + raw features
-> TabPFN-3) and adds our two *distinct* RealMLP bases (bs128, ls0.08), which the offline
screen showed contribute more to the stack than our redundant ref ensemble.

Bases come from ONE private dataset (jeffreypikeai/s6e6-stack-bases) of pre-normalized
(N,3)/(M,3) arrays — built locally with verified, alignment-checked loaders. This avoids
Kaggle kernel-mount filename ambiguity and authors changing their output formats (an earlier
kernel_sources version broke when TabM's test file format diverged from its OOF).
TabPFN-3 weights download from HuggingFace at runtime (enable_internet=True).
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
M = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
C = lambda s: cells.append(nbf.v4.new_code_cell(s))

M("""# 🌌 TabPFN-3 Stacker — diverse bases → logits + raw features → TabPFN-3
Reproduces the current S6E6 meta (Chris Deotte's stacker + philippsinger's TabPFN-3 version):
each base model's OOF/test probabilities → **logits**, append the **raw features**, and let
**TabPFN-3** be the meta-model. Adds two *distinct* RealMLP bases of our own (small-batch +
high-label-smoothing) that an offline screen showed are less redundant with the pool.

**Setup:** GPU (T4×2 ideal), Internet **On**; inputs = competition data + the
`s6e6-stack-bases` dataset.""")

M("## 1. Install TabPFN")
C('''import sys, subprocess, torch
# Pin Kaggle's pre-installed torch via a constraint so pip installs ALL of tabpfn's deps
# (tabpfn_common_utils, einops, …) WITHOUT swapping torch for a build that can't run on the GPU.
# NOTE: requires a T4 (sm_75) accelerator — Kaggle's torch 2.10 dropped Pascal/P100 (sm_60) support.
open("/tmp/torch_constraint.txt", "w").write(f"torch=={torch.__version__}\\n")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tabpfn", "-c", "/tmp/torch_constraint.txt"], check=True)
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| device:",
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
assert cap >= (7, 0), f"GPU compute {cap} unsupported by torch {torch.__version__} — set accelerator to GPU T4 x2 (P100 won't work)"
print("installed tabpfn")''')

M("## 2. Imports + device + TabPFN-3 weights")
C('''import os, glob, numpy as np, pandas as pd, torch
from sklearn.metrics import balanced_accuracy_score
# Use the TabPFN-3 weights from the added model input (avoids the gated HuggingFace download +
# license prompt, which has no interactive terminal in a batch kernel). Point the cache dir at
# the mounted model so .fit() finds the weights locally.
_w = (glob.glob("/kaggle/input/**/tabpfn-3/**/*.ckpt", recursive=True)
      or glob.glob("/kaggle/input/**/tabpfn*3*/**/*.ckpt", recursive=True)
      or glob.glob("/kaggle/input/**/*.ckpt", recursive=True))
if _w:
    os.environ["TABPFN_MODEL_CACHE_DIR"] = os.path.dirname(_w[0])
    print("TABPFN_MODEL_CACHE_DIR =", os.environ["TABPFN_MODEL_CACHE_DIR"])
else:
    print("WARNING: TabPFN-3 .ckpt not found under /kaggle/input — add the prior-labsai/tabpfn-3 model input")
ndev = torch.cuda.device_count()
DEVICE = [f"cuda:{i}" for i in range(ndev)] if ndev > 1 else ("cuda" if ndev == 1 else "cpu")
print("CUDA devices:", ndev, "->", DEVICE)
ROOT = "/kaggle/input"; OUT = "/kaggle/working"
EPS, CLIP = 1e-15, 30.0
TMAP = {"GALAXY": 0, "QSO": 1, "STAR": 2}; CLASSES = ["GALAXY", "QSO", "STAR"]
def logit(p):
    p = np.clip(p, EPS, 1 - EPS).astype(np.float64)
    return np.clip(np.log(p / (1 - p)), -CLIP, CLIP)''')

M("## 3. Load data + pre-normalized base predictions")
C('''comp = glob.glob(f"{ROOT}/**/train.csv", recursive=True)[0].rsplit("/", 1)[0]
train = pd.read_csv(f"{comp}/train.csv"); test = pd.read_csv(f"{comp}/test.csv")
y = train["class"].map(TMAP).to_numpy(); test_ids = test["id"]; N, M_ = len(train), len(test)
BASEDIR = glob.glob(f"{ROOT}/**/oof_xgb0.npy", recursive=True)[0].rsplit("/", 1)[0]
print("train", N, "test", M_, "| bases:", BASEDIR)

# 6 public diverse bases + our 2 distinct RealMLP bases (all clean (N,3)/(M,3) arrays)
BASES = ["xgb0", "xgb1", "realmlp0", "realmlp1", "tabm", "cat", "bs128", "ls008"]
oof_parts, test_parts = [], []
for nm in BASES:
    o = np.load(f"{BASEDIR}/oof_{nm}.npy").astype(np.float64)
    t = np.load(f"{BASEDIR}/test_{nm}.npy").astype(np.float64)
    ba = balanced_accuracy_score(y, o.argmax(1))
    assert o.shape == (N, 3) and t.shape == (M_, 3), f"{nm} shape {o.shape}/{t.shape}"
    assert ba > 0.90, f"{nm}: solo BA {ba:.3f} too low"
    print(f"  {nm:9s} BA={ba:.4f}")
    oof_parts.append(logit(o)); test_parts.append(logit(t))

feat = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
Xtr = np.hstack(oof_parts + [train[feat].to_numpy(np.float64)]).astype(np.float32)
Xte = np.hstack(test_parts + [test[feat].to_numpy(np.float64)]).astype(np.float32)
print(f"stack matrix: {len(BASES)} bases x3 logits + {len(feat)} raw = {Xtr.shape[1]} cols")''')

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
# headless `kaggle kernels push` runs via papermill, which requires a kernelspec.
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
nbf.write(nb, "notebooks/stellar-tabpfn-stacker.ipynb")
print(f"wrote notebooks/stellar-tabpfn-stacker.ipynb ({len(cells)} cells)")
