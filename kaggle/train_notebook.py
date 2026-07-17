# %% [markdown]
# # GAVE2 Task 1 training on Kaggle
#
# One-time setup (do this before running any cells):
# 1. Notebook settings (right panel) -> Accelerator: GPU T4 x2 (or P100).
#    Internet: On (needed for git clone + Hugging Face weights download).
# 2. Add-ons -> Secrets -> add a secret named `GH_TOKEN` containing a GitHub
#    Personal Access Token (classic, `repo` scope) for aaronjji/gave2-challenge
#    (it's a private repo). Attach the secret to this notebook.
# 3. Upload the GAVE2_preliminary dataset as a **PRIVATE** Kaggle Dataset
#    (Create -> New Dataset -> upload the extracted data/raw/GAVE2_preliminary
#    folder -- keep it Private, the competition prohibits public re-hosting).
#    Add it to this notebook via "Add Input".
# 4. To resume across sessions: after a run, "Save Version" (commit) the
#    notebook -- its /kaggle/working output becomes a versioned output you
#    can "Add Input" in the next session. Copy the checkpoint back into
#    runs/task1/foldN/latest.pth and re-run with --resume.

# %% [markdown]
# ## Cell 1: clone the repo

# %%
import os
from kaggle_secrets import UserSecretsClient

token = UserSecretsClient().get_secret("GH_TOKEN")
repo_url = f"https://{token}@github.com/aaronjji/gave2-challenge.git"

if not os.path.exists("/kaggle/working/gave2-challenge"):
    os.system(f"git clone --recurse-submodules {repo_url} /kaggle/working/gave2-challenge")
else:
    os.system("cd /kaggle/working/gave2-challenge && git pull && git submodule update --init --recursive")

os.chdir("/kaggle/working/gave2-challenge")
print(os.getcwd())

# %% [markdown]
# ## Cell 2: install dependencies
# (torch/torchvision are preinstalled on Kaggle -- skip those.)

# %%
os.system("pip install -q albumentations opencv-python-headless scikit-image scipy networkx sknw huggingface_hub safetensors")

# %% [markdown]
# ## Cell 3: point at the uploaded dataset
# Replace `YOUR-DATASET-SLUG` with whatever slug Kaggle assigned your
# uploaded dataset (visible in the "Add Input" panel / the input path).

# %%
# NOTE: private Kaggle datasets mount one level deeper than public ones --
# /kaggle/input/datasets/<username>/<slug>/... not /kaggle/input/<slug>/...
# (confirmed empirically 2026-07-17). This tries both layouts.
candidates = [
    "/kaggle/input/datasets/aaronajit/gave2-preliminary/GAVE2_preliminary",
    "/kaggle/input/gave2-preliminary/GAVE2_preliminary",
    "/kaggle/input/gave2-preliminary",
]
os.system("find /kaggle/input -maxdepth 4")

DATA_ROOT = None
for c in candidates:
    if os.path.exists(f"{c}/training/images"):
        DATA_ROOT = c
        break
assert DATA_ROOT is not None, f"Dataset not found in any of {candidates} -- check the find output above and set DATA_ROOT manually"
print("DATA_ROOT =", DATA_ROOT)

# %% [markdown]
# ## Cell 4: (optional) resume from a previous session's checkpoint
# If you added a previous notebook version's output as an input, copy its
# checkpoint into place before training so --resume picks it up.

# %%
import shutil
from pathlib import Path

PREV_CHECKPOINT = None  # e.g. "/kaggle/input/your-notebook-output-vN/runs/task1/fold0/latest.pth"
if PREV_CHECKPOINT and os.path.exists(PREV_CHECKPOINT):
    dst = Path("runs/task1/fold0/latest.pth")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(PREV_CHECKPOINT, dst)
    print(f"Restored checkpoint from {PREV_CHECKPOINT}")

# %% [markdown]
# ## Cell 5: train one fold
# Kaggle sessions cap out around 9-12h; --max-seconds leaves a safety
# margin so the run checkpoints and exits cleanly rather than getting killed
# mid-step. Re-run this same cell (with --resume) if you need more epochs
# than one session allows -- it picks up from runs/task1/fold{N}/latest.pth.

# %%
FOLD = 0
os.system(
    f"python -u src/train_task1.py "
    f"--fold {FOLD} --data-root {DATA_ROOT} "
    f"--epochs 60 --steps-per-epoch 50 --num-workers 2 "
    f"--checkpoint-every-epochs 2 --max-seconds 30000 "
    f"--out-dir runs/task1 --resume"
)

# %% [markdown]
# ## Cell 6: repeat for the other folds
# Change FOLD to 1, 2, 3, 4 and re-run Cell 5 (each fold trains
# independently). Once satisfied, "Save Version" to persist runs/ as
# this notebook's output, then download the checkpoints (or add this
# version as an input to a follow-up inference notebook/session).
