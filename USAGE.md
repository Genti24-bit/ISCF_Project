# How to use the HUSTmotor data (start here)

Repo: https://github.com/Genti24-bit/ISCF_Project
Paper: Zhao et al., *Multimodal unified generalization and translation network...*, EAAI 162 (2025) 112559

**Current data version: v1.1 data file, v1.2 code.** Default file for the whole team: `hustmotor_1024_v1.1.npz`.
It has the same signals as v1.0, plus the extra arrays `rms_v`, `rms_a`, `dc_v`, `dc_a`, `start` (section 9).
v1.2 adds the leak-free `norm="rms"` option to `load_dg_task` (section 10). Code-only change, no new data file.
`load_dg_task` works unchanged with every file listed in the README.

## 1. Setup (5 minutes)

```bash
git clone https://github.com/Genti24-bit/ISCF_Project.git
cd ISCF_Project
pip install -r requirements.txt          # numpy only; install PyTorch yourself for models
```

Download `hustmotor_1024_v1.1.npz` from **GitHub -> Releases -> v1.1** and put it in this folder. Then check:

```bash
sha256sum hustmotor_1024_v1.1.npz    # must be: b73efa9973af72b4a1b91fd5c54acc7035b082577df66b990ab0e6340624f986
python check_v11.py                  # optional: only works if you also have the v1.0 file
```
(Windows without Git Bash: `certutil -hashfile hustmotor_1024_v1.1.npz SHA256`)

If the hash differs, do not train. Re-download the file.

## 2. What the data is

A motor with **6 health states** recorded at **4 speeds**. Each sample is two signals of 1024 points cut at the same moment: one **vibration** and one **acoustic**.

| Label `y` | State | | Domain | Speed |
|---|---|---|---|---|
| 0 | Healthy | | 0 = B1 | 5 Hz |
| 1 | Bearing fault | | 1 = B2 | 10 Hz |
| 2 | Bowed rotor | | 2 = B3 | 20 Hz |
| 3 | Broken rotor bars | | 3 = B4 | 30 Hz |
| 4 | Misalignment | | | |
| 5 | Voltage unbalance | | | |

Total 24,000 samples = 6 states x 4 speeds x 1000. In the default file, signals are already **z-score normalised per sample**. Do not normalise again.

**The goal (domain generalisation):** train on 3 speeds, test on the 4th speed, which the model never sees.

## 3. Load a task (one line)

```python
from preprocess_hustmotor import load_dg_task

t = load_dg_task("hustmotor_1024_v1.1.npz", target_domain=0)

Xv_tr, Xa_tr, y_tr = t["train"]    # train : 3 seen speeds, 800/class/speed -> 14400 samples
Xv_va, Xa_va, y_va = t["val"]      # val   : same 3 seen speeds, other 200/class/speed -> 3600
Xv_te, Xa_te, y_te = t["target"]   # test  : the UNSEEN speed, 200/class -> 1200
d_tr = t["train_domains"]          # speed id (0-3) of every training sample
```

`Xv_*` = vibration `(N, 1024)`, `Xa_*` = acoustic `(N, 1024)`, `y_*` = labels `(N,)`.

| Task | Train on | Test on (unseen) | `target_domain` |
|---|---|---|---|
| 9  | 10, 20, 30 Hz | 5 Hz  | 0 |
| 10 | 5, 20, 30 Hz  | 10 Hz | 1 |
| 11 | 5, 10, 30 Hz  | 20 Hz | 2 |
| 12 | 5, 10, 20 Hz  | 30 Hz | 3 |

Run all four tasks and report the average, as the paper does.

## 4. Which arrays to feed which model

| Your model | Input | Paper name |
|---|---|---|
| **Vibration only** | `Xv_*` | M1 |
| **Acoustic only** | `Xa_*` | M2 |
| **Multimodal** (fusion, SimMMDG, MUGTN) | `Xv_*` **and** `Xa_*` together | M3-M10, Proposed |

Everything else (labels, split, task) is identical for all three. That is what makes results comparable.

```python
import numpy as np, torch

# 1D CNN wants (N, channels, length)
xv = torch.from_numpy(Xv_tr)[:, None, :]            # vibration only -> (N, 1, 1024)
xa = torch.from_numpy(Xa_tr)[:, None, :]            # acoustic only  -> (N, 1, 1024)
y  = torch.from_numpy(y_tr.astype(np.int64))        # IMPORTANT: labels are int8, loss needs int64

# Multimodal, two separate branches (as in the paper: one extractor per modality):
#   feat = concat([F_v(xv), F_a(xa)], dim=1)
# Multimodal, data-level fusion (paper M3): stack as 2 channels
x2 = torch.from_numpy(np.stack([Xv_tr, Xa_tr], axis=1))   # (N, 2, 1024)
```

**MUGTN / cross-domain pairing:** it needs same-class samples from different speeds. Use `y_tr` and `d_tr`:
```python
ids = np.where(y_tr == c)[0]          # all samples of class c
speeds = d_tr[ids]                    # which speed each one is from
```

**Missing sensor (Scenario 2):** train normally. Only at **test time**, replace one modality of the target set:
```python
Xv_missing = np.zeros_like(Xv_te)     # vibration sensor failed (zero padding baseline)
Xa_missing = np.zeros_like(Xa_te)     # acoustic sensor failed
```

## 5. Paper's training settings (use these for a fair comparison)

Adam, 50 epochs, batch size 256, learning rate `0.0005 / (1 + 10*p)**0.75` with `p = epoch / total_epochs`, same network architecture for all compared methods, **10 independent runs, report mean +/- std**. Metric: accuracy on the unseen speed.

## 6. Rules. Please read.

1. **Never touch the unseen speed (`t["target"]`) until the final evaluation.** No training, no early stopping, no choosing hyperparameters or the best epoch with it. Use `t["val"]` for that. (`val` comes from the seen speeds, so it is optimistic. That is fine. It is only for picking epochs.)
2. **Do not change the split, normalisation or channel yourself.** Everyone uses `hustmotor_1024_v1.1.npz` (data v1.1). Want to try something else (channel Y, min-max, RMS)? Tell the owner so it becomes a new version for everybody.
3. **Do not use the vibration Z channel.** It is not in this file on purpose (it behaves differently for the healthy motor, which would leak the label).
4. **Do not shuffle across `train` / `val` / `target`.** Shuffle inside the training set only.
5. **Save a results table per task** (mean, std over 10 runs, seeds used) and write "data v1.1" next to it.

## 7. Known data notes

- UNBAL at 5 Hz has a noisier acoustic signal than the other classes at 5 Hz. If that class is suspiciously easy on Task 9, this is probably why.
- The paper does not say which vibration axis it used. We use **X**.
- Samples within one recording overlap (stride about 159 of 1024 points). Train/val/test never share a recording block, so there is no leakage, but nearby training samples are very similar.
- Amplitude-preserving RMS normalisation is available through `load_dg_task(..., norm="rms")` (section 10). Do NOT use the old `hustmotor_1024_rms.npz`; its scale was computed from all four speeds. Use it only after the team agrees; the default stays z-score.

## 8. Quick sanity check for your model

A model that always guesses gets **16.7 %** (6 classes). If your unseen-speed accuracy is below ~30 %, something is wrong with the pipeline (labels not int64, wrong array, normalised twice) before it is a modelling problem. For reference, the paper reports (Scenario 1, motor): vibration-only baseline ~68 %, acoustic-only ~50 %, proposed MUGTN ~78 % average.

## 9. Extra arrays (since v1.1) (for error analysis and plotting)

Every v1.1+ file contains these arrays, one value per sample (24,000 in total):

| Array | Meaning |
|---|---|
| `rms_v`, `rms_a` | RMS of the raw vibration / acoustic window, after removing its DC offset (so the amplitude that z-score removes) |
| `dc_v`, `dc_a` | the DC offset (window mean) that was removed |
| `start` | row in the original raw `.txt` file where the window begins (window = rows `start` to `start+1023`) |

`load_dg_task` returns only signals and labels. To get the extra arrays for the **same samples in the same order** (for example the unseen test set), build the same mask from the file:

```python
import numpy as np
d = np.load("hustmotor_1024_v1.1.npz")
target = 0                                                   # Task 9
mask = (d["domain"] == target) & (d["split"] == 1)           # same rule as load_dg_task(..., eval_on="test")
rms_v, rms_a, start, y = d["rms_v"][mask], d["rms_a"][mask], d["start"][mask], d["y"][mask]
# row i here is the same sample as row i of t["target"]  (use eval_on="all" -> drop the split condition)
```

Typical use: after predicting, find test samples where one modality fails and compare their `rms_v` / `rms_a`
(is the sound weak? is the vibration quiet?). Use `start` to plot the original signal from the raw file.

## 10. Amplitude-preserving RMS normalisation (v1.2, leak-free)

Z-score gives every window RMS = 1, so loud and quiet windows look the same. `norm="rms"` keeps the relative amplitude:
each window has its DC offset removed, then ALL windows of a channel are divided by one RMS value.

```python
t = load_dg_task("hustmotor_1024_v1.1.npz", target_domain=0, norm="rms")
print(t["scale"])        # (vibration scale, acoustic scale) used for this task: report these
```

- The scale is computed **per task, from the source-train windows only**. The unseen speed and the source val windows never influence it.
- It works on any v1.1+ file (the loader rebuilds the DC-removed signal from `rms_v` / `rms_a`). No extra download.
- Expect the unseen target to be on a different scale than training (Task 12: about 4x louder; Task 9: about 0.08x). That is the real domain gap, not a bug.
- Default is still `norm="zscore"`. Write "data v1.1, code v1.2, norm=rms" next to results that use it.
