# HUSTmotor preprocessing (MUGTN, EAAI 162 (2025) 112559)

**New here? Read [USAGE.md](USAGE.md) first.** Current data file: `hustmotor_1024_v1.3.npz` (GitHub -> Releases -> v1.3).

```bash
pip install -r requirements.txt
python preprocess_hustmotor.py --raw_dir "Raw data" --out hustmotor_1024_v1.3.npz
```
Output arrays: `vib`, `ac` (24000x1024 float32), `y` (0-5: H,BF,BOW,BROKEN,MISAL,UNBAL),
`domain` (0-3: 5/10/20/30 Hz = B1..B4), `block`, `win`, `split` (0=train, 1=test),
plus `rms_v`, `rms_a`, `dc_v`, `dc_a`, `start`, and 11 time-domain features `feat_v`, `feat_a` (v1.3).

DG tasks 9-12 (leave-one-speed-out):
```python
from preprocess_hustmotor import load_dg_task
t = load_dg_task("hustmotor_1024_v1.3.npz", target_domain=0)                  # Task 9: train B2,B3,B4 -> test B1
t = load_dg_task("hustmotor_1024_v1.3.npz", target_domain=0, norm="rms")      # amplitude-preserving, leak-free
t = load_dg_task("hustmotor_1024_v1.3.npz", target_domain=0, features=True)   # + 11 time-domain features
```
Raw data: https://github.com/CHAOZHAO-1/HUSTmotor-multi-modal-dataset (please cite the paper).

## Data protocol (do not change without bumping the version)
- Vibration channel: X | Normalization: per-window z-score | Split: block (blocks 0-3 train, block 4 test)
- Tasks 9-12: `load_dg_task(path, target_domain=0..3)`
- Never use the unseen target for training, early stopping or hyperparameter selection;
  use the source validation set returned by `load_dg_task`.
- Report mean ± std over 10 runs. Cite Zhao et al., EAAI 162 (2025) 112559.
- Write the data version next to every result (for example "data v1.3").

## Data versions
| Version | File | Notes |
|---|---|---|
| v1.0 | hustmotor_1024.npz | per-window z-score, original, still valid |
| v1.1 | hustmotor_1024_v1.1.npz | same signals as v1.0, plus rms_v, rms_a, dc_v, dc_a, start |
| v1.2 | (code only, same data file as v1.1) | `load_dg_task(..., norm="rms")`: leak-free RMS scale per task; replaces hustmotor_1024_rms.npz |
| v1.3 | hustmotor_1024_v1.3.npz | same signals as v1.1, plus 11 time-domain features `feat_v`, `feat_a` (`load_dg_task(..., features=True)`). **Current default.** |

Extra arrays: rms_v / rms_a = raw window RMS (DC removed), dc_v / dc_a = removed DC offset,
start = row in the raw .txt file where the window begins (use it to plot the original signal).
Features (Tran et al., Mathematics 12 (2024) 1569, Table 2): rms, var, peak, crest, kurtosis, clearance, shape,
line_integral, peak2peak, skewness, impulse.

SHA-256
```
hustmotor_1024.npz       bd4e3c7273fa5efced76a39863e7a7f80255b8e382383519d613492cf671c658
hustmotor_1024_v1.1.npz  b73efa9973af72b4a1b91fd5c54acc7035b082577df66b990ab0e6340624f986
hustmotor_1024_v1.3.npz  605a041d35aab9a12749d9f5f45580cca87e0b5b63b8330bcafdb215da3a8e7d
```
(Regenerating a file gives identical arrays but a different file hash, because zip timestamps differ.
Compare arrays, not the hash, if you regenerate.)
