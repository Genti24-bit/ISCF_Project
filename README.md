# HUSTmotor preprocessing (MUGTN, EAAI 162 (2025) 112559)

```bash
pip install -r requirements.txt
python preprocess_hustmotor.py --raw_dir "Raw data" --out hustmotor_1024.npz
```
Output `hustmotor_1024.npz`: `vib`,`ac` (24000x1024 float32), `y` (0-5: H,BF,BOW,BROKEN,MISAL,UNBAL),
`domain` (0-3: 5/10/20/30 Hz = B1..B4), `block`, `win`, `split` (0=train, 1=test).

DG tasks 9-12 (leave-one-speed-out):
```python
from preprocess_hustmotor import load_dg_task
t = load_dg_task("hustmotor_1024.npz", target_domain=0)   # Task 9: train B2,B3,B4 -> test B1
```
Raw data: https://github.com/CHAOZHAO-1/HUSTmotor-multi-modal-dataset (please cite the paper).


## Data protocol v1.0 (do not change without bumping the version)
- Vibration channel: X | Normalization: per-window z-score | Split: block (blocks 0-3 train, block 4 test)
- Tasks 9-12: `load_dg_task("hustmotor_1024.npz", target_domain=0..3)`
- Never use the unseen target for training, early stopping or hyperparameter selection;
  use the source validation set returned by `load_dg_task`.
- Report mean ± std over 10 runs. Cite Zhao et al., EAAI 162 (2025) 112559.

SHA-256 of the released file `hustmotor_1024.npz`:
bd4e3c7273fa5efced76a39863e7a7f80255b8e382383519d613492cf671c658
(Regenerating the file gives identical arrays but a different file hash, because zip timestamps differ.
Compare arrays, not the hash, if you regenerate.)

## Data versions
| Version | File | Normalisation | Notes |
|---|---|---|---|
| v1.0 | hustmotor_1024.npz | per-window z-score | original, still valid |
| v1.1 | hustmotor_1024_v1.1.npz | per-window z-score | same signals as v1.0, plus rms_v, rms_a, dc_v, dc_a, start |
| v1.1-rms | hustmotor_1024_rms.npz | per-window DC removal + one channel-wide RMS | keeps relative amplitude |

Extra arrays: rms_v / rms_a = raw window RMS (DC removed), dc_v / dc_a = removed DC offset,
start = row in the raw .txt file where the window begins (use it to plot the original signal).
Always report which file you used next to your results.

SHA-256
hustmotor_1024_v1.1.npz  b73efa9973af72b4a1b91fd5c54acc7035b082577df66b990ab0e6340624f986
hustmotor_1024_rms.npz   e996775f92eac672acfaa856ba44c9b315c940357a7609b8afb90cbd16bc3683