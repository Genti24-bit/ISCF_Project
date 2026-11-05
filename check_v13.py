"""Quick check of hustmotor_1024_v1.3.npz. Run:  python check_v13.py"""
import numpy as np
from preprocess_hustmotor import load_dg_task

f = "hustmotor_1024_v1.3.npz"
d = np.load(f)
print("vib/ac:", d["vib"].shape, d["ac"].shape)
print("per class :", np.bincount(d["y"]), "(expect 4000 each)")
print("per domain:", np.bincount(d["domain"]), "(expect 6000 each)")
print("extra arrays:", [k for k in ["rms_v", "rms_a", "dc_v", "dc_a", "start", "feat_v", "feat_a"] if k in d.files])
print("feature shapes:", d["feat_v"].shape, d["feat_a"].shape, "| NaN/Inf:", int((~np.isfinite(d["feat_v"])).sum() + (~np.isfinite(d["feat_a"])).sum()))
print("window std (z-score, expect 1.0 1.0):", d["vib"].std(1).min().round(3), d["vib"].std(1).max().round(3))
print("feature names:", [str(n) for n in d["feat_names"]])
t = load_dg_task(f, 3, features=True)
print("Task 12 train/target:", t["train"][0].shape, t["target"][0].shape, "| train_feat:", t["train_feat"][0].shape, "| scale:", t["scale"])
t = load_dg_task(f, 3, norm="rms")
print("Task 12 norm=rms scale (expect ~0.01276 0.00753):", tuple(round(x, 5) for x in t["scale"]))
try:
    old = np.load("hustmotor_1024_v1.1.npz")
    print("signals identical to v1.1:", np.array_equal(d["vib"], old["vib"]) and np.array_equal(d["ac"], old["ac"]))
except FileNotFoundError:
    print("(hustmotor_1024_v1.1.npz not found next to this script - skipping the v1.1 comparison)")
