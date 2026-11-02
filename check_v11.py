import numpy as np
for f in ["hustmotor_1024_v1.1.npz", "hustmotor_1024_rms.npz"]:
    d = np.load(f)
    print(f, d["vib"].shape, np.bincount(d["y"]), np.bincount(d["domain"]))
    print("  extra arrays:", [k for k in ["rms_v","rms_a","dc_v","dc_a","start"] if k in d.files])
    print("  window std range:", d["vib"].std(1).min().round(3), d["vib"].std(1).max().round(2))
a = np.load("hustmotor_1024_v1.1.npz"); b = np.load("hustmotor_1024.npz")
print("same as v1.0 data:", np.array_equal(a["vib"], b["vib"]) and np.array_equal(a["ac"], b["ac"]))