import numpy as np
from preprocess_hustmotor import load_dg_task

d = np.load("hustmotor_1024.npz")
print(d["vib"].shape, d["ac"].shape)
print("per class :", np.bincount(d["y"]))        # expect 4000 each
print("per domain:", np.bincount(d["domain"]))   # expect 6000 each
t = load_dg_task("hustmotor_1024.npz", target_domain=0)   # Task 9
print("train", t["train"][0].shape, "| unseen target", t["target"][0].shape)