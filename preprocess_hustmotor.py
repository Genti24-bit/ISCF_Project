#!/usr/bin/env python3
"""
Preprocess the HUSTmotor multimodal dataset (vibration + acoustic) the way
Zhao et al., EAAI 162 (2025) 112559 (MUGTN) describe it:

  * 6 health states x 4 speeds (5/10/20/30 Hz = domains B1..B4)
  * 1000 samples per (state, domain); each sample = 1 vibration window (1024 pts)
    + 1 acoustic window (1024 pts), both cut at the SAME time indices (synchronous)
  * 800 train / 200 test per (state, domain)

Facts about the raw files that this script handles (verified on the data):
  * 18-line text header, then 5 columns: time, X, Y, Z (vibration), Sound
  * 163,840 rows = 5 blocks x 32,768 rows. The time column RESTARTS every block,
    so the file is 5 separate 1.28 s acquisitions glued together. Windows must
    never straddle a block boundary -> windows are cut inside each block.
  * 32768/1024 = 32 -> only 160 non-overlapping windows per file, but the paper
    uses 1000 -> 200 evenly spaced (overlapping, stride ~159.5) windows per block.

Usage:
    python preprocess_hustmotor.py --raw_dir "Raw data" --out hustmotor_1024.npz
"""
import argparse
import re
from pathlib import Path

import numpy as np

CLASSES = ["H", "BF", "BOW", "BROKEN", "MISAL", "UNBAL"]   # label 0..5
SPEEDS = {5: 0, 10: 1, 20: 2, 30: 3}                        # Hz -> domain B1..B4 (0..3)
FS = 25_600
N_ROWS, BLOCK, N_BLOCKS = 163_840, 32_768, 5
CHAN = {"X": 1, "Y": 2, "Z": 3}                             # column index in raw file
SOUND_COL = 4


# --------------------------------------------------------------------------- I/O
def read_raw(path: Path) -> np.ndarray:
    """Return array (163840, 5): time, X, Y, Z, sound. Validates the file."""
    with open(path, "r", errors="ignore") as fh:
        head = [next(fh) for _ in range(40)]
    skip = next(i for i, l in enumerate(head) if l.startswith("Time (seconds)")) + 1
    a = np.loadtxt(path, skiprows=skip, dtype=np.float64)
    if a.shape != (N_ROWS, 5):
        raise ValueError(f"{path.name}: expected {(N_ROWS, 5)}, got {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError(f"{path.name}: contains NaN/Inf")
    return a


# ------------------------------------------------------------------ normalisation
def normalise(w: np.ndarray, mode: str) -> np.ndarray:
    """w: (n_windows, 1024). Statistics are computed PER WINDOW (no train/test leakage)."""
    w = w.astype(np.float32)
    if mode == "none":
        return w
    if mode == "zscore":                                   # remove DC offset + amplitude scale
        mu = w.mean(1, keepdims=True)
        sd = w.std(1, keepdims=True)
        return (w - mu) / np.maximum(sd, 1e-8)
    if mode == "minmax":                                   # -> [0, 1] like Fig. 6 of the paper
        lo = w.min(1, keepdims=True)
        hi = w.max(1, keepdims=True)
        return (w - lo) / np.maximum(hi - lo, 1e-8)
    if mode == "center":                                   # remove DC offset only, keep amplitude
        return w - w.mean(1, keepdims=True)
    raise ValueError(mode)



# ---------------------------------------------------------------------- windowing
def block_windows(sig: np.ndarray, win: int, per_block: int) -> np.ndarray:
    """sig: (163840,) -> (5*per_block, win). Windows are evenly spaced INSIDE each block."""
    starts = np.round(np.linspace(0, BLOCK - win, per_block)).astype(int)
    out = []
    for b in range(N_BLOCKS):
        seg = sig[b * BLOCK:(b + 1) * BLOCK]
        out.append(np.stack([seg[s:s + win] for s in starts]))
    return np.concatenate(out)



def find_raw_dir(raw_dir: Path) -> Path:
    """Locate the folder that really contains H_5HZ.txt (searches sub-folders too)."""
    raw_dir = Path(raw_dir)
    if (raw_dir / "H_5HZ.txt").exists():
        return raw_dir
    hits = sorted(raw_dir.rglob("H_5HZ.txt")) if raw_dir.exists() else []
    if hits:
        print(f"[info] raw files found in: {hits[0].parent}")
        return hits[0].parent
    raise SystemExit(
        f"\nCould not find H_5HZ.txt in or under: {raw_dir.resolve()}\n"
        "Check that you unzipped Raw_data.zip and pass the folder that holds the 24 .txt files,\n"
        "e.g.  --raw_dir \"C:\\Users\\you\\Desktop\\My Project\\Raw data\"\n"
        "Tip: run  dir /s /b H_5HZ.txt  (Windows) to see where the file really is.")

# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, type=Path)
    ap.add_argument("--out", default="hustmotor_1024.npz", type=Path)
    ap.add_argument("--win", type=int, default=1024)
    ap.add_argument("--n_per_file", type=int, default=1000, help="samples per (state, speed)")
    ap.add_argument("--vib_channel", choices=list(CHAN), default="X",
                    help="vibration axis. Z is NOT recommended (see README).")
    ap.add_argument("--norm", choices=["zscore", "rms", "minmax", "center", "none"], default="zscore")
    ap.add_argument("--split", choices=["block", "random"], default="block",
                    help="block: blocks 0-3 train / block 4 test (no overlap leakage, 800/200). "
                         "random: random 800/200 as in the paper (overlapping windows can leak).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.raw_dir = find_raw_dir(args.raw_dir)
    per_block = args.n_per_file // N_BLOCKS
    assert per_block * N_BLOCKS == args.n_per_file, "n_per_file must be divisible by 5"
    rng = np.random.default_rng(args.seed)

    V, A, Y, D, B, W, S = [], [], [], [], [], [], []
    RV, RA, DV, DA, ST = [], [], [], [], []   # raw per-window RMS / DC (before normalisation) + start index
    for c_idx, cname in enumerate(CLASSES):
        for hz, d_idx in SPEEDS.items():
            f = args.raw_dir / f"{cname}_{hz}HZ.txt"
            raw = read_raw(f)
            vib = block_windows(raw[:, CHAN[args.vib_channel]], args.win, per_block)
            aco = block_windows(raw[:, SOUND_COL], args.win, per_block)   # same indices -> synchronous
            n = len(vib)
            blk = np.repeat(np.arange(N_BLOCKS), per_block)
            if args.split == "block":
                split = (blk == N_BLOCKS - 1).astype(np.int8)               # 1 = test
            else:
                split = np.zeros(n, np.int8)
                split[rng.permutation(n)[: int(0.2 * n)]] = 1
            rms_ = lambda x: x.astype(np.float64).std(1).astype(np.float32)   # RMS after removing DC
            RV.append(rms_(vib)); RA.append(rms_(aco))
            DV.append(vib.mean(1).astype(np.float32)); DA.append(aco.mean(1).astype(np.float32))
            st = np.round(np.linspace(0, BLOCK - args.win, per_block)).astype(np.int32)
            ST.append(np.tile(st, N_BLOCKS) + np.repeat(np.arange(N_BLOCKS), per_block) * BLOCK)  # row index in raw file
            if args.norm == "rms":      # centre each window, scale later by ONE channel-wide RMS (amplitude preserved)
                V.append(normalise(vib, "center")); A.append(normalise(aco, "center"))
            else:
                V.append(normalise(vib, args.norm)); A.append(normalise(aco, args.norm))
            Y.append(np.full(n, c_idx, np.int8)); D.append(np.full(n, d_idx, np.int8))
            B.append(blk.astype(np.int8)); W.append(np.tile(np.arange(per_block), N_BLOCKS).astype(np.int16))
            S.append(split)
            print(f"{f.name:18s} -> {n} samples  (train {int((split==0).sum())} / test {int((split==1).sum())})")

    if args.norm == "rms":
        trn = np.concatenate(S) == 0                      # statistics from TRAIN-split windows only
        Vc, Ac = np.concatenate(V), np.concatenate(A)
        gv = np.sqrt((Vc[trn].astype(np.float64) ** 2).mean()); ga = np.sqrt((Ac[trn].astype(np.float64) ** 2).mean())
        V, A = [Vc / gv], [Ac / ga]
        print(f"channel-wide RMS used: vib={gv:.5f} ac={ga:.5f}")
    np.savez(args.out,
             vib=np.concatenate(V), ac=np.concatenate(A), y=np.concatenate(Y),
             domain=np.concatenate(D), block=np.concatenate(B), win=np.concatenate(W),
             split=np.concatenate(S),
             rms_v=np.concatenate(RV), rms_a=np.concatenate(RA), dc_v=np.concatenate(DV), dc_a=np.concatenate(DA),
             start=np.concatenate(ST),
             classes=np.array(CLASSES), domains=np.array(["B1_5Hz", "B2_10Hz", "B3_20Hz", "B4_30Hz"]),
             meta=np.array([f"win={args.win}", f"vib={args.vib_channel}", f"norm={args.norm}",
                            f"split={args.split}", f"fs={FS}"]))
    print(f"\nSaved {args.out}  vib/ac shape = ({sum(len(x) for x in V)}, {args.win})")


# ------------------------------------------------- helper for the 4 DG tasks (Table 4)
def load_dg_task(npz_path, target_domain: int, eval_on="test", norm="zscore"):
    """Leave-one-domain-out. target_domain 0..3 == Task 9..12 (B1..B4).
    Returns dict of (vib, ac, y) for source-train, source-val(test split) and unseen target.
    eval_on: 'test' (200/class, as in paper) or 'all' (1000/class).

    norm: 'zscore' (default) -> signals exactly as stored (per-window z-score, no leakage possible).
          'rms'    -> amplitude-preserving: DC removed, then divided by ONE RMS per channel that is
                      computed ONLY from this task's source-train windows (the unseen speed and the
                      source val/test windows never influence it). Needs a v1.1 file (rms_v / rms_a).
    The scales used are returned as t["scale"] = (vib_scale, ac_scale) (None for zscore); report them.
    """
    d = np.load(npz_path)
    src = d["domain"] != target_domain
    tgt = ~src
    tr = src & (d["split"] == 0)
    va = src & (d["split"] == 1)
    te = tgt & ((d["split"] == 1) if eval_on == "test" else True)
    vib, ac, scale = d["vib"], d["ac"], None
    if norm == "rms":
        if "rms_v" not in d.files:
            raise ValueError("norm='rms' needs a v1.1 file (with rms_v / rms_a); use hustmotor_1024_v1.1.npz")
        # z-score stored (w-mean)/std and rms_* is std, so vib*rms_v is the DC-removed signal in original units
        vib = vib * d["rms_v"][:, None]
        ac = ac * d["rms_a"][:, None]
        sv = float(np.sqrt((vib[tr].astype(np.float64) ** 2).mean()))
        sa = float(np.sqrt((ac[tr].astype(np.float64) ** 2).mean()))
        vib, ac, scale = vib / sv, ac / sa, (sv, sa)
    elif norm != "zscore":
        raise ValueError("norm must be 'zscore' or 'rms'")
    pack = lambda m: (vib[m], ac[m], d["y"][m])
    return {"train": pack(tr), "val": pack(va), "target": pack(te), "train_domains": d["domain"][tr], "scale": scale}


if __name__ == "__main__":
    main()
