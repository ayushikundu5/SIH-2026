"""Step 2 of the real-combat data: label every second with a sound classifier.

The combat videos are not noise-only: soldiers shout, narrators talk, some
have music. Training on that as "noise" would teach the model that voices are
noise - the one thing this project must never learn. So every 2 s window (1 s
hop) is scored by PANNs Cnn14 (Kong et al. 2020, AudioSet, 527 classes) and
the full probability matrix is stored, so the keep/drop thresholds can be
chosen from the measured distribution rather than guessed, and re-chosen
without re-running the network.

Needs PyTorch, so it runs in WSL:
    bash /mnt/c/SIH26052_data/wsl_run.sh scripts/tag_drive_audio.py

Output: <prepared>/drive_combat/tags/<id>.npz  with
    starts_s  (n,)        window start times
    probs     (n, 527)    float16 clip-wise probabilities per window
    rms_db    (n,)        window level, dBFS
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.audio import local_path  # noqa: E402

WIN_S, HOP_S, PANNS_SR = 2.0, 1.0, 32000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="C:/SIH26052_data/prepared/drive_combat")
    ap.add_argument("--ckpt", default="/root/panns_data/Cnn14_mAP=0.431.pth")
    ap.add_argument("--batch", type=int, default=48)
    a = ap.parse_args()

    import torch
    from panns_inference import AudioTagging

    d = Path(local_path(a.dir))
    out = d / "tags"
    out.mkdir(exist_ok=True)
    at = AudioTagging(checkpoint_path=a.ckpt, device="cuda" if torch.cuda.is_available() else "cpu")

    wavs = sorted(d.glob("*.wav"))
    for n, w in enumerate(wavs, 1):
        dst = out / f"{w.stem}.npz"
        if dst.exists():
            continue
        x, sr = sf.read(w, dtype="float32")
        x32 = resample_poly(x, PANNS_SR, sr).astype(np.float32)
        win, hop = int(WIN_S * PANNS_SR), int(HOP_S * PANNS_SR)
        if len(x32) < win:
            x32 = np.pad(x32, (0, win - len(x32)))
        starts = np.arange(0, len(x32) - win + 1, hop)
        probs, levels = [], []
        for i in range(0, len(starts), a.batch):
            seg = np.stack([x32[s:s + win] for s in starts[i:i + a.batch]])
            levels.append(10 * np.log10(np.mean(seg ** 2, axis=1) + 1e-10))
            clip, _ = at.inference(seg)
            probs.append(clip.astype(np.float16))
        np.savez_compressed(dst, starts_s=(starts / PANNS_SR).astype(np.float32),
                            probs=np.concatenate(probs), rms_db=np.concatenate(levels).astype(np.float32))
        print(f"[{n:3d}/{len(wavs)}] {len(starts):5d} windows  {w.stem}", flush=True)


if __name__ == "__main__":
    main()
