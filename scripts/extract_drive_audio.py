"""Step 1 of the real-combat data: pull 16 kHz mono audio out of the videos.

Reads `<raw>/drive_combat/<id>.mp4` (from download_drive_combat.py) and writes
`<prepared>/drive_combat/<id>.wav` (PCM16, 16 kHz, mono) plus `audio_index.json`
with each file's duration, peak and clipping share.

The clipping share matters: a camera mic next to a rifle saturates, and that
flat-topped audio is exactly what our own mic will capture in the field - it
is kept, not filtered, but it is measured so the mixer's behaviour on it can
be checked later.

Decoding uses PyAV (already a project dependency), so no ffmpeg install is
needed on Windows. Resumable: existing wavs are skipped.

    python scripts/extract_drive_audio.py [--workers 6]
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SR = 16000


def extract(src: str, dst: str) -> dict:
    import av
    import soundfile as sf

    out = Path(dst)
    if out.exists():
        x, sr = sf.read(out, dtype="float32")
    else:
        with av.open(src) as c:
            if not c.streams.audio:
                return {"id": Path(src).stem, "error": "no audio stream"}
            rs = av.AudioResampler(format="s16", layout="mono", rate=SR)
            parts = []
            for frame in c.decode(c.streams.audio[0]):
                for f in rs.resample(frame):
                    parts.append(f.to_ndarray().reshape(-1))
            for f in rs.resample(None):
                parts.append(f.to_ndarray().reshape(-1))
        if not parts:
            return {"id": Path(src).stem, "error": "no decodable audio"}
        pcm = np.concatenate(parts)
        tmp = out.with_suffix(".tmp.wav")
        sf.write(tmp, pcm, SR, subtype="PCM_16")
        tmp.replace(out)
        x = pcm.astype(np.float32) / 32768.0
    a = np.abs(x)
    return {"id": out.stem, "dur_s": round(len(x) / SR, 2),
            "peak": round(float(a.max()), 4),
            "clipped_pct": round(100.0 * float(np.mean(a > 0.99)), 3),
            "rms_dbfs": round(float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="C:/SIH26052_data/raw/drive_combat")
    ap.add_argument("--out", default="C:/SIH26052_data/prepared/drive_combat")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    idx = json.loads((raw / "index.json").read_text(encoding="utf-8"))
    titles = {i["id"]: i["title"] for i in idx["items"]}
    vids = sorted(p for p in raw.glob("*.mp4") if p.with_suffix(".mp4.done").exists())

    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(extract, str(v), str(out / f"{v.stem}.wav")): v for v in vids}
        for n, fu in enumerate(futs, 1):
            r = fu.result()
            r["title"] = titles.get(r["id"], "")
            rows.append(r)
            print(f"[{n:3d}/{len(vids)}] {r.get('dur_s', 0):8.1f} s  "
                  f"clip {r.get('clipped_pct', 0):6.2f}%  {r.get('error', '')} {r['id']}", flush=True)

    ok = [r for r in rows if "error" not in r]
    total = sum(r["dur_s"] for r in ok)
    (out / "audio_index.json").write_text(json.dumps(
        {"sr": SR, "total_hours": round(total / 3600, 2), "items": rows},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{len(ok)}/{len(rows)} videos with audio, {total / 3600:.2f} h total, "
          f"median {np.median([r['dur_s'] for r in ok]) / 60:.1f} min, "
          f"heavily clipped (>1% samples): {sum(r['clipped_pct'] > 1 for r in ok)}")


if __name__ == "__main__":
    main()
