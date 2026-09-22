"""Step 3 of the real-combat data: cut noise-only segments and split by video.

Uses the PANNs tags from tag_drive_audio.py. Measured on the 12.7 h of combat
audio (22 Sep 2026): the median window scores 0.47 for speech - more than half
of it has voices in it - so selection is strict:

  keep a 1 s window only if   speech < 0.10  and  music < 0.10  and  level > -50 dBFS

Consecutive kept windows are merged, then 0.5 s is trimmed off BOTH ends of
every segment (a window that just misses a shout still borders it), and
segments shorter than 3 s are dropped (the loader loops short clips to fill a
4 s mixture, which sounds unnatural).

Splits are by VIDEO, never by segment (CLAUDE.md invariant 1): segments of one
video share a location, weapon and microphone. Compilations may re-use footage
from other uploads, so they always go to train. Roughly 15% of kept duration
goes to test and 5% to val.

Writes <prepared>/drive_combat/segments/<video>_<start_ms>.wav and
<prepared>/drive_combat/segments.json. `add_drive_to_manifest.py` merges that
into manifests/manifest.json as the `combat_real` category.

    python scripts/select_drive_segments.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf

SPEECH = ["Speech", "Male speech, man speaking", "Female speech, woman speaking",
          "Child speech, kid speaking", "Conversation", "Narration, monologue", "Babbling",
          "Speech synthesizer", "Shout", "Bellow", "Whoop", "Yell", "Battle cry",
          "Children shouting", "Screaming", "Whispering", "Laughter", "Crying, sobbing",
          "Wail, moan", "Groan", "Singing", "Choir", "Chant", "Mantra", "Male singing",
          "Female singing", "Rapping", "Humming", "Crowd", "Hubbub, speech noise, speech babble",
          "Cheering", "Children playing"]
MUSIC = ["Music", "Musical instrument"]
# The classifier names distant gunfire "Firecracker"/"Fireworks" and shell
# impacts "Thump, thud" - measured as the top labels of otherwise-unlabelled
# kept windows - so they count toward the combat types.
TYPES = {
    "gunfire": ["Gunshot, gunfire", "Machine gun", "Fusillade", "Cap gun", "Firecracker", "Fireworks"],
    "explosion": ["Explosion", "Artillery fire", "Boom", "Eruption", "Burst, pop", "Thump, thud", "Thunder", "Rumble"],
    "vehicle": ["Vehicle", "Motor vehicle (road)", "Truck", "Engine", "Heavy engine (low frequency)",
                "Medium engine (mid frequency)", "Idling", "Accelerating, revving, vroom"],
    "aircraft": ["Aircraft", "Helicopter", "Fixed-wing aircraft, airplane", "Jet engine",
                 "Propeller, airscrew", "Aircraft engine"],
}
SR, WIN_S, TRIM_S, MIN_S = 16000, 2.0, 0.5, 3.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="C:/SIH26052_data/prepared/drive_combat")
    ap.add_argument("--speech-max", type=float, default=0.10)
    ap.add_argument("--music-max", type=float, default=0.10)
    ap.add_argument("--min-db", type=float, default=-50.0)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20260922)
    a = ap.parse_args()

    d = Path(a.dir)
    tags = d / "tags"
    labels = [r["display_name"] for r in csv.DictReader(
        open(tags / "class_labels_indices.csv", encoding="utf-8"))]
    L = {n: i for i, n in enumerate(labels)}
    ix = lambda names: [L[n] for n in names if n in L]  # noqa: E731
    sp_i, mu_i = ix(SPEECH), ix(MUSIC)
    ty_i = {k: ix(v) for k, v in TYPES.items()}
    titles = {r["id"]: r.get("title", "") for r in
              json.loads((d / "audio_index.json").read_text(encoding="utf-8"))["items"]}

    seg_dir = d / "segments"
    seg_dir.mkdir(exist_ok=True)
    per_video: dict[str, list[dict]] = {}
    for npz in sorted(tags.glob("*.npz")):
        vid = npz.stem
        z = np.load(npz)
        P, starts, lv = z["probs"].astype(np.float32), z["starts_s"], z["rms_db"]
        keep = (P[:, sp_i].max(1) < a.speech_max) & (P[:, mu_i].max(1) < a.music_max) & (lv > a.min_db)
        x = None
        segs = []
        i = 0
        while i < len(keep):
            if not keep[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(keep) and keep[j + 1]:
                j += 1
            t0, t1 = float(starts[i]) + TRIM_S, float(starts[j]) + WIN_S - TRIM_S
            if t1 - t0 >= MIN_S:
                if x is None:
                    x, _ = sf.read(d / f"{vid}.wav", dtype="float32")
                clip = x[int(t0 * SR):int(t1 * SR)]
                scores = {k: float(P[i:j + 1][:, v].max(1).mean()) for k, v in ty_i.items()}
                kind = max(scores, key=scores.get) if max(scores.values()) > 0.2 else "ambient"
                name = f"{vid}_{int(t0 * 1000):08d}.wav"
                sf.write(seg_dir / name, clip, SR, subtype="PCM_16")
                segs.append({"path": str(seg_dir / name), "video": vid, "t0": round(t0, 2),
                             "dur": round(len(clip) / SR, 2), "type": kind,
                             "speech_max": round(float(P[i:j + 1][:, sp_i].max()), 3)})
            i = j + 1
        if segs:
            per_video[vid] = segs

    # ---- split by video --------------------------------------------------
    rng = np.random.default_rng(a.seed)
    vids = sorted(per_video)
    comp = {v for v in vids if "compilation" in titles.get(v, "").lower()}
    order = [v for v in rng.permutation(vids) if v not in comp]
    total = sum(s["dur"] for v in vids for s in per_video[v])
    split = {v: "train" for v in vids}
    acc = {"test": 0.0, "val": 0.0}
    for v in order:
        dur = sum(s["dur"] for s in per_video[v])
        for name, frac in (("test", a.test_frac), ("val", a.val_frac)):
            if acc[name] + dur <= frac * total * 1.15 and acc[name] < frac * total:
                split[v] = name
                acc[name] += dur
                break

    out = {"train": [], "val": [], "test": []}
    for v in vids:
        for s in per_video[v]:
            out[split[v]].append({**s, "group": f"drive:{v}", "source": "drive_combat",
                                  "category": "combat_real"})
    groups = {k: {r["group"] for r in out[k]} for k in out}
    assert not (groups["train"] & groups["val"] or groups["train"] & groups["test"]
                or groups["val"] & groups["test"]), "video leaked across splits"

    summary = {}
    for k, rows in out.items():
        types = {}
        for r in rows:
            types[r["type"]] = types.get(r["type"], 0.0) + r["dur"]
        summary[k] = {"videos": len(groups[k]), "segments": len(rows),
                      "hours": round(sum(r["dur"] for r in rows) / 3600, 3),
                      "by_type_min": {t: round(s / 60, 1) for t, s in sorted(types.items())}}
    (d / "segments.json").write_text(json.dumps(
        {"rule": {"speech_max": a.speech_max, "music_max": a.music_max, "min_db": a.min_db,
                  "trim_s": TRIM_S, "min_s": MIN_S}, "seed": a.seed, "summary": summary,
         "splits": out}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
