"""Step 4 of the real-combat data: a second, independent check for speech.

PANNs (step 2) decides per window from sound-event statistics; a shout under
machine-gun fire can score low for "Speech" and slip through. Whisper is a
different kind of detector - it tries to recognise words - so it catches a
different set of misses. A segment is DROPPED if Whisper transcribes words it
is actually confident about:

    no_speech_prob < 0.5  and  avg_logprob > -1.0  and  at least 2 words

Whisper also hallucinates short phrases ("Thank you.") on pure noise; those
come with a high no_speech_prob or a very low avg_logprob and are not counted.
Every decision is written back into segments.json (`whisper` field) so it can
be audited. temperature=0.0 disables Whisper's sampling fallback (CLAUDE.md
trap 15: the default makes the same file score differently run to run).

    python scripts/whisper_check_segments.py [--model small]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import soundfile as sf


def verdict(segs) -> tuple[bool, str, float, float]:
    words, best_ns, best_lp, heard = 0, 1.0, -9.0, []
    for s in segs:
        best_ns = min(best_ns, s.no_speech_prob)
        best_lp = max(best_lp, s.avg_logprob)
        if s.no_speech_prob < 0.5 and s.avg_logprob > -1.0:
            w = s.text.strip()
            words += len(w.split())
            heard.append(w)
    return words >= 2, " | ".join(heard)[:200], best_ns, best_lp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="C:/SIH26052_data/prepared/drive_combat")
    ap.add_argument("--model", default="small")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--device", default="cpu", help="cuda: run in WSL, ~10x faster")
    a = ap.parse_args()
    from faster_whisper import WhisperModel

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.audio import local_path
    d = Path(local_path(a.dir))
    meta = json.loads((d / "segments.json").read_text(encoding="utf-8"))
    model = WhisperModel(a.model, device=a.device, cpu_threads=a.threads,
                         compute_type="float16" if a.device == "cuda" else "int8")
    rows = [r for k in ("train", "val", "test") for r in meta["splits"][k]]
    todo = [r for r in rows if "whisper" not in r]
    t0, done_s = time.time(), 0.0
    for n, r in enumerate(todo, 1):
        x, _ = sf.read(local_path(r["path"]), dtype="float32")
        segs, info = model.transcribe(x, temperature=0.0, beam_size=1, vad_filter=False,
                                      condition_on_previous_text=False)
        speech, heard, ns, lp = verdict(list(segs))
        r["whisper"] = {"speech": speech, "heard": heard, "lang": info.language,
                        "min_no_speech": round(ns, 3), "max_logprob": round(lp, 3)}
        done_s += r["dur"]
        if n % 50 == 0 or n == len(todo):
            (d / "segments.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
            el = time.time() - t0
            print(f"[{n:5d}/{len(todo)}] {done_s / el:5.1f}x real time, "
                  f"flagged so far {sum(1 for q in rows if q.get('whisper', {}).get('speech'))}", flush=True)

    for k in ("train", "val", "test"):
        seg = meta["splits"][k]
        bad = [r for r in seg if r["whisper"]["speech"]]
        print(f"{k:5s}: {len(bad)}/{len(seg)} segments flagged as speech "
              f"({sum(r['dur'] for r in bad) / 60:.1f} of {sum(r['dur'] for r in seg) / 60:.1f} min)")


if __name__ == "__main__":
    main()
