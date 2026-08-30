"""Repair an over-suppressed enhancement, and level it for listening.

WHY THIS EXISTS. On a low-SNR field recording the shipped model can suppress far
past the noise: measured on `test-result/`, 99.2% of the output energy landed
below 1 kHz and *exactly zero* survived above 2 kHz. The voice is still there but
every consonant cue is gone, so it reads as "muffled and unclear" no matter how
loud you make it. Gain alone cannot fix that - it gives you a louder muffle.

WHAT IT DOES. Two stages, deliberately separate:

  1. RESTORE (in scope for this repo - it changes the suppression decision).
     Recover the implied per-bin gain the model applied, G = |After| / |Before|,
     and refuse to let it cut deeper than the noise in that band justifies:

         G_final = max(G_model, restore * floor(f) * wiener(t,f) * (1 - burst(t)))

     The three factors on the lift each withhold it for a different reason:
     `floor` will not reopen a band that genuinely carries the noise, `wiener`
     will not reopen a bin with no signal in it, and `burst` will not reopen
     anything during a transient - so the gunfire still dies. The noisy phase is
     reused, as every mask-based suppressor does.

     Measured on the test-result pair at these defaults, this puts 1-4 kHz back
     (0.8% of the output energy -> 16.2%, against 10.6% for genuine clean
     speech) while transient frames end up 21.1 dB below the input rather than
     11.0 dB, both normalised for the speech gain.

  2. LEVEL (out of scope, see CLAUDE.md - the model applies no gain; the analog
     path downstream does). Provided anyway because you cannot judge stage 1 by
     ear at -31 dBFS. Static gain to a target active-speech level, then the same
     soft limiter the mixer models. `--no-level` skips it.

    python scripts/post_enhance.py --before test-result/before.wav \
        --after test-result/after.wav --out-dir test-result/repaired
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.signal import correlate

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import audio as A                                        # noqa: E402
from src.framing import HOP, N_FFT, SR                            # noqa: E402
# The tested WOLA pair - reused rather than reimplemented so this file cannot
# drift from the framing the baselines and the model already agree on.
from src.baselines.classical import _analyse, _synthesise         # noqa: E402

_EPS = 1e-10


def measure_lag(before: np.ndarray, after: np.ndarray) -> tuple[int, float]:
    """Sample offset of `after` relative to `before`, by waveform correlation.

    Measured, never assumed. `stream_demo.run_file` writes the two files at equal
    length, but a recording captured any other way (a live pass, a separate
    take, a manual trim) carries an arbitrary offset - the test-result pair is
    1280 samples apart. Correlating an unaligned pair silently compares speech
    against noise, and every conclusion drawn from it is then wrong.

    Correlation is taken over the window where the ENHANCED signal is strongest,
    because that is the only stretch guaranteed to hold speech in both files.
    """
    spec, _, _ = _analyse(after)
    energy = np.sqrt((np.abs(spec) ** 2).mean(axis=1))
    centre = int(np.argmax(energy)) * HOP
    w = int(0.6 * SR)
    start = max(0, min(centre - w // 2, max(len(after) - w, 0)))
    seg = after[start:start + w]
    lo, hi = max(0, start - SR), min(len(before), start + w + SR)
    ref = before[lo:hi]
    if len(ref) <= len(seg) or len(seg) == 0:
        return 0, 0.0
    num = correlate(ref - ref.mean(), seg - seg.mean(), mode="valid")
    den = np.array([np.linalg.norm(ref[i:i + len(seg)] - ref[i:i + len(seg)].mean())
                    for i in range(len(num))]) * np.linalg.norm(seg - seg.mean())
    corr = num / (den + _EPS)
    best = int(np.argmax(np.abs(corr)))
    return (lo + best) - start, float(corr[best])


def align(before: np.ndarray, after: np.ndarray, lag: int):
    """Trim both to a common timeline given the offset of `after`."""
    if lag > 0:
        before = before[lag:]
    elif lag < 0:
        after = after[-lag:]
    n = min(len(before), len(after))
    return before[:n], after[:n]


def noise_floor(mag: np.ndarray, pct: float = 30.0) -> np.ndarray:
    """Per-bin stationary noise magnitude, as a percentile over time.

    A LOW percentile (the textbook 10th) is wrong here. It reports the quietest
    moment rather than the typical noise level, so the Wiener term below sees a
    huge posterior SNR in almost every bin, opens to unity, and hands back the
    unprocessed input - which is exactly what the first version of this file did.
    """
    return np.percentile(mag, pct, axis=0)


def suppression_floor(noise: np.ndarray, depth_db: float, min_floor_db: float,
                      max_floor_db: float) -> np.ndarray:
    """Per-BIN cap on how deep the suppression may cut, in dB.

    The insight the fix rests on: how hard a band should be cut depends on how
    much noise is actually in it. This recording is a low-mid roar - 72% of the
    input energy sits in 500-1000 Hz, while 4-8 kHz carries 1.4%. Cutting the
    quiet bands by 40 dB buys almost no noise reduction and costs every
    consonant. So aim for a FLAT residual noise floor: cut each bin only by the
    amount that brings it down to a common target, and no further.

    `max_floor_db` is what stops that argument running away. Taken literally the
    rule fully reopens any band quieter than the target, and since the makeup
    gain downstream is +10 dB or more, that turns a band holding 1.4% of the
    input into 25% of the output - audible hiss, measured. No band is ever
    reopened past this ceiling.
    """
    noise_db = 20 * np.log10(noise + _EPS)
    target = noise_db.max() - depth_db
    return np.clip(target - noise_db, min_floor_db, max_floor_db)


def burst_weight(mag_before: np.ndarray, guard_db: float) -> np.ndarray:
    """1.0 on frames that look like a transient, 0.0 on ordinary frames.

    Without this the floor is static in time, so it re-admits the gunfire at
    floor level - undoing the one thing the model is here to do. Bursts are
    found as frames whose broadband energy jumps above the running median.
    """
    fe = 20 * np.log10(np.sqrt((mag_before ** 2).mean(axis=1)) + _EPS)
    k = 31
    pad = np.pad(fe, (k // 2, k // 2), mode="edge")
    med = np.array([np.median(pad[i:i + k]) for i in range(len(fe))])
    excess = (fe - med) / max(guard_db, _EPS)
    return np.clip(excess, 0.0, 1.0)


def restore_mask(mag_before, mag_after, noise, restore, depth_db, min_floor_db,
                 max_floor_db, alpha, guard_db, max_gain_db):
    """G = max(model mask, lift), where lift re-opens over-suppressed bins.

        lift(t,f) = floor(f) * wiener(t,f) * (1 - burst(t))

    Each factor withholds the lift for a different reason: `floor` refuses to
    reopen bands that genuinely carry the noise, `wiener` refuses to reopen a bin
    with no signal in it (so silence stays silent instead of gaining a noise
    bed), and `burst` refuses to reopen anything during a transient.
    """
    g_model = mag_after / (mag_before + _EPS)

    # Over-subtracted posterior SNR: alpha > 1 means a bin must clear the noise
    # estimate by a margin before it counts as signal.
    snr_post = np.maximum((mag_before / (alpha * noise[None, :] + _EPS)) ** 2 - 1.0, 0.0)
    g_wiener = snr_post / (1.0 + snr_post)

    floor = 10 ** (suppression_floor(noise, depth_db, min_floor_db,
                                     max_floor_db) / 20.0)
    lift = floor[None, :] * g_wiener * (1.0 - burst_weight(mag_before, guard_db))[:, None]

    g = np.maximum(g_model, restore * lift)
    return np.clip(g, 0.0, 10 ** (max_gain_db / 20.0))


def level(x: np.ndarray, target_dbfs: float, limit: float):
    """Static gain to a target active-speech level, then the soft limiter."""
    cur = A.active_rms(x, SR)
    if cur <= _EPS:
        return x.astype(np.float32), 0.0
    gain = 10 ** (target_dbfs / 20.0) / cur
    y = A.soft_limit(x * gain, limit)
    return y.astype(np.float32), 20 * np.log10(gain)


def band_report(x: np.ndarray, label: str) -> None:
    """Energy share per band - the measurement that exposed the problem."""
    spec, _, _ = _analyse(x)
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    total = (mag ** 2).sum() + _EPS
    shares = []
    for f1, f2 in [(0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]:
        sel = (freqs >= f1) & (freqs < f2)
        shares.append(100 * (mag[:, sel] ** 2).sum() / total)
    print("  %-24s active %7.1f dBFS | peak %6.1f dBFS |  0-.5k %5.1f  .5-1k %5.1f"
          "  1-2k %5.1f  2-4k %5.1f  4-8k %5.1f"
          % (label, 20 * np.log10(A.active_rms(x, SR) + _EPS),
             20 * np.log10(np.abs(x).max() + _EPS), *shares))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--lag", type=int, default=None,
                   help="offset of after vs before in samples (default: measure it)")
    p.add_argument("--restore", type=float, default=1.0,
                   help="0 = model output untouched, 1 = full restoration")
    p.add_argument("--depth-db", type=float, default=25.0,
                   help="suppression aimed at the NOISIEST band; quieter bands "
                        "are cut proportionally less")
    p.add_argument("--min-floor-db", type=float, default=-30.0,
                   help="hardest suppression any bin may receive")
    p.add_argument("--max-floor-db", type=float, default=-14.0,
                   help="gentlest suppression any bin may receive; stops quiet "
                        "bands being reopened into hiss")
    p.add_argument("--noise-pct", type=float, default=30.0,
                   help="percentile over time used as the noise estimate")
    p.add_argument("--alpha", type=float, default=1.5,
                   help="noise over-subtraction factor for the lift gate")
    p.add_argument("--guard-db", type=float, default=9.0,
                   help="energy jump over the running median that counts as a burst")
    p.add_argument("--max-gain-db", type=float, default=0.0,
                   help="mask ceiling; >0 would let the mask amplify")
    p.add_argument("--target-dbfs", type=float, default=-20.0)
    p.add_argument("--limit", type=float, default=0.89)
    p.add_argument("--no-level", action="store_true",
                   help="skip the gain stage; emit suppression-only output")
    p.add_argument("--tag", default="",
                   help="suffix for the output filenames, for A/B-ing settings")
    args = p.parse_args()

    before = A.load_audio(args.before)
    after = A.load_audio(args.after)
    print("loaded  before %.2f s | after %.2f s" % (len(before) / SR, len(after) / SR))

    if args.lag is None:
        lag, corr = measure_lag(before, after)
        print("measured lag %d samples (%.1f ms), correlation %.3f"
              % (lag, lag / SR * 1000, corr))
        if abs(corr) < 0.3:
            print("  WARNING: weak correlation - these may not be the same take. "
                  "Pass --lag explicitly if you know the offset.")
    else:
        lag = args.lag
        print("using supplied lag %d samples" % lag)

    before, after = align(before, after, lag)
    print("aligned to %.2f s" % (len(before) / SR))

    sb, w, nfr = _analyse(before)
    sa, _, _ = _analyse(after)
    nfr = min(len(sb), len(sa))
    sb, sa = sb[:nfr], sa[:nfr]

    g = restore_mask(np.abs(sb), np.abs(sa),
                     noise_floor(np.abs(sb), args.noise_pct),
                     args.restore, args.depth_db, args.min_floor_db,
                     args.max_floor_db, args.alpha, args.guard_db,
                     args.max_gain_db)
    repaired = _synthesise(g * sb, w, nfr, len(before))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t = args.tag
    outputs = {}

    if args.no_level:
        outputs["repaired%s.wav" % t] = repaired
    else:
        lev_after, ga = level(after, args.target_dbfs, args.limit)
        lev_rep, gr = level(repaired, args.target_dbfs, args.limit)
        print("levelling gain: model output %+.1f dB | repaired %+.1f dB" % (ga, gr))
        outputs["after_levelled%s.wav" % t] = lev_after   # loudness fix only
        outputs["after_repaired%s.wav" % t] = lev_rep     # restoration + loudness

    print("\n=== measured (energy share per band, %) ===")
    band_report(before, "before (noisy)")
    band_report(after, "after (model, as-is)")
    for name, y in outputs.items():
        band_report(y, name)

    for name, y in outputs.items():
        A.save_audio(args.out_dir / name, y)
    A.save_audio(args.out_dir / "before_aligned.wav", before)
    print("\nwrote %d files to %s" % (len(outputs) + 1, args.out_dir))


if __name__ == "__main__":
    main()
