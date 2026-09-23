"""Framing constants and sample/frame mapping - NumPy only, no PyTorch.

Split out of `stft.py` deliberately. The deployed inference path (ONNX Runtime)
and the classical baselines need these constants but have no business pulling in
PyTorch: a teammate who only wants to run the shipped model should not need a
2 GB CUDA download, and on a machine where PyTorch cannot load at all - Windows
Smart App Control blocks its unsigned DLLs, for instance - the model must still
run.

`stft.py` re-exports everything here, so existing imports keep working.

These values are NOT free parameters: they must match the model that produced
the weights, or the weights are meaningless. The shipped configuration is

    n_fft = 512  ->  32 ms analysis window
    hop   = 256  ->  16 ms, the chunk size agreed with the hardware team
    window = hann(512) ** 0.5   (sqrt-Hann, analysis AND synthesis)

and that is what every checkpoint in `checkpoints/` was trained at.

The 32 ms window is also where the latency budget goes: algorithmic latency is
chunk buffering (hop) plus the overlap-add delay (win - hop), so with 512/256 it
is 32 ms before a single multiply happens - and the problem statement asks for
under 32 ms. The only way out is a shorter window, which is why the size is a
parameter here instead of a literal.

    SIH_NFFT=320 SIH_HOP=160 python -m src.train ...   ->  10 + 10 = 20 ms

Set SIH_NFFT (and optionally SIH_HOP, default n_fft/2) in the environment and
every part of the pipeline - training, export, the streaming runtime, the
classical baselines - follows. UNSET IS THE SHIPPED 512/256; nothing changes for
anything that does not opt in. A model must be run at the size it was trained
at: `StreamingEnhancer` checks the exported graph against these constants and
refuses a mismatch rather than producing quiet nonsense.
"""
from __future__ import annotations

import os

import numpy as np

SR = 16000


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        v = int(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not an integer") from None
    if v <= 0:
        raise ValueError(f"{name} must be positive, got {v}")
    return v


# The ERB band split per FFT size: (linear bins kept as-is, ERB bands above).
# The first number sets the crossover frequency - erb_subband_1/nfft*fs, which
# is ~2 kHz for every entry here, so the low band keeps full resolution where
# the formants are. The second is the GTCRN band count above it. Upstream is
# the 512 row; the others keep the same proportions so the architecture is
# unchanged in shape, only in size.
ERB_SUBBANDS = {512: (65, 64), 384: (49, 48), 320: (41, 40), 256: (33, 32)}

N_FFT = _env_int("SIH_NFFT", 512)
HOP = _env_int("SIH_HOP", N_FFT // 2)
WIN = N_FFT

if N_FFT not in ERB_SUBBANDS:
    raise ValueError(
        f"SIH_NFFT={N_FFT} has no ERB band split; known sizes: "
        f"{sorted(ERB_SUBBANDS)}. Add a row to ERB_SUBBANDS in src/framing.py "
        "(and re-train - no existing checkpoint will load)."
    )
if HOP > WIN:
    raise ValueError(f"SIH_HOP={HOP} exceeds the window {WIN}")


def erb_subbands(nfft: int | None = None) -> tuple[int, int]:
    """(erb_subband_1, erb_subband_2) for this FFT size."""
    return ERB_SUBBANDS[N_FFT if nfft is None else int(nfft)]


def n_bands(nfft: int | None = None) -> int:
    """Width of the band axis the encoder sees: linear bins + ERB bands."""
    return sum(erb_subbands(nfft))


def _half(n: int) -> int:
    """One encoder ConvBlock, kernel (1,5) stride (1,2) padding (0,2)."""
    return (n + 2 * 2 - 5) // 2 + 1


def freq_width(nfft: int | None = None) -> int:
    """Band axis reaching the dual-path RNN - two stride-2 convs down from
    `n_bands`. 33 at the shipped 512, and the F in every cache shape."""
    return _half(_half(n_bands(nfft)))


def cache_shapes(width: int = 16, nfft: int | None = None):
    """(conv, tra, inter) streaming cache shapes.

    conv:  (enc/dec, B, C, sum of (kT-1)*dilation = 2*(1+2+5), F)
    tra:   (enc/dec, 3 GTConv blocks, 1, B, GRU hidden = C)
    inter: (2 DPGRNNs, 1, B*F, hidden = C)

    Kept here, beside the constants, so the torch-free inference path can state
    the shapes without importing the model. Anything running an EXPORTED model
    should still read them from the ONNX graph's own inputs - that cannot go
    stale.
    """
    f = freq_width(nfft)
    w = int(width)
    return (2, 1, w, 16, f), (2, 3, 1, 1, w), (2, 1, f, w)


CONV_CACHE, TRA_CACHE, INTER_CACHE = cache_shapes(16)


def zero_caches_np():
    """Fresh zeroed caches for the start of a stream."""
    return (np.zeros(CONV_CACHE, dtype="float32"),
            np.zeros(TRA_CACHE, dtype="float32"),
            np.zeros(INTER_CACHE, dtype="float32"))


def np_window() -> np.ndarray:
    """sqrt-Hann analysis/synthesis window, matching torch.hann_window(WIN)**0.5."""
    return (np.hanning(WIN + 1)[:WIN] ** 0.5).astype(np.float32)


def n_frames(n_samples: int) -> int:
    """Frame count torch.stft produces for n_samples with center=True."""
    return n_samples // HOP + 1


def samples_to_frame_mask(sample_mask, n_frame: int) -> np.ndarray:
    """Collapse a sample-level boolean mask to STFT frames.

    center=True means frame t is centred on sample t*hop and spans
    [t*hop - win/2, t*hop + win/2). A frame counts as active if ANY of its
    samples are. Getting this alignment wrong silently poisons the
    transient-weighted loss and the burst-local metrics, so it is unit-tested
    in tests/test_core.py.
    """
    sample_mask = np.asarray(sample_mask, dtype=bool)
    out = np.zeros(n_frame, dtype=bool)
    half = WIN // 2
    for t in range(n_frame):
        lo = max(0, t * HOP - half)
        hi = min(len(sample_mask), t * HOP + half)
        if lo < hi and sample_mask[lo:hi].any():
            out[t] = True
    return out
