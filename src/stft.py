"""STFT front end.

These constants are NOT free parameters: they must match GTCRN's own framing
(see third_party/gtcrn/infer.py) or the pretrained weights are meaningless.

    n_fft = 512  ->  32 ms analysis window
    hop   = 256  ->  16 ms, the chunk size agreed with the hardware team
    window = hann(512) ** 0.5   (sqrt-Hann, used for BOTH analysis and synthesis)

The 32 ms window is also where the latency budget is spent: algorithmic latency
equals the window length, so it is 32 ms before any compute. See src/bench.py.
"""
import torch

SR = 16000
N_FFT = 512
HOP = 256
WIN = 512

_WINDOW_CACHE: dict = {}


def window(device=None, dtype=torch.float32) -> torch.Tensor:
    key = (str(device), dtype)
    if key not in _WINDOW_CACHE:
        _WINDOW_CACHE[key] = torch.hann_window(WIN, device=device, dtype=dtype).pow(0.5)
    return _WINDOW_CACHE[key]


def stft(x: torch.Tensor) -> torch.Tensor:
    """(..., n) waveform -> (..., 257, T, 2) real/imag, GTCRN's expected input."""
    spec = torch.stft(x, N_FFT, HOP, WIN, window(x.device), return_complex=True)
    return torch.view_as_real(spec)


def istft(spec: torch.Tensor, length: int | None = None) -> torch.Tensor:
    """(..., 257, T, 2) -> (..., n) waveform."""
    return torch.istft(
        torch.view_as_complex(spec.contiguous()),
        N_FFT, HOP, WIN, window(spec.device), length=length,
    )


def n_frames(n_samples: int) -> int:
    """Frame count torch.stft produces for n_samples with center=True."""
    return n_samples // HOP + 1


def samples_to_frame_mask(sample_mask, n_frame: int):
    """Collapse a sample-level boolean mask to STFT frames.

    center=True means frame t is centred on sample t*hop and spans
    [t*hop - win/2, t*hop + win/2). A frame counts as active if ANY of its
    samples are. Getting this alignment wrong silently poisons the
    transient-weighted loss, so it is unit-tested in tests/test_stft.py.
    """
    import numpy as np
    sample_mask = np.asarray(sample_mask, dtype=bool)
    out = np.zeros(n_frame, dtype=bool)
    half = WIN // 2
    for t in range(n_frame):
        lo = max(0, t * HOP - half)
        hi = min(len(sample_mask), t * HOP + half)
        if lo < hi and sample_mask[lo:hi].any():
            out[t] = True
    return out
