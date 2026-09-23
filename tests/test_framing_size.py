"""The STFT size is a parameter now - and must stay opt-in.

Shortening the analysis window is the only way under the 32 ms latency ceiling
(chunk buffering + overlap-add delay = hop + (win - hop) = win, before any
arithmetic). Making the size configurable risks three silent failures, one per
test group below:

  - the DEFAULT drifts, and every existing checkpoint quietly means something
    else - all of them were trained at 512/256;
  - the band arithmetic does not round-trip, so the decoder output does not line
    up with the encoder input at a new size;
  - a model is run at a size it was not trained at. That is the dangerous one:
    the architecture is otherwise identical, so it loads, runs, and emits
    confident nonsense. `nfft_of` reads the size back out of the weights so it
    can be checked instead of assumed.

Most of this needs no PyTorch, deliberately - `framing.py` is the torch-free
half of the inference path.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import framing as F  # noqa: E402

try:
    import torch
    HAS_TORCH = True
except Exception:  # noqa: BLE001
    torch = None
    HAS_TORCH = False


# --------------------------------------------------------------- the default

def test_default_framing_is_the_shipped_one():
    """Every checkpoint in checkpoints/ was trained at 512/256. If an unset
    environment ever stops meaning that, all of them silently become wrong."""
    assert (F.N_FFT, F.HOP, F.WIN) == (512, 256, 512)
    assert F.SR == 16000
    assert F.CONV_CACHE == (2, 1, 16, 16, 33)
    assert F.TRA_CACHE == (2, 3, 1, 1, 16)
    assert F.INTER_CACHE == (2, 1, 33, 16)
    assert F.erb_subbands() == (65, 64)
    assert F.n_bands() == 129 and F.freq_width() == 33
    assert len(F.np_window()) == 512


def test_pytest_process_has_no_framing_override():
    """A stray SIH_NFFT in the environment would make every other test in this
    suite measure something other than the shipped model."""
    assert not os.environ.get("SIH_NFFT"), "unset SIH_NFFT before running tests"


# ------------------------------------------------------------ band arithmetic

def _deconv(n: int) -> int:
    """One decoder ConvBlock: ConvTranspose2d kernel (1,5) stride (1,2) pad (0,2)."""
    return (n - 1) * 2 - 2 * 2 + 5


@pytest.mark.parametrize("nfft", sorted(F.ERB_SUBBANDS))
def test_band_axis_round_trips(nfft):
    """The encoder halves the band axis twice and the decoder doubles it twice.
    If those do not land on the same number the model does not build - or
    worse, builds and misaligns the mask against the spectrum."""
    erb1, erb2 = F.ERB_SUBBANDS[nfft]
    assert erb1 + erb2 == F.n_bands(nfft)
    assert erb1 <= nfft // 2 + 1, "linear band cannot exceed the spectrum"
    f = F.freq_width(nfft)
    assert _deconv(_deconv(f)) == F.n_bands(nfft)


@pytest.mark.parametrize("nfft", sorted(F.ERB_SUBBANDS))
def test_crossover_frequency_stays_near_2_khz(nfft):
    """erb_subband_1/nfft*fs is where ERB compression starts. It must not drift
    down into the formants when the window shortens."""
    erb1, _ = F.ERB_SUBBANDS[nfft]
    hz = erb1 / nfft * F.SR
    assert 1900 <= hz <= 2200, f"n_fft {nfft} crosses over at {hz:.0f} Hz"


@pytest.mark.parametrize("nfft", sorted(F.ERB_SUBBANDS))
def test_cache_shapes_follow_size_and_width(nfft):
    f = F.freq_width(nfft)
    conv, tra, inter = F.cache_shapes(32, nfft)
    assert conv == (2, 1, 32, 16, f)
    assert tra == (2, 3, 1, 1, 32)
    assert inter == (2, 1, f, 32)


# -------------------------------------------------------------------- opt-in

def test_env_override_changes_everything_downstream():
    """A shorter window has to reach the caches and the latency too, not just
    the FFT call - the caches are what the hardware team builds against."""
    env = {**os.environ, "SIH_NFFT": "320"}
    r = subprocess.run(
        [sys.executable, "-c",
         "import src.framing as F; print(F.N_FFT, F.HOP, F.WIN, F.freq_width(),"
         " F.CONV_CACHE, F.INTER_CACHE)"],
        cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = r.stdout.strip()
    assert out.startswith("320 160 320 21 ")
    assert "(2, 1, 16, 16, 21)" in out and "(2, 1, 21, 16)" in out


def test_unknown_size_is_refused_not_guessed():
    env = {**os.environ, "SIH_NFFT": "400"}
    r = subprocess.run([sys.executable, "-c", "import src.framing"],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode != 0
    assert "ERB_SUBBANDS" in r.stderr


# -------------------------------------------------------------------- models

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch unavailable")
@pytest.mark.parametrize("nfft", sorted(F.ERB_SUBBANDS))
def test_nfft_is_recoverable_from_the_weights(nfft):
    from src.models.gtcrn_wide import make_gtcrn, nfft_of, width_of

    state = make_gtcrn(16, nfft).state_dict()
    assert nfft_of(state) == nfft
    assert width_of(state) == 16
    # The ERB filter bank must actually cover the spectrum: an empty row is a
    # band that can never be heard, and nothing would raise on it.
    erb1, erb2 = F.ERB_SUBBANDS[nfft]
    w = state["erb.erb_fc.weight"]
    assert tuple(w.shape) == (erb2, nfft // 2 + 1 - erb1)
    assert bool((w.abs().sum(dim=1) > 0).all()), "ERB filter bank has an empty band"


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch unavailable")
@pytest.mark.parametrize("nfft", [320, 512])
def test_forward_shape_matches_its_spectrum(nfft):
    from src.models.gtcrn_wide import make_gtcrn

    m = make_gtcrn(32, nfft).eval()
    spec = torch.randn(1, nfft // 2 + 1, 25, 2) * 0.1
    with torch.no_grad():
        out = m(spec)
    assert out.shape == spec.shape


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch unavailable")
def test_short_window_streaming_matches_offline():
    """The same guarantee the width wrapper carries: frame-by-frame streaming
    must equal the offline model, or the exported ONNX is subtly wrong while
    appearing to work."""
    from src.models.gtcrn_wide import (cache_shapes, make_gtcrn,
                                       make_stream_gtcrn)

    nfft, width = 320, 32
    torch.manual_seed(0)
    offline = make_gtcrn(width, nfft).eval()
    for mod in offline.modules():        # defaults make this trivially easy
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.uniform_(-0.1, 0.1)
            mod.running_var.uniform_(0.5, 1.5)
    stream = make_stream_gtcrn(width, nfft).eval()
    from modules.convert import convert_to_stream  # noqa: PLC0415
    convert_to_stream(stream, offline)

    spec = torch.randn(1, nfft // 2 + 1, 40, 2) * 0.1
    with torch.no_grad():
        ref = offline(spec)
        conv, tra, inter = (torch.zeros(*s) for s in cache_shapes(width, nfft))
        outs = []
        for t in range(spec.shape[2]):
            o, conv, tra, inter = stream(spec[:, :, t:t + 1], conv, tra, inter)
            outs.append(o)
    got = torch.cat(outs, dim=2)
    err = (got - ref).abs().max().item()
    assert err < 1e-4 * max(ref.abs().max().item(), 1e-6) + 1e-5, f"max err {err:.3e}"
