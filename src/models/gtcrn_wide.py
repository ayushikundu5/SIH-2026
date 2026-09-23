"""GTCRN with a configurable channel width - a wrapper, not an edit.

Upstream GTCRN hardcodes 16 channels everywhere (48K parameters). Fine-tuning
that model on defence noise plateaus: val PESQ moved 1.779 -> 1.788 over 60
epochs while training loss kept falling, and three differently-trained 16-wide
models all land at PESQ 1.81-1.93 on the frozen set. That is a capacity
ceiling, and the deployment target (Jetson-class SoC) has room for a model many
times larger.

Every `forward` in upstream GTCRN - offline and streaming - is already
width-agnostic; the width only appears in constructors. So each class here
subclasses its upstream counterpart and overrides `__init__` alone, inheriting
the forward logic verbatim. `gtcrn.py` and `gtcrn_stream.py` stay untouched and
diffable against upstream.

`width=16` builds a model whose state_dict keys and shapes are identical to
upstream's, so every existing checkpoint (and the DNS3/VCTK pretrained weights)
loads unchanged. `make_gtcrn` still returns the upstream class at width 16, so
nothing that worked before goes through new code.

Width must be a multiple of 4 (grouped convs halve it; the grouped GRUs inside
the dual-path RNN halve it again).

The STFT size is the second parameter here, for the same reason: the band axis
(ERB split, the two stride-2 convs, the dual-path RNN width and every streaming
cache) is derived from `n_fft`, and shortening the window is the only way to get
under the 32 ms latency ceiling. It defaults to whatever `src/framing.py` is
configured for - 512 unless SIH_NFFT says otherwise - so an unset environment
builds exactly the shipped architecture.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn

from ..framing import ERB_SUBBANDS, erb_subbands, freq_width
from ..framing import N_FFT as DEFAULT_NFFT
from ..framing import cache_shapes as _frame_cache_shapes
from .gtcrn import (DPGRNN, ERB, GTCRN, SFE, ConvBlock, Decoder, Encoder,
                    GTConvBlock, Mask)

DEFAULT_WIDTH = 16
_ROOT = Path(__file__).resolve().parents[2]
_STREAM_DIR = _ROOT / "third_party" / "gtcrn" / "stream"


def _check(width: int) -> int:
    width = int(width)
    if width < 4 or width % 4:
        raise ValueError(f"GTCRN width must be a positive multiple of 4, got {width}")
    return width


def cache_shapes(width: int = DEFAULT_WIDTH,
                 nfft: int | None = None) -> tuple[tuple, tuple, tuple]:
    """(conv, tra, inter) streaming cache shapes for a model of this size.

    Delegates to `framing.cache_shapes`, which owns the arithmetic so the
    torch-free inference path can state the shapes without importing a model.
    At width 16 / n_fft 512 these equal the constants in `src/framing.py`.
    """
    return _frame_cache_shapes(_check(width), nfft)


def width_of(state_dict: dict) -> int:
    """Recover the width a checkpoint was trained at from its weight shapes -
    no separate metadata to go missing or disagree with the weights."""
    return int(state_dict["encoder.en_convs.0.conv.weight"].shape[0])


def nfft_of(state_dict: dict) -> int:
    """Recover the STFT size the same way, from the ERB filter bank.

    `erb_fc.weight` is (erb_subband_2, n_fft//2 + 1 - erb_subband_1), and the
    pairs in ERB_SUBBANDS give one n_fft each. Reading it from the weights
    means a checkpoint cannot be run at the wrong window size by accident -
    which would load, run, and quietly produce rubbish, since the architecture
    is otherwise identical.
    """
    shape = tuple(state_dict["erb.erb_fc.weight"].shape)
    for nfft, (erb1, erb2) in ERB_SUBBANDS.items():
        if shape == (erb2, nfft // 2 + 1 - erb1):
            return nfft
    raise ValueError(f"erb.erb_fc.weight has shape {shape}, which matches no "
                     f"entry in framing.ERB_SUBBANDS ({sorted(ERB_SUBBANDS)})")


# ---------------------------------------------------------------- offline

class WideEncoder(Encoder):
    def __init__(self, w: int):
        nn.Module.__init__(self)
        self.en_convs = nn.ModuleList([
            ConvBlock(3*3, w, (1,5), stride=(1,2), padding=(0,2), use_deconv=False, is_last=False),
            ConvBlock(w, w, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=False, is_last=False),
            GTConvBlock(w, w, (3,3), stride=(1,1), padding=(0,1), dilation=(1,1), use_deconv=False),
            GTConvBlock(w, w, (3,3), stride=(1,1), padding=(0,1), dilation=(2,1), use_deconv=False),
            GTConvBlock(w, w, (3,3), stride=(1,1), padding=(0,1), dilation=(5,1), use_deconv=False),
        ])


class WideDecoder(Decoder):
    def __init__(self, w: int):
        nn.Module.__init__(self)
        self.de_convs = nn.ModuleList([
            GTConvBlock(w, w, (3,3), stride=(1,1), padding=(2*5,1), dilation=(5,1), use_deconv=True),
            GTConvBlock(w, w, (3,3), stride=(1,1), padding=(2*2,1), dilation=(2,1), use_deconv=True),
            GTConvBlock(w, w, (3,3), stride=(1,1), padding=(2*1,1), dilation=(1,1), use_deconv=True),
            ConvBlock(w, w, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=True, is_last=False),
            ConvBlock(w, 2, (1,5), stride=(1,2), padding=(0,2), use_deconv=True, is_last=True),
        ])


class GTCRNWide(GTCRN):
    def __init__(self, width: int = DEFAULT_WIDTH, nfft: int | None = None):
        nn.Module.__init__(self)
        w = _check(width)
        nfft = DEFAULT_NFFT if nfft is None else int(nfft)
        erb1, erb2 = erb_subbands(nfft)
        f = freq_width(nfft)
        self.width = w
        self.nfft = nfft
        self.erb = ERB(erb1, erb2, nfft=nfft)
        self.sfe = SFE(3, 1)
        self.encoder = WideEncoder(w)
        self.dpgrnn1 = DPGRNN(w, f, w)
        self.dpgrnn2 = DPGRNN(w, f, w)
        self.decoder = WideDecoder(w)
        self.mask = Mask()


def make_gtcrn(width: int = DEFAULT_WIDTH, nfft: int | None = None) -> nn.Module:
    """Upstream GTCRN at the shipped width and window, the wrapper otherwise."""
    nfft = DEFAULT_NFFT if nfft is None else int(nfft)
    if _check(width) == DEFAULT_WIDTH and nfft == 512:
        return GTCRN()
    return GTCRNWide(width, nfft)


# -------------------------------------------------------------- streaming

def make_stream_gtcrn(width: int = DEFAULT_WIDTH, nfft: int | None = None) -> nn.Module:
    """Streaming counterpart, for ONNX export. Imports the upstream streaming
    code lazily: it resolves `modules.*` relative to its own directory."""
    if str(_STREAM_DIR) not in sys.path:
        sys.path.insert(0, str(_STREAM_DIR))
    import gtcrn_stream as up  # noqa: PLC0415

    w = _check(width)
    nfft = DEFAULT_NFFT if nfft is None else int(nfft)
    erb1, erb2 = erb_subbands(nfft)
    f = freq_width(nfft)
    if w == DEFAULT_WIDTH and nfft == 512:
        return up.StreamGTCRN()

    class _Enc(up.StreamEncoder):
        def __init__(self):
            nn.Module.__init__(self)
            self.en_convs = nn.ModuleList([
                up.ConvBlock(3*3, w, (1,5), stride=(1,2), padding=(0,2), use_deconv=False, is_last=False),
                up.ConvBlock(w, w, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=False, is_last=False),
                up.StreamGTConvBlock(w, w, (3,3), stride=(1,1), padding=(0,1), dilation=(1,1), use_deconv=False),
                up.StreamGTConvBlock(w, w, (3,3), stride=(1,1), padding=(0,1), dilation=(2,1), use_deconv=False),
                up.StreamGTConvBlock(w, w, (3,3), stride=(1,1), padding=(0,1), dilation=(5,1), use_deconv=False),
            ])

    class _Dec(up.StreamDecoder):
        def __init__(self):
            nn.Module.__init__(self)
            self.de_convs = nn.ModuleList([
                up.StreamGTConvBlock(w, w, (3,3), stride=(1,1), padding=(0,1), dilation=(5,1), use_deconv=True),
                up.StreamGTConvBlock(w, w, (3,3), stride=(1,1), padding=(0,1), dilation=(2,1), use_deconv=True),
                up.StreamGTConvBlock(w, w, (3,3), stride=(1,1), padding=(0,1), dilation=(1,1), use_deconv=True),
                up.ConvBlock(w, w, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=True, is_last=False),
                up.ConvBlock(w, 2, (1,5), stride=(1,2), padding=(0,2), use_deconv=True, is_last=True),
            ])

    class _Stream(up.StreamGTCRN):
        def __init__(self):
            nn.Module.__init__(self)
            self.width = w
            self.nfft = nfft
            self.erb = up.ERB(erb1, erb2, nfft=nfft)
            self.sfe = up.SFE(3, 1)
            self.encoder = _Enc()
            self.dpgrnn1 = up.DPGRNN(w, f, w)
            self.dpgrnn2 = up.DPGRNN(w, f, w)
            self.decoder = _Dec()
            self.mask = up.Mask()

    return _Stream()
