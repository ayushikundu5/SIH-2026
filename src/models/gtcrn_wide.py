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
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn

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


def cache_shapes(width: int = DEFAULT_WIDTH) -> tuple[tuple, tuple, tuple]:
    """(conv, tra, inter) streaming cache shapes for a model of this width.

    conv:  (enc/dec, B, C, sum of (kT-1)*dilation = 2*(1+2+5), F=33)
    tra:   (enc/dec, 3 GTConv blocks, 1, B, GRU hidden = C)
    inter: (2 DPGRNNs, 1, B*F = 33, hidden = C)
    At width 16 these equal the constants in `src/framing.py`.
    """
    w = _check(width)
    return (2, 1, w, 16, 33), (2, 3, 1, 1, w), (2, 1, 33, w)


def width_of(state_dict: dict) -> int:
    """Recover the width a checkpoint was trained at from its weight shapes -
    no separate metadata to go missing or disagree with the weights."""
    return int(state_dict["encoder.en_convs.0.conv.weight"].shape[0])


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
    def __init__(self, width: int = DEFAULT_WIDTH):
        nn.Module.__init__(self)
        w = _check(width)
        self.width = w
        self.erb = ERB(65, 64)
        self.sfe = SFE(3, 1)
        self.encoder = WideEncoder(w)
        self.dpgrnn1 = DPGRNN(w, 33, w)
        self.dpgrnn2 = DPGRNN(w, 33, w)
        self.decoder = WideDecoder(w)
        self.mask = Mask()


def make_gtcrn(width: int = DEFAULT_WIDTH) -> nn.Module:
    """Upstream GTCRN at the default width, the wide wrapper otherwise."""
    return GTCRN() if _check(width) == DEFAULT_WIDTH else GTCRNWide(width)


# -------------------------------------------------------------- streaming

def make_stream_gtcrn(width: int = DEFAULT_WIDTH) -> nn.Module:
    """Streaming counterpart, for ONNX export. Imports the upstream streaming
    code lazily: it resolves `modules.*` relative to its own directory."""
    if str(_STREAM_DIR) not in sys.path:
        sys.path.insert(0, str(_STREAM_DIR))
    import gtcrn_stream as up  # noqa: PLC0415

    w = _check(width)
    if w == DEFAULT_WIDTH:
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
            self.erb = up.ERB(65, 64)
            self.sfe = up.SFE(3, 1)
            self.encoder = _Enc()
            self.dpgrnn1 = up.DPGRNN(w, 33, w)
            self.dpgrnn2 = up.DPGRNN(w, 33, w)
            self.decoder = _Dec()
            self.mask = up.Mask()

    return _Stream()
