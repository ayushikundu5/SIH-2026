"""The width-configurable GTCRN wrapper must be a strict generalisation.

Two things would fail silently if it were not:
  - at width 16 it must BE upstream GTCRN, or the pretrained weights and every
    existing checkpoint quietly mean something different;
  - at any other width the streaming model must reproduce the offline one frame
    by frame, or the shipped ONNX sounds subtly worse while appearing to work
    (CLAUDE.md invariant: verify the streaming export against the offline model).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

try:
    import torch
    HAS_TORCH = True
except Exception:  # noqa: BLE001
    torch = None
    HAS_TORCH = False
pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch unavailable")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DNS3 = ROOT / "checkpoints" / "pretrained" / "model_trained_on_dns3.tar"


def _state(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj.get("model", obj.get("state_dict", obj))


def test_width_16_is_upstream_exactly():
    from src.models.gtcrn import GTCRN
    from src.models.gtcrn_wide import GTCRNWide, width_of

    up, wide = GTCRN().eval(), GTCRNWide(16).eval()
    assert list(up.state_dict()) == list(wide.state_dict())
    for k, v in up.state_dict().items():
        assert wide.state_dict()[k].shape == v.shape, k

    state = _state(DNS3)
    assert width_of(state) == 16
    up.load_state_dict(state)
    wide.load_state_dict(state)
    from src import stft as S
    x = torch.randn(16000)
    with torch.no_grad():
        a, b = up(S.stft(x)[None]), wide(S.stft(x)[None])
    assert torch.equal(a, b)


@pytest.mark.parametrize("width", [32, 48])
def test_wide_model_trains_and_reports_width(width):
    from src.models.gtcrn_wide import make_gtcrn, width_of
    from src import stft as S

    m = make_gtcrn(width)
    assert width_of(m.state_dict()) == width
    n16 = sum(p.numel() for p in make_gtcrn(16).parameters())
    assert sum(p.numel() for p in m.parameters()) > 2 * n16
    x = torch.randn(2, 16000)
    out = m(S.stft(x))
    out.abs().mean().backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


@pytest.mark.parametrize("width", [32])
def test_wide_streaming_matches_offline(width):
    """Frame-by-frame streaming model vs whole-utterance offline model."""
    from src import stft as S
    from src.models.gtcrn_wide import cache_shapes, make_gtcrn, make_stream_gtcrn

    torch.manual_seed(0)
    offline = make_gtcrn(width).eval()
    # BatchNorm running stats at their defaults make the test trivially
    # easy; randomise them so the cache wiring is actually exercised.
    for mod in offline.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.uniform_(-0.1, 0.1)
            mod.running_var.uniform_(0.5, 1.5)
    stream = make_stream_gtcrn(width).eval()     # puts the upstream stream dir on sys.path
    from modules.convert import convert_to_stream  # noqa: PLC0415
    convert_to_stream(stream, offline)

    x = torch.randn(16000) * 0.1
    spec = S.stft(x)[None]                            # (1, 257, T, 2)
    with torch.no_grad():
        ref = offline(spec)
        conv, tra, inter = (torch.zeros(*s) for s in cache_shapes(width))
        outs = []
        for t in range(spec.shape[2]):
            o, conv, tra, inter = stream(spec[:, :, t:t + 1], conv, tra, inter)
            outs.append(o)
    got = torch.cat(outs, dim=2)
    err = (got - ref).abs().max().item()
    assert err < 1e-4 * max(ref.abs().max().item(), 1e-6) + 1e-5, f"max err {err:.3e}"
