"""Training loss.

Built on GTCRN's own `HybridLoss` (third_party/gtcrn/loss.py) rather than a
reinvention, so fine-tuning starts from the objective the pretrained weights
were produced under. That loss is:

    30 * (compressed real + imag MSE)  +  70 * (compressed magnitude MSE)  +  SI-SNR

with power-law compression p = 0.3, the DNS-challenge convention (compression
matters because raw magnitudes are dominated by a handful of loud bins, and the
quiet bins are where intelligibility lives).

What we add is the TRANSIENT TERM. The roadmap doc names this as the number-one
risk and prescribes the fix directly:

    Gunshots still get through -> weight the training so sudden sounds count
    more. Do not just train for longer, that will not fix it.

So the same spectral loss is recomputed over only the frames the mixer marked as
containing a burst, and added with weight `w_transient`. Those frames are a
small minority of any clip; without this term they are numerically invisible in
the average and the model optimises the easy stationary background instead.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .stft import HOP, N_FFT, WIN, window

_EPS = 1e-12


def _compressed(spec: torch.Tensor, p: float = 0.3):
    """(B, F, T, 2) -> compressed magnitude and compressed real/imag."""
    re, im = spec[..., 0], spec[..., 1]
    mag = torch.sqrt(re ** 2 + im ** 2 + _EPS)
    comp = mag ** p
    scale = mag ** (1.0 - p)
    return comp, re / scale, im / scale


def _spec_mse(pred, true, mask=None):
    """Compressed-spectrum MSE, optionally over a frame subset.

    `mask` is (B, T) over frames. Reduction is over MASKED ELEMENTS ONLY, so the
    value does not shrink just because few frames are selected - otherwise the
    transient term would silently weaken exactly when bursts are rare.
    """
    p_mag, p_re, p_im = _compressed(pred)
    t_mag, t_re, t_im = _compressed(true)

    if mask is None:
        mag = torch.mean((p_mag - t_mag) ** 2)
        re = torch.mean((p_re - t_re) ** 2)
        im = torch.mean((p_im - t_im) ** 2)
        return mag, re, im

    m = mask[:, None, :].to(pred.dtype)            # (B, 1, T)
    denom = m.sum() * pred.shape[1] + _EPS
    mag = torch.sum(((p_mag - t_mag) ** 2) * m) / denom
    re = torch.sum(((p_re - t_re) ** 2) * m) / denom
    im = torch.sum(((p_im - t_im) ** 2) * m) / denom
    return mag, re, im


def _si_snr_from_spec(pred, true):
    """SI-SNR computed in the time domain after iSTFT - anchors the phase, which
    a magnitude-only loss leaves free to drift."""
    w = window(pred.device, torch.float32)
    y_pred = torch.istft(torch.view_as_complex(pred.contiguous()),
                         N_FFT, HOP, WIN, window=w)
    y_true = torch.istft(torch.view_as_complex(true.contiguous()),
                         N_FFT, HOP, WIN, window=w)
    proj = (torch.sum(y_true * y_pred, dim=-1, keepdim=True) * y_true
            / (torch.sum(y_true ** 2, dim=-1, keepdim=True) + 1e-8))
    noise = y_pred - proj
    ratio = (torch.norm(proj, dim=-1) ** 2) / (torch.norm(noise, dim=-1) ** 2 + 1e-8)
    return -torch.log10(ratio + 1e-8).mean()


class TransientWeightedLoss(nn.Module):
    """GTCRN HybridLoss + a burst-frame term.

    Set `w_transient=0` to recover the upstream objective exactly, which is how
    the ablation in the results table is produced.
    """

    def __init__(self, w_mag: float = 70.0, w_ri: float = 30.0,
                 w_sisnr: float = 1.0, w_transient: float = 2.5):
        super().__init__()
        self.w_mag = w_mag
        self.w_ri = w_ri
        self.w_sisnr = w_sisnr
        self.w_transient = w_transient

    def forward(self, pred_stft: torch.Tensor, true_stft: torch.Tensor,
                transient_mask: torch.Tensor | None = None):
        mag, re, im = _spec_mse(pred_stft, true_stft)
        sisnr = _si_snr_from_spec(pred_stft, true_stft)
        base = self.w_ri * (re + im) + self.w_mag * mag + self.w_sisnr * sisnr

        parts = {"mag": mag.detach(), "ri": (re + im).detach(),
                 "sisnr": sisnr.detach(),
                 "transient": torch.zeros((), device=pred_stft.device)}

        total = base
        if (self.w_transient > 0 and transient_mask is not None
                and transient_mask.any()):
            t_mag, t_re, t_im = _spec_mse(pred_stft, true_stft, transient_mask)
            t_loss = self.w_ri * (t_re + t_im) + self.w_mag * t_mag
            total = total + self.w_transient * t_loss
            parts["transient"] = t_loss.detach()

        parts["total"] = total.detach()
        return total, parts
