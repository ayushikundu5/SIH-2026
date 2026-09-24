# Results - onnx

- generated: 2026-09-24T13:06:55+00:00
- git: `not-a-repo`
- test set: `C:\SIH26052_data\testset` (frozen, seed 20260827)
- device: `cpu`
- PESQ available: True
- clips: 720 per method
- eval workers: 11 (RTF column is under parallel load - see results/bench.json)

Targets: PESQ > 2.5 (scored on WIDEBAND PESQ; narrowband shown for reference), STOI > 0.85, SNR > 15.0 dB (scored both as OUTPUT SNR and as SNR GAIN), RTF < 0.5.

## Burst-local performance (impulsive categories)

SI-SDR gain measured INSIDE the gunfire/explosion bursts only. A whole-clip score is diluted by the ~88% of each clip containing no transient, so a model that removes none of the gunfire can still look acceptable overall. This column is the one that answers the question.

| category | method | burst SI-SDR gain | whole-clip SI-SDR gain | burst % of clip |
|---|---|---|---|---|
| artillery | `onnx:artifacts/model_fresh32_simple.onnx` | **+8.83** | +9.14 | 20.3% |
| artillery | `onnx:artifacts/model_combat32_simple.onnx` | **+8.78** | +9.14 | 20.3% |
| artillery | `onnx:artifacts/model_wide32_simple.onnx` | **+8.44** | +8.66 | 20.3% |
| artillery | `onnx:artifacts/model_simple.onnx` | **+7.90** | +8.07 | 20.3% |
| artillery | `onnx:artifacts/model_short24_simple.onnx` | **+7.59** | +8.23 | 20.3% |
| artillery | `onnx:artifacts/model_lowsnr_simple.onnx` | **+7.56** | +7.41 | 20.3% |
| artillery | `onnx:artifacts/gtcrn_dns3_simple.onnx` | **+7.03** | +7.28 | 20.3% |
| artillery | `wiener` | **+0.16** | +0.32 | 20.3% |
| artillery | `unprocessed` | **+0.00** | +0.00 | 20.3% |
| gunshot | `onnx:artifacts/model_fresh32_simple.onnx` | **+8.49** | +8.24 | 18.7% |
| gunshot | `onnx:artifacts/model_combat32_simple.onnx` | **+8.39** | +8.25 | 18.7% |
| gunshot | `onnx:artifacts/model_wide32_simple.onnx` | **+8.12** | +7.87 | 18.7% |
| gunshot | `onnx:artifacts/model_short24_simple.onnx` | **+7.43** | +7.14 | 18.7% |
| gunshot | `onnx:artifacts/model_simple.onnx` | **+7.16** | +6.98 | 18.7% |
| gunshot | `onnx:artifacts/model_lowsnr_simple.onnx` | **+7.15** | +6.58 | 18.7% |
| gunshot | `onnx:artifacts/gtcrn_dns3_simple.onnx` | **+5.83** | +5.74 | 18.7% |
| gunshot | `wiener` | **+0.19** | +0.48 | 18.7% |
| gunshot | `unprocessed` | **+0.00** | +0.00 | 18.7% |

### artillery

| method | PESQ-WB | PESQ-NB | STOI | SI-SDR gain | output SNR | SNR gain | RTF | meets targets |
|---|---|---|---|---|---|---|---|---|
| `onnx:artifacts/gtcrn_dns3_simple.onnx` | 1.631 | 2.240 | 0.832 | +7.28 | +8.68 | +8.14 | 0.4685 | PESQFAIL STOIFAIL SNRoutFAIL SNRgainFAIL RTFPASS |
| `onnx:artifacts/model_combat32_simple.onnx` | 1.856 | 2.469 | 0.853 | +9.14 | +10.29 | +9.74 | 0.7011 | PESQFAIL STOIPASS SNRoutFAIL SNRgainFAIL RTFFAIL |
| `onnx:artifacts/model_fresh32_simple.onnx` | 1.882 | 2.495 | 0.858 | +9.14 | +10.31 | +9.77 | 0.7720 | PESQFAIL STOIPASS SNRoutFAIL SNRgainFAIL RTFFAIL |
| `onnx:artifacts/model_lowsnr_simple.onnx` | 1.665 | 2.250 | 0.826 | +7.41 | +8.92 | +8.38 | 0.4909 | PESQFAIL STOIFAIL SNRoutFAIL SNRgainFAIL RTFPASS |
| `onnx:artifacts/model_short24_simple.onnx` | 1.724 | 2.300 | 0.840 | +8.23 | +9.40 | +8.86 | 0.6498 | PESQFAIL STOIFAIL SNRoutFAIL SNRgainFAIL RTFFAIL |
| `onnx:artifacts/model_simple.onnx` | 1.725 | 2.311 | 0.836 | +8.07 | +9.33 | +8.79 | 0.4949 | PESQFAIL STOIFAIL SNRoutFAIL SNRgainFAIL RTFPASS |
| `onnx:artifacts/model_wide32_simple.onnx` | 1.807 | 2.407 | 0.849 | +8.66 | +9.87 | +9.32 | 0.6938 | PESQFAIL STOIFAIL SNRoutFAIL SNRgainFAIL RTFFAIL |
| `unprocessed` | 1.188 | 1.645 | 0.765 | +0.00 | +0.54 | +0.00 | 0.0000 | PESQFAIL STOIFAIL SNRoutFAIL SNRgainFAIL RTFPASS |
| `wiener` | 1.205 | 1.685 | 0.764 | +0.32 | +1.07 | +0.53 | 0.0094 | PESQFAIL STOIFAIL SNRoutFAIL SNRgainFAIL RTFPASS |

### babble

| method | PESQ-WB | PESQ-NB | STOI | SI-SDR gain | output SNR | SNR gain | RTF | meets targets |
|---|---|---|---|---|---|---|---|---|
| `onnx:artifacts/gtcrn_dns3_simple.onnx` | 1.699 | 2.226 | 0.818 | +3.86 | +9.45 | +4.78 | 0.5792 | PESQFAIL STOIFAIL SNRoutFAIL SNRgainFAIL RTFFAIL |
| `onnx:artifacts/model_combat32_simple.onnx` | 1.996 | 2.513 | 0.851 | +5.66 | +10.97 | +6.30 | 0.6812 | PESQFAIL STOIPASS SNRoutFAIL SNRgainFAIL RTFFAIL |
| `onnx:artifacts/model_fresh32_simple.onnx` | 1.996 | 2.525 | 0.851 | +5.65 | +11.00 | +6.33 | 0.7059 | PESQFAIL STOIPASS SNRoutFAIL SNRgainFAIL RTFFAIL |
| `onnx:artifacts/model_lowsnr_simple.onnx` | 1.751 | 2.255 | 0.814 | +3.30 | +9.33 | +4.65 | 0.4569 | PESQFAIL STOIFAIL SNRoutFAIL SNRgainFAIL RTFPASS |
| `onnx:artifacts/model_short24_simple.onnx` | 1.872 | 2.369 | 0.836 | +4.79 | +10.10 | +5.43 | 0.7068 | PESQFAIL STOIFAIL SNRoutFAIL SNRgainFAIL RTFFAIL |
| `onnx:artifacts/model_simple.onnx` | 1.829 | 2.337 | 0.827 | +4.55 | +10.12 | +5.44 | 0.4897 | PESQFAIL STOIFAIL SNRoutFAIL SNRgainFAIL RTFPASS |
| `onnx:artifacts/model_wide32_simple.onnx` | 1.945 | 2.456 | 0.845 | +5.18 | +10.68 | +6.01 | 0.6848 | PESQFAIL STOIFAIL SNRoutFAIL SNRgainFAIL RTFFAIL |
| `unprocessed` | 1.333 | 1.805 | 0.785 | +0.00 | +4.67 | +0.00 | 0.0000 | PESQFAIL STOIFAIL SNRoutFAIL SNRgainFAIL RTFPASS |
| `wiener` | 1.374 | 1.849 | 0.783 | +0.72 | +5.61 | +0.94 | 0.0179 | PESQFAIL STOIFAIL SNRoutFAIL SNRgainFAIL RTFPASS |

### engine

| method | PESQ-WB | PESQ-NB | STOI | SI-SDR gain | output SNR | SNR gain | RTF | meets targets |
|---|---|---|---|---|---|---|---|---|
| `onnx:artifacts/gtcrn_dns3_simple.onnx` | 1.963 | 2.555 | 0.870 | +4.82 | +10.93 | +5.34 | 0.5470 | PESQFAIL STOIPASS SNRoutFAIL SNRgainFAIL RTFFAIL |
| `onnx:artifacts/model_combat32_simple.onnx` | 2.269 | 2.821 | 0.889 | +6.43 | +12.38 | +6.79 | 0.6860 | PESQFAIL STOIPASS SNRoutFAIL SNRgainFAIL RTFFAIL |
| `onnx:artifacts/model_fresh32_simple.onnx` | 2.279 | 2.844 | 0.892 | +6.58 | +12.52 | +6.93 | 0.7482 | PESQFAIL STOIPASS SNRoutFAIL SNRgainFAIL RTFFAIL |
| `onnx:artifacts/model_lowsnr_simple.onnx` | 1.991 | 2.578 | 0.867 | +4.66 | +10.79 | +5.20 | 0.4578 | PESQFAIL STOIPASS SNRoutFAIL SNRgainFAIL RTFPASS |
| `onnx:artifacts/model_short24_simple.onnx` | 2.081 | 2.635 | 0.874 | +5.46 | +11.30 | +5.70 | 0.6284 | PESQFAIL STOIPASS SNRoutFAIL SNRgainFAIL RTFFAIL |
| `onnx:artifacts/model_simple.onnx` | 2.061 | 2.636 | 0.874 | +5.38 | +11.40 | +5.81 | 0.4948 | PESQFAIL STOIPASS SNRoutFAIL SNRgainFAIL RTFPASS |
| `onnx:artifacts/model_wide32_simple.onnx` | 2.206 | 2.777 | 0.887 | +6.23 | +12.22 | +6.62 | 0.7152 | PESQFAIL STOIPASS SNRoutFAIL SNRgainFAIL RTFFAIL |
| `unprocessed` | 1.380 | 1.918 | 0.816 | +0.00 | +5.59 | +0.00 | 0.0000 | PESQFAIL STOIFAIL SNRoutFAIL SNRgainFAIL RTFPASS |
| `wiener` | 1.422 | 2.000 | 0.819 | +1.09 | +6.88 | +1.29 | 0.0102 | PESQFAIL STOIFAIL SNRoutFAIL SNRgainFAIL RTFPASS |

### gunshot

| method | PESQ-WB | PESQ-NB | STOI | SI-SDR gain | output SNR | SNR gain | RTF | meets targets |
|---|---|---|---|---|---|---|---|---|
| `onnx:artifacts/gtcrn_dns3_simple.onnx` | 1.796 | 2.369 | 0.849 | +5.74 | +9.26 | +6.43 | 0.4837 | PESQFAIL STOIFAIL SNRoutFAIL SNRgainFAIL RTFPASS |
| `onnx:artifacts/model_combat32_simple.onnx` | 2.092 | 2.654 | 0.871 | +8.25 | +11.51 | +8.68 | 0.7162 | PESQFAIL STOIPASS SNRoutFAIL SNRgainFAIL RTFFAIL |
| `onnx:artifacts/model_fresh32_simple.onnx` | 2.118 | 2.679 | 0.873 | +8.24 | +11.53 | +8.69 | 0.8051 | PESQFAIL STOIPASS SNRoutFAIL SNRgainFAIL RTFFAIL |
| `onnx:artifacts/model_lowsnr_simple.onnx` | 1.840 | 2.403 | 0.847 | +6.58 | +10.10 | +7.27 | 0.4980 | PESQFAIL STOIFAIL SNRoutFAIL SNRgainFAIL RTFPASS |
| `onnx:artifacts/model_short24_simple.onnx` | 1.936 | 2.490 | 0.855 | +7.14 | +10.44 | +7.60 | 0.8048 | PESQFAIL STOIPASS SNRoutFAIL SNRgainFAIL RTFFAIL |
| `onnx:artifacts/model_simple.onnx` | 1.903 | 2.443 | 0.854 | +6.98 | +10.43 | +7.60 | 0.5083 | PESQFAIL STOIPASS SNRoutFAIL SNRgainFAIL RTFFAIL |
| `onnx:artifacts/model_wide32_simple.onnx` | 2.052 | 2.598 | 0.866 | +7.87 | +11.19 | +8.36 | 0.7119 | PESQFAIL STOIPASS SNRoutFAIL SNRgainFAIL RTFFAIL |
| `unprocessed` | 1.297 | 1.767 | 0.782 | +0.00 | +2.83 | +0.00 | 0.0000 | PESQFAIL STOIFAIL SNRoutFAIL SNRgainFAIL RTFPASS |
| `wiener` | 1.316 | 1.790 | 0.781 | +0.48 | +3.49 | +0.66 | 0.0070 | PESQFAIL STOIFAIL SNRoutFAIL SNRgainFAIL RTFPASS |

### rotor

| method | PESQ-WB | PESQ-NB | STOI | SI-SDR gain | output SNR | SNR gain | RTF | meets targets |
|---|---|---|---|---|---|---|---|---|
| `onnx:artifacts/gtcrn_dns3_simple.onnx` | 1.865 | 2.497 | 0.869 | +4.03 | +10.58 | +4.47 | 0.4851 | PESQFAIL STOIPASS SNRoutFAIL SNRgainFAIL RTFPASS |
| `onnx:artifacts/model_combat32_simple.onnx` | 2.173 | 2.780 | 0.890 | +5.73 | +12.18 | +6.07 | 0.6797 | PESQFAIL STOIPASS SNRoutFAIL SNRgainFAIL RTFFAIL |
| `onnx:artifacts/model_fresh32_simple.onnx` | 2.189 | 2.793 | 0.891 | +5.73 | +12.17 | +6.06 | 0.7840 | PESQFAIL STOIPASS SNRoutFAIL SNRgainFAIL RTFFAIL |
| `onnx:artifacts/model_lowsnr_simple.onnx` | 1.891 | 2.497 | 0.868 | +3.97 | +10.56 | +4.44 | 0.5090 | PESQFAIL STOIPASS SNRoutFAIL SNRgainFAIL RTFFAIL |
| `onnx:artifacts/model_short24_simple.onnx` | 2.005 | 2.603 | 0.876 | +4.84 | +11.21 | +5.10 | 0.6281 | PESQFAIL STOIPASS SNRoutFAIL SNRgainFAIL RTFFAIL |
| `onnx:artifacts/model_simple.onnx` | 1.992 | 2.591 | 0.875 | +4.74 | +11.22 | +5.11 | 0.4925 | PESQFAIL STOIPASS SNRoutFAIL SNRgainFAIL RTFPASS |
| `onnx:artifacts/model_wide32_simple.onnx` | 2.139 | 2.723 | 0.888 | +5.53 | +12.02 | +5.90 | 0.7024 | PESQFAIL STOIPASS SNRoutFAIL SNRgainFAIL RTFFAIL |
| `unprocessed` | 1.354 | 1.896 | 0.823 | +0.00 | +6.11 | +0.00 | 0.0000 | PESQFAIL STOIFAIL SNRoutFAIL SNRgainFAIL RTFPASS |
| `wiener` | 1.404 | 1.975 | 0.824 | +0.74 | +7.01 | +0.90 | 0.0088 | PESQFAIL STOIFAIL SNRoutFAIL SNRgainFAIL RTFPASS |

### siren

| method | PESQ-WB | PESQ-NB | STOI | SI-SDR gain | output SNR | SNR gain | RTF | meets targets |
|---|---|---|---|---|---|---|---|---|
| `onnx:artifacts/gtcrn_dns3_simple.onnx` | 1.930 | 2.537 | 0.885 | +4.44 | +11.50 | +4.80 | 0.6037 | PESQFAIL STOIPASS SNRoutFAIL SNRgainFAIL RTFFAIL |
| `onnx:artifacts/model_combat32_simple.onnx` | 2.251 | 2.849 | 0.905 | +6.10 | +13.00 | +6.31 | 0.6972 | PESQFAIL STOIPASS SNRoutFAIL SNRgainFAIL RTFFAIL |
| `onnx:artifacts/model_fresh32_simple.onnx` | 2.283 | 2.879 | 0.908 | +6.26 | +13.15 | +6.46 | 0.7456 | PESQFAIL STOIPASS SNRoutFAIL SNRgainFAIL RTFFAIL |
| `onnx:artifacts/model_lowsnr_simple.onnx` | 1.985 | 2.572 | 0.884 | +4.26 | +11.41 | +4.71 | 0.4351 | PESQFAIL STOIPASS SNRoutFAIL SNRgainFAIL RTFPASS |
| `onnx:artifacts/model_short24_simple.onnx` | 2.093 | 2.680 | 0.894 | +5.38 | +12.15 | +5.46 | 0.6475 | PESQFAIL STOIPASS SNRoutFAIL SNRgainFAIL RTFFAIL |
| `onnx:artifacts/model_simple.onnx` | 2.078 | 2.654 | 0.892 | +5.25 | +12.21 | +5.51 | 0.5157 | PESQFAIL STOIPASS SNRoutFAIL SNRgainFAIL RTFFAIL |
| `onnx:artifacts/model_wide32_simple.onnx` | 2.193 | 2.789 | 0.902 | +5.67 | +12.63 | +5.94 | 0.7079 | PESQFAIL STOIPASS SNRoutFAIL SNRgainFAIL RTFFAIL |
| `unprocessed` | 1.364 | 1.888 | 0.837 | +0.00 | +6.70 | +0.00 | 0.0000 | PESQFAIL STOIFAIL SNRoutFAIL SNRgainFAIL RTFPASS |
| `wiener` | 1.408 | 1.937 | 0.837 | +0.58 | +7.43 | +0.73 | 0.0157 | PESQFAIL STOIFAIL SNRoutFAIL SNRgainFAIL RTFPASS |

## All targets by INPUT SNR (all categories pooled)

| method | input SNR | n | PESQ-WB | PESQ-NB | STOI | output SNR | SNR gain |
|---|---|---|---|---|---|---|---|
| `onnx:artifacts/gtcrn_dns3_simple.onnx` | <0 dB | 170 | 1.317 | 1.773 | 0.730 | +4.00 | +8.64 |
| `onnx:artifacts/gtcrn_dns3_simple.onnx` | 0-5 dB | 232 | 1.625 | 2.214 | 0.845 | +8.94 | +6.43 |
| `onnx:artifacts/gtcrn_dns3_simple.onnx` | 5-10 dB | 169 | 1.957 | 2.624 | 0.904 | +12.02 | +4.52 |
| `onnx:artifacts/gtcrn_dns3_simple.onnx` | 10-15 dB | 95 | 2.393 | 3.075 | 0.947 | +15.76 | +3.41 |
| `onnx:artifacts/gtcrn_dns3_simple.onnx` | >15 dB | 54 | 2.726 | 3.338 | 0.958 | +17.89 | +0.48 |
| `onnx:artifacts/model_combat32_simple.onnx` | <0 dB | 170 | 1.477 | 1.967 | 0.762 | +6.32 | +10.97 |
| `onnx:artifacts/model_combat32_simple.onnx` | 0-5 dB | 232 | 1.890 | 2.496 | 0.871 | +10.29 | +7.78 |
| `onnx:artifacts/model_combat32_simple.onnx` | 5-10 dB | 169 | 2.289 | 2.946 | 0.923 | +13.26 | +5.76 |
| `onnx:artifacts/model_combat32_simple.onnx` | 10-15 dB | 95 | 2.826 | 3.394 | 0.959 | +17.35 | +5.00 |
| `onnx:artifacts/model_combat32_simple.onnx` | >15 dB | 54 | 3.183 | 3.638 | 0.971 | +20.18 | +2.77 |
| `onnx:artifacts/model_fresh32_simple.onnx` | <0 dB | 170 | 1.488 | 1.994 | 0.766 | +6.33 | +10.97 |
| `onnx:artifacts/model_fresh32_simple.onnx` | 0-5 dB | 232 | 1.916 | 2.523 | 0.873 | +10.37 | +7.86 |
| `onnx:artifacts/model_fresh32_simple.onnx` | 5-10 dB | 169 | 2.321 | 2.959 | 0.925 | +13.27 | +5.77 |
| `onnx:artifacts/model_fresh32_simple.onnx` | 10-15 dB | 95 | 2.829 | 3.418 | 0.960 | +17.41 | +5.06 |
| `onnx:artifacts/model_fresh32_simple.onnx` | >15 dB | 54 | 3.167 | 3.642 | 0.973 | +20.41 | +3.00 |
| `onnx:artifacts/model_lowsnr_simple.onnx` | <0 dB | 170 | 1.366 | 1.812 | 0.724 | +5.28 | +9.92 |
| `onnx:artifacts/model_lowsnr_simple.onnx` | 0-5 dB | 232 | 1.672 | 2.248 | 0.842 | +8.88 | +6.37 |
| `onnx:artifacts/model_lowsnr_simple.onnx` | 5-10 dB | 169 | 1.987 | 2.632 | 0.903 | +11.60 | +4.10 |
| `onnx:artifacts/model_lowsnr_simple.onnx` | 10-15 dB | 95 | 2.430 | 3.079 | 0.947 | +15.49 | +3.14 |
| `onnx:artifacts/model_lowsnr_simple.onnx` | >15 dB | 54 | 2.738 | 3.331 | 0.956 | +17.47 | +0.06 |
| `onnx:artifacts/model_short24_simple.onnx` | <0 dB | 170 | 1.377 | 1.809 | 0.738 | +5.60 | +10.24 |
| `onnx:artifacts/model_short24_simple.onnx` | 0-5 dB | 232 | 1.717 | 2.291 | 0.854 | +9.30 | +6.79 |
| `onnx:artifacts/model_short24_simple.onnx` | 5-10 dB | 169 | 2.100 | 2.757 | 0.913 | +12.19 | +4.69 |
| `onnx:artifacts/model_short24_simple.onnx` | 10-15 dB | 95 | 2.655 | 3.280 | 0.955 | +16.29 | +3.94 |
| `onnx:artifacts/model_short24_simple.onnx` | >15 dB | 54 | 3.071 | 3.568 | 0.969 | +19.17 | +1.76 |
| `onnx:artifacts/model_simple.onnx` | <0 dB | 170 | 1.379 | 1.832 | 0.733 | +5.41 | +10.06 |
| `onnx:artifacts/model_simple.onnx` | 0-5 dB | 232 | 1.721 | 2.301 | 0.852 | +9.43 | +6.92 |
| `onnx:artifacts/model_simple.onnx` | 5-10 dB | 169 | 2.097 | 2.731 | 0.911 | +12.31 | +4.81 |
| `onnx:artifacts/model_simple.onnx` | 10-15 dB | 95 | 2.579 | 3.191 | 0.953 | +16.41 | +4.06 |
| `onnx:artifacts/model_simple.onnx` | >15 dB | 54 | 2.914 | 3.458 | 0.964 | +18.85 | +1.44 |
| `onnx:artifacts/model_wide32_simple.onnx` | <0 dB | 170 | 1.428 | 1.901 | 0.754 | +5.60 | +10.24 |
| `onnx:artifacts/model_wide32_simple.onnx` | 0-5 dB | 232 | 1.846 | 2.439 | 0.867 | +10.12 | +7.61 |
| `onnx:artifacts/model_wide32_simple.onnx` | 5-10 dB | 169 | 2.243 | 2.895 | 0.921 | +13.07 | +5.56 |
| `onnx:artifacts/model_wide32_simple.onnx` | 10-15 dB | 95 | 2.762 | 3.345 | 0.958 | +17.22 | +4.87 |
| `onnx:artifacts/model_wide32_simple.onnx` | >15 dB | 54 | 3.117 | 3.590 | 0.970 | +20.19 | +2.78 |
| `unprocessed` | <0 dB | 170 | 1.098 | 1.415 | 0.653 | -4.64 | +0.00 |
| `unprocessed` | 0-5 dB | 232 | 1.176 | 1.643 | 0.781 | +2.51 | +0.00 |
| `unprocessed` | 5-10 dB | 169 | 1.357 | 1.909 | 0.861 | +7.50 | +0.00 |
| `unprocessed` | 10-15 dB | 95 | 1.583 | 2.268 | 0.925 | +12.35 | +0.00 |
| `unprocessed` | >15 dB | 54 | 2.050 | 2.787 | 0.951 | +17.41 | +0.00 |
| `wiener` | <0 dB | 170 | 1.094 | 1.431 | 0.651 | -3.88 | +0.76 |
| `wiener` | 0-5 dB | 232 | 1.201 | 1.688 | 0.783 | +3.65 | +1.14 |
| `wiener` | 5-10 dB | 169 | 1.392 | 1.967 | 0.861 | +8.22 | +0.72 |
| `wiener` | 10-15 dB | 95 | 1.662 | 2.376 | 0.925 | +13.09 | +0.74 |
| `wiener` | >15 dB | 54 | 2.180 | 2.879 | 0.950 | +17.76 | +0.35 |

## SNR gain by INPUT SNR (dB)

The >15 dB target is only reachable in the low-input-SNR regime; at high input SNR there is little noise left to remove.

| method | <0 dB | 0-5 dB | 5-10 dB | 10-15 dB | >15 dB |
|---|---|---|---|---|---|
| `onnx:artifacts/gtcrn_dns3_simple.onnx` | +8.64 (n=170) | +6.43 (n=232) | +4.52 (n=169) | +3.41 (n=95) | +0.48 (n=54) |
| `onnx:artifacts/model_combat32_simple.onnx` | +10.97 (n=170) | +7.78 (n=232) | +5.76 (n=169) | +5.00 (n=95) | +2.77 (n=54) |
| `onnx:artifacts/model_fresh32_simple.onnx` | +10.97 (n=170) | +7.86 (n=232) | +5.77 (n=169) | +5.06 (n=95) | +3.00 (n=54) |
| `onnx:artifacts/model_lowsnr_simple.onnx` | +9.92 (n=170) | +6.37 (n=232) | +4.10 (n=169) | +3.14 (n=95) | +0.06 (n=54) |
| `onnx:artifacts/model_short24_simple.onnx` | +10.24 (n=170) | +6.79 (n=232) | +4.69 (n=169) | +3.94 (n=95) | +1.76 (n=54) |
| `onnx:artifacts/model_simple.onnx` | +10.06 (n=170) | +6.92 (n=232) | +4.81 (n=169) | +4.06 (n=95) | +1.44 (n=54) |
| `onnx:artifacts/model_wide32_simple.onnx` | +10.24 (n=170) | +7.61 (n=232) | +5.56 (n=169) | +4.87 (n=95) | +2.78 (n=54) |
| `unprocessed` | +0.00 (n=170) | +0.00 (n=232) | +0.00 (n=169) | +0.00 (n=95) | +0.00 (n=54) |
| `wiener` | +0.76 (n=170) | +1.14 (n=232) | +0.72 (n=169) | +0.74 (n=95) | +0.35 (n=54) |