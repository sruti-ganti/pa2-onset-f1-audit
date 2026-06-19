# Rhythm2Drums (PA2 Extension): Onset-F1 Evaluation

## Motivation
The onset-F1 metrics reported for Model 2 (rhythm conditioned generation) were dramatically lower than the TRIA baseline — e.g. F1_Kick@30ms of 0.10 vs. TRIA's 0.52. Instead of assuming that the model itself was simply worse, this extension assesses the evaluation pipeline (`eval_onset_f1.py`) to check whether the metric was truly measuring what it claimed to measure.

Analysis of the evaluation methodology uncovered two independent, quantified defects — expanded on below. A third defect was identified but could not be empirically validated (please refer to Limitations section). No model retraining or regeneration was involved. Both empirically validated findings are derived entirely from analysis of the AVP ground-truth dataset and the existing evaluation code.

Reproduce:
```
# Download the public AVP dataset (Delgado et al., 2019; CC-BY-4.0)
curl -L -C - -o data/AVP_Dataset.zip "https://zenodo.org/records/3245959/files/AVP_Dataset.zip?download=1"
unzip data/AVP_Dataset.zip -d data/AVP_Dataset

# Finding 2 — label classification audit (no audio needed)
python audit_label_bug.py --avp-dir data/AVP_Dataset/Fixed

# Finding 1 — band-split mismatch audit (audio only, no model)
python audit_band_split.py --avp-dir data/AVP_Dataset/Fixed --plot
```

## Result 1: Band-split mismatch
`eval_onset_f1.py` transcribes generated audio using fixed frequency bands: kick as a low-pass filter below 200 Hz, snare as a band-pass filter ranging from 200-5000 Hz. However, the model was never trained on a kick/snare split. Rather, it was conditioned on an adaptive LOW/HIGH split (`compute_2band.py`, `adaptive_2band.py`) that splits each clip's mel-spectrogram energy 50/50 by frequency, with split point varying by spectral content.

`audit_band_split.py` recomputes this adaptive split exactly as `compute_2band` does, including the log-compression step, across all 28 AVP "Fixed" improvisation files and compares it to the hardcoded 200 Hz threshold.

| Metric | Value |
|---|---|
| Files examined  | 28 |
| Per-file split, mean  | 1990.6 Hz |
| Per-file split, median  | 1182.8 Hz |
| Per-file split, std  | 1533.7 Hz |
| Per-file split, min (P7)  | 375.8 Hz |
| Per-file split, max (P27)  | 5050.3 Hz |
| Global aggregate split  | 1439.9 Hz |
| Original kick cutoff  | 200.0 Hz |
| Mismatch  | +1239.9 Hz (+620%) |

Every one of the 28 files exceeded 200 Hz. Even the smallest (P7, 375.8 HZ) is approximately 88% above the cutoff.

![Adaptive split vs. 200 Hz cutoff, (per participant)](figures/band_split_audit.png)

This shows that the model's actual LOW band extends well past the 200 Hz cutoff up to 1000-2000 Hz of real spectral content. The original kick detector only saw a fraction of that. The remainder was routed into the snare detector's wider 200-5000 Hz net instead, which suppressed measured kick onsets (lower recall) and contaminated snare detection with data the model never classified as snare (lower precision).

## Result 2: Hi-hat omission in ground-truth labeling
`eval_onset_f1.py`'s label classifier only recognized kick and snare tokens. That is, any other label returns `None` and is silently dropped from scoring. `audit_labl_analysis.py` compares this against a corrected, exact-token classifier that also recognizes hi-hat labels across each annotation in the 28-file AVP set.

| File | Label | Time (s) |
|---|---|---|
| P1_Improvisation_Fixed.csv | hhc | 2.524 |
| P1_Improvisation_Fixed.csv | hho | 5.927 |
| P10_Improvisation_Fixed.csv | hho | 6.676 |

More importantly, the disagreements follow the same pattern: None maps to snare and every example is a hi-hat label (hhc/hho).

Examples:
| Metric | Value |
|---|---|
| Total ref onsets examined  | 1596 |
| Classified identically  | 921 (57.7%) |
| Classified differently  | 675 (42.3%) |

The original snare detector is frequency based and hi-hats appear in the same 200-5000 Hz range. Thus, the detector miclassifies real hi-hat hits as part of its snare predictions. However, since 42.3% of reference onsets (true hi-hat hits) were dropped from the ground truth, the correctly detected hi-hat onsets have no matching reference to pair with. They register as false positives and result in directly suppressing snare precision.

## Result 3: Timing offset
A third potential defect is constant timing offset between generated audio and the rhythm prompt due to VAE/decoder latency. This could not be empirically validated since measurement requires cross-correlating a generated sample's onset envelope against its source prompt's envelope. However, GPU cluster access along with the full set of generated samples and corresponding AVP prompts, were not accessible for this extension. There was only one generated sample available (continuous_conditioned.mp3) used for the presentation demo and its source prompt could not be identified. Hence, a case study was not possible even for this generated audio sample at n = 1. Despite a lack of empirical results, this section is included for completeness since it is a previously established hypothesis.

## Limitations
Result 1 and 2 are fully validated across the 28-file AVP evaluation set used in the original results. However, Result 3 is analytical only. Due to lack of GPU access, the scope of this extension is limited to evaluation pipeline analysis rather than a full re-scored F1 table.

