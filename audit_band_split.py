"""
Quantify the mismatch between the model's actual adaptive LOW/HIGH split 
frequency and the FIXED frequency boundaries eval_onset_f1.py hardcodes 
for kick/snare transcription:
    kick  = low-pass  < 200 Hz
    snare = band-pass 200-5000 Hz

Usage:
    python audit_band_split.py --avp-dir /path/to/AVP_Dataset/Fixed

Requirements:
    pip install torch torchaudio numpy
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import re

_ORIG_KICK_LOWPASS_HZ = 200.0
_ORIG_SNARE_BAND_HZ = (200.0, 5000.0)

def load_mono(path: str, target_sr: int | None = None) -> tuple[torch.Tensor, int]:
    data, sr = sf.read(path, dtype="float32", always_2d=True)  # [samples, channels]
    waveform = torch.from_numpy(data.T)  # [channels, samples]
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if target_sr is not None and sr != target_sr:
        import torchaudio
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr
    return waveform, sr

def adaptive_split_index(mel_power: torch.Tensor) -> int:
    # mel_power here is the LOG-compressed mel spectrogram (not raw linear power)
    avg = mel_power.mean(dim=1)
    cumsum = avg.cumsum(dim=0)
    half = cumsum[-1] / 2.0
    idx = int((cumsum < half).sum().item())
    return max(1, min(idx, mel_power.shape[0] - 1))

def split_idx_to_hz(sample_rate: int, n_mels: int, split_idx: int) -> float:
    f_min, f_max = 0.0, sample_rate / 2.0
    m_min = 2595.0 * np.log10(1.0 + f_min / 700.0)
    m_max = 2595.0 * np.log10(1.0 + f_max / 700.0)
    m_pts = np.linspace(m_min, m_max, n_mels + 2)
    f_pts = 700.0 * (10.0 ** (m_pts / 2595.0) - 1.0)
    return float(f_pts[min(split_idx + 1, len(f_pts) - 1)])

def collect_improvisation_files(avp_dir: Path) -> list[Path]:
    files = []
    for entry in sorted(avp_dir.iterdir()):
        if not entry.is_dir():
            continue
        for wav in sorted(entry.glob("*.wav")):
            if "_Improvisation_" in wav.name:
                files.append(wav)
    return files

def _participant_num(filename: str) -> int:
    m = re.match(r"P(\d+)", filename)
    return int(m.group(1)) if m else -1

def save_split_plot(filenames, split_hzs, global_hz, out_path):
    order = sorted(range(len(filenames)), key=lambda i: _participant_num(filenames[i]))
    labels = [f"P{_participant_num(filenames[i])}" for i in order]
    values = [split_hzs[i] for i in order]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#4C72B0" if v > _ORIG_KICK_LOWPASS_HZ else "#C44E52" for v in values]
    ax.bar(labels, values, color=colors)
    ax.axhline(_ORIG_KICK_LOWPASS_HZ, color="#C44E52", linestyle="--", linewidth=1.5,
               label=f"Original eval kick cutoff ({_ORIG_KICK_LOWPASS_HZ:.0f} Hz)")
    ax.axhline(global_hz, color="#55A868", linestyle="--", linewidth=1.5,
               label=f"Global aggregate split ({global_hz:.0f} Hz)")
    ax.set_ylabel("Adaptive split frequency (Hz)")
    ax.set_xlabel("AVP participant (improvisation file)")
    ax.set_title("Adaptive LOW/HIGH Split vs. Original Eval's Fixed Kick Cutoff")
    ax.legend(fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved: {out_path}")

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--avp-dir", required=True,
                   help="Path to the AVP 'Fixed' folder (Participant_N subdirs).")
    p.add_argument("--sample-rate", type=int, default=44100)
    p.add_argument("--n-fft", type=int, default=2048)
    p.add_argument("--hop-length", type=int, default=512)
    p.add_argument("--n-mels", type=int, default=80)
    p.add_argument("--n-files", type=int, default=None,
                   help="Limit to N files for a quicker run (default: all).")
    p.add_argument("--plot", action="store_true",
                   help="Save a bar chart of per-file split frequencies vs. 200Hz.")
    p.add_argument("--plot-output", default="band_split_audit.png",
                   help="Output path for the plot (default: band_split_audit.png).")
    args = p.parse_args()

    avp_dir = Path(args.avp_dir)
    files = collect_improvisation_files(avp_dir)
    if not files:
        sys.exit(f"No *_Improvisation_*.wav files found under {avp_dir}")
    if args.n_files:
        files = files[: args.n_files]
    print(f"Computing adaptive split over {len(files)} improv files...\n")

    split_idxs, split_hzs, split_names = [], [], []
    accumulated_mel = None
    count = 0

    for i, f in enumerate(files):
        try:
            waveform, sr = load_mono(str(f), target_sr=args.sample_rate)
            mel_transform = T.MelSpectrogram(
                sample_rate=sr, n_fft=args.n_fft, hop_length=args.hop_length,
                n_mels=args.n_mels, power=2.0,
            )
            mel = torch.log1p(mel_transform(waveform).squeeze(0))  # [n_mels, T]
        except Exception as e:
            print(f"  SKIP {f.name}: {e}")
            continue

        idx = adaptive_split_index(mel)
        hz = split_idx_to_hz(sr, args.n_mels, idx)
        split_idxs.append(idx)
        split_hzs.append(hz)
        split_names.append(f.name)
        print(f"  [{i+1}/{len(files)}] {f.name:<45} split_idx={idx:3d}  split_hz={hz:7.1f} Hz")

        frame_avg = mel.mean(dim=1, keepdim=True)  # [n_mels, 1]
        accumulated_mel = frame_avg if accumulated_mel is None else accumulated_mel + frame_avg
        count += 1

    if count == 0:
        sys.exit("No files processed successfully.")

    idxs = np.array(split_idxs)
    hzs = np.array(split_hzs)
    global_avg_mel = accumulated_mel / count
    global_idx = adaptive_split_index(global_avg_mel)
    global_hz = split_idx_to_hz(args.sample_rate, args.n_mels, global_idx)

    print("\nADAPTIVE SPLIT vs ORIGINAL EVAL'S HARDCODED BOUNDARIES")
    print("-" * 68)
    print(f"  n files processed                  : {count}")
    print(f"  per-file split (mean)              : {hzs.mean():7.1f} Hz  (bin {idxs.mean():.1f}/{args.n_mels})")
    print(f"  per-file split (median)            : {np.median(hzs):7.1f} Hz")
    print(f"  per-file split (std)               : {hzs.std():7.1f} Hz")
    print(f"  global aggregate split             : {global_hz:7.1f} Hz  (bin {global_idx}/{args.n_mels})")
    print()
    print(f"  original eval kick cutoff          : {_ORIG_KICK_LOWPASS_HZ:7.1f} Hz  (low-pass)")
    print(f"  original eval snare band           : {_ORIG_SNARE_BAND_HZ[0]:.1f}-{_ORIG_SNARE_BAND_HZ[1]:.1f} Hz")
    print()
    mismatch = global_hz - _ORIG_KICK_LOWPASS_HZ
    pct = 100 * mismatch / _ORIG_KICK_LOWPASS_HZ
    print(f"  mismatch (global split - 200Hz)    : {mismatch:+.1f} Hz  ({pct:+.0f}% relative to 200Hz)")
    
    if args.plot:
        save_split_plot(split_names, split_hzs, global_hz, args.plot_output)

if __name__ == "__main__":
    main()
