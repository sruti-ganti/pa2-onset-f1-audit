"""
Annotation-analysis script: quantify AVP reference onsets are
classified differently by the substring-matching label classifier
in eval_onset_f1.py vs. the exact-token classifier in eval_onset_f1_claude.py.

Usage:
    python audit_label_analysis.py --avp-dir /path/to/AVP_Dataset/Fixed

Requirements:
    pip install pandas
    pip install jams   # if annotations are .jams files
"""

import argparse
import sys
from pathlib import Path

# ORIGINAL classifier — verbatim from eval_onset_f1.py

_KICK_LABELS_ORIG = {"kd", "bd", "kick", "bassdrum", "bass_drum", "k"}
_SNARE_LABELS_ORIG = {"sd", "snare", "sn", "snaredrum", "snare_drum"}

def classify_original(label: str) -> str | None:
    l = label.lower().strip()
    if any(k in l for k in _KICK_LABELS_ORIG):
        return "kick"
    if any(s in l for s in _SNARE_LABELS_ORIG):
        return "snare"
    return None

# CORRECTED classifier — verbatim from eval_onset_f1_claude.py

_KICK_LABELS = {"kd", "bd", "kick", "bassdrum", "bass_drum", "kick_drum"}
_SNARE_LABELS = {"sd", "snare", "sn", "snaredrum", "snare_drum"}
_HAT_LABELS = {
    "hho", "hhc", "hh", "chh", "ohh", "hihat", "hi-hat", "hi_hat",
    "closed_hihat", "opened_hihat", "open_hihat",
    "closed_hi-hat", "opened_hi-hat", "cym", "cymbal",
}

def classify_corrected(label: str) -> str | None:
    l = label.lower().strip()
    if l in _KICK_LABELS:
        return "low"
    if l in _SNARE_LABELS or l in _HAT_LABELS:
        return "high"
    return None

_BAND_TO_INSTRUMENT = {"low": "kick", "high": "snare"}

def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def load_raw_labels(ann_path: Path) -> list[tuple[float, str]]:
    if ann_path.suffix == ".jams":
        import jams
        jam = jams.load(str(ann_path))
        out = []
        for ann in jam.annotations:
            for obs in ann.data:
                label = str(obs.value) if hasattr(obs, "value") else ""
                t = (
                    float(obs.time.total_seconds())
                    if hasattr(obs.time, "total_seconds")
                    else float(obs.time)
                )
                out.append((t, label))
        return out

    import pandas as pd

    raw = ann_path.read_text()
    sep = "\t" if "\t" in raw else ","
    first_tok = raw.strip().split("\n")[0].split(sep)[0].strip()
    has_header = not _is_float(first_tok)

    if has_header:
        df = pd.read_csv(ann_path, sep=sep)
        df.columns = [c.lower() for c in df.columns]
        onset_col = next(
            (c for c in df.columns if any(k in c for k in ("onset", "time", "start"))),
            df.columns[0],
        )
        label_col = next(
            (c for c in df.columns if any(k in c for k in ("label", "event", "class", "inst"))),
            df.columns[-1],
        )
    else:
        df = pd.read_csv(ann_path, sep=sep, header=None)
        onset_col, label_col = 0, 1

    return [(float(row[onset_col]), str(row[label_col])) for _, row in df.iterrows()]


def find_annotation(audio_path: Path) -> Path | None:
    for ext in (".jams", ".csv", ".tsv", ".txt"):
        cand = audio_path.with_suffix(ext)
        if cand.exists():
            return cand
    return None


def collect_improvisation_pairs(avp_dir: Path) -> list[tuple[Path, Path]]:
    samples = []
    for entry in sorted(avp_dir.iterdir()):
        if not entry.is_dir():
            continue
        for wav in sorted(entry.glob("*.wav")):
            if "_Improvisation_" not in wav.name:
                continue
            ann = find_annotation(wav)
            if ann is not None:
                samples.append((wav, ann))
    return samples

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--avp-dir", required=True,
                   help="Path to the AVP 'Fixed' folder (Participant_N subdirs).")
    p.add_argument("--max-examples", type=int, default=15,
                   help="Number of example disagreements to print (default: 15).")
    args = p.parse_args()

    avp_dir = Path(args.avp_dir)
    if not avp_dir.is_dir():
        sys.exit(f"Not a directory: {avp_dir}")

    pairs = collect_improvisation_pairs(avp_dir)
    if not pairs:
        sys.exit(
            f"No improvisation files with annotations found under {avp_dir}.\n"
            "Expected: Participant_N/ subdirs containing *_Improvisation_*.wav "
            "with matching .jams/.csv/.tsv/.txt annotation files."
        )
    print(f"Found {len(pairs)} improvisation files with annotations.\n")

    total = 0
    agree = 0
    disagree_counts: dict[str, int] = {}
    disagree_examples: list[tuple[str, str, float, str | None, str | None]] = []
    parse_failures = 0

    for wav, ann in pairs:
        try:
            raw_labels = load_raw_labels(ann)
        except Exception as e:
            print(f"  SKIP {ann.name}: failed to parse ({e})")
            parse_failures += 1
            continue

        for t, label in raw_labels:
            total += 1
            orig = classify_original(label)            # 'kick' | 'snare' | None
            corr_band = classify_corrected(label)       # 'low' | 'high' | None
            corr = _BAND_TO_INSTRUMENT.get(corr_band) if corr_band else None

            if orig == corr:
                agree += 1
            else:
                key = f"{orig!r} -> {corr!r}"
                disagree_counts[key] = disagree_counts.get(key, 0) + 1
                if len(disagree_examples) < args.max_examples:
                    disagree_examples.append((ann.name, label, t, orig, corr))

    if total == 0:
        sys.exit("No onsets parsed — check annotation format / parse failures above.")

    disagree = total - agree
    print("LABEL CLASSIFICATION AUDIT — original vs. corrected")
    print("-" * 68)
    print(f"  files with parse failures        : {parse_failures}")
    print(f"  total reference onsets examined  : {total}")
    print(f"  aligned                          : {agree}  ({100 * agree / total:.1f}%)")
    print(f"  unaligned                        : {disagree}  ({100 * disagree / total:.1f}%)")
    print()
    print("Disagreement breakdown (original_label -> corrected_label):")
    for key, count in sorted(disagree_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {key:<28} : {count}  ({100 * count / total:.2f}% of all onsets)")
    print()
    print(f"Example disagreements (showing up to {args.max_examples}):")
    print(f"  {'file':<42} {'raw_label':<14} {'time':>8}  {'orig':<8} {'corrected'}")
    for fname, label, t, orig, corr in disagree_examples:
        print(f"  {fname:<42} {label!r:<14} {t:8.3f}  {str(orig):<8} {corr}")


if __name__ == "__main__":
    main()
