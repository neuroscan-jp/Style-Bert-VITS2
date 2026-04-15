import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf

from prepare_audio_only_dataset import build_inference, compute_embedding, detect_speech_regions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-split suspect speaker segments and keep only chunks close to a reference speaker centroid."
    )
    parser.add_argument("--reference_dir", type=Path, required=True)
    parser.add_argument("--target_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--target_prefix", type=str, required=True)
    parser.add_argument("--reference_exclude_prefix", type=str)
    parser.add_argument("--min_sec", type=float, default=1.0)
    parser.add_argument("--max_sec", type=float, default=4.5)
    parser.add_argument("--min_silence_dur_ms", type=int, default=250)
    parser.add_argument("--margin_ms", type=int, default=120)
    parser.add_argument("--threshold_quantile", type=float, default=0.15)
    return parser.parse_args()


def collect_wavs(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*.wav") if path.is_file())


def build_reference_set(reference_dir: Path, exclude_prefix: str | None) -> list[Path]:
    wavs = collect_wavs(reference_dir)
    if exclude_prefix:
        wavs = [path for path in wavs if not path.stem.startswith(exclude_prefix)]
    if not wavs:
        raise SystemExit("No reference wav files found after filtering")
    return wavs


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right))


def compute_centroid(embeddings: np.ndarray) -> np.ndarray:
    centroid = embeddings.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm == 0:
        return centroid
    return centroid / norm


def slice_regions(
    wav_path: Path,
    min_silence_dur_ms: int,
    min_sec: float,
    max_sec: float,
    margin_ms: int,
) -> list[tuple[np.ndarray, int, int, int]]:
    timestamps = detect_speech_regions(
        wav_path,
        min_silence_dur_ms=min_silence_dur_ms,
        min_sec=min_sec,
        max_sec=max_sec,
    )
    audio, sample_rate = sf.read(wav_path, dtype="float32")
    total_ms = int(len(audio) / sample_rate * 1000)
    results: list[tuple[np.ndarray, int, int, int]] = []
    for start_sec, end_sec in timestamps:
        start_ms = max(int(start_sec * 1000) - margin_ms, 0)
        end_ms = min(int(end_sec * 1000) + margin_ms, total_ms)
        start_sample = int(start_ms / 1000 * sample_rate)
        end_sample = int(end_ms / 1000 * sample_rate)
        segment = audio[start_sample:end_sample]
        if len(segment) == 0:
            continue
        results.append((segment, sample_rate, start_ms, end_ms))
    if results:
        return results

    start_ms = 0
    end_ms = total_ms
    return [(audio, sample_rate, start_ms, end_ms)]


def write_segment(output_path: Path, audio: np.ndarray, sample_rate: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, sample_rate, subtype="PCM_16")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_audio_dir = args.output_dir / "audio"
    report_path = args.output_dir / "refine_report.tsv"

    reference_wavs = build_reference_set(args.reference_dir, args.reference_exclude_prefix)
    target_wavs = [path for path in collect_wavs(args.target_dir) if path.stem.startswith(args.target_prefix)]
    if not target_wavs:
        raise SystemExit(f"No target wav files found for prefix: {args.target_prefix}")

    inference = build_inference()
    reference_embeddings = np.vstack([compute_embedding(inference, wav_path) for wav_path in reference_wavs])
    centroid = compute_centroid(reference_embeddings)
    reference_similarities = reference_embeddings @ centroid
    similarity_threshold = float(np.quantile(reference_similarities, args.threshold_quantile))

    kept_count = 0
    dropped_count = 0
    with report_path.open("w", encoding="utf-8", newline="") as report_file:
        writer = csv.writer(report_file, delimiter="\t")
        writer.writerow(["source_path", "output_path", "start_ms", "end_ms", "similarity", "action"])

        for wav_path in target_wavs:
            regions = slice_regions(
                wav_path,
                min_silence_dur_ms=args.min_silence_dur_ms,
                min_sec=args.min_sec,
                max_sec=args.max_sec,
                margin_ms=args.margin_ms,
            )
            for index, (segment, sample_rate, start_ms, end_ms) in enumerate(regions):
                temp_path = args.output_dir / "_tmp_eval.wav"
                write_segment(temp_path, segment, sample_rate)
                embedding = compute_embedding(inference, temp_path)
                similarity = cosine_similarity(embedding, centroid)
                if similarity >= similarity_threshold:
                    output_name = f"{wav_path.stem}__{index:02d}-{start_ms}-{end_ms}.wav"
                    output_path = output_audio_dir / output_name
                    write_segment(output_path, segment, sample_rate)
                    writer.writerow([
                        wav_path.name,
                        output_path.relative_to(args.output_dir).as_posix(),
                        start_ms,
                        end_ms,
                        f"{similarity:.6f}",
                        "keep",
                    ])
                    kept_count += 1
                else:
                    writer.writerow([
                        wav_path.name,
                        "",
                        start_ms,
                        end_ms,
                        f"{similarity:.6f}",
                        "drop",
                    ])
                    dropped_count += 1

    temp_path = args.output_dir / "_tmp_eval.wav"
    temp_path.unlink(missing_ok=True)

    print(f"reference_files={len(reference_wavs)}")
    print(f"target_files={len(target_wavs)}")
    print(f"threshold={similarity_threshold:.6f}")
    print(f"kept_segments={kept_count}")
    print(f"dropped_segments={dropped_count}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()