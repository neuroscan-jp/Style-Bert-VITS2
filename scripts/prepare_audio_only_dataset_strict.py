import argparse
import csv
import math
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from prepare_audio_only_dataset import (
    build_inference,
    collect_audio_files,
    compute_embedding,
    detect_speech_regions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a stricter audio-only corpus by clustering speaker embeddings and dropping noisy outliers."
    )
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--speaker_name", type=str, required=True)
    parser.add_argument("--min_sec", type=float, default=0.9)
    parser.add_argument("--max_sec", type=float, default=5.0)
    parser.add_argument("--min_silence_dur_ms", type=int, default=220)
    parser.add_argument("--margin_ms", type=int, default=120)
    parser.add_argument("--cluster_similarity", type=float, default=0.80)
    parser.add_argument("--min_similarity", type=float, default=0.77)
    parser.add_argument("--similarity_quantile", type=float, default=0.25)
    parser.add_argument("--min_voiced_ratio", type=float, default=0.45)
    parser.add_argument("--min_contrast_db", type=float, default=11.0)
    return parser.parse_args()


def compute_voiced_stats(audio: np.ndarray, sample_rate: int) -> tuple[float, float]:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    frame_ms = 20
    frame_length = max(1, int(sample_rate * frame_ms / 1000))
    frame_count = int(np.ceil(len(audio) / frame_length))
    padded = np.pad(audio, (0, frame_count * frame_length - len(audio)))
    frames = padded.reshape(frame_count, frame_length)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    threshold = max(float(np.quantile(rms, 0.65)) * 0.35, 0.004)
    voiced_ratio = float(np.mean(rms > threshold))
    low = float(np.quantile(rms, 0.20))
    high = float(np.quantile(rms, 0.95))
    contrast_db = 20.0 * math.log10((high + 1e-6) / (low + 1e-6))
    return voiced_ratio, contrast_db


def slice_audio_file(
    audio_file: Path,
    source_root: Path,
    raw_dir: Path,
    min_sec: float,
    max_sec: float,
    min_silence_dur_ms: int,
    margin_ms: int,
) -> list[Path]:
    timestamps = detect_speech_regions(audio_file, min_silence_dur_ms, min_sec, max_sec)
    data, sample_rate = sf.read(audio_file)
    total_ms = len(data) / sample_rate * 1000
    relative_parent = audio_file.relative_to(source_root).parent
    output_parent = raw_dir / relative_parent
    output_parent.mkdir(parents=True, exist_ok=True)

    written_files: list[Path] = []
    for index, timestamp in enumerate(timestamps):
        start_ms = max(timestamp[0] * 1000 - margin_ms, 0)
        end_ms = min(timestamp[1] * 1000 + margin_ms, total_ms)
        start_sample = int(start_ms / 1000 * sample_rate)
        end_sample = int(end_ms / 1000 * sample_rate)
        segment = data[start_sample:end_sample]
        if len(segment) == 0:
            continue
        output_name = f"{audio_file.stem}-{int(start_ms)}-{int(end_ms)}.wav"
        output_path = output_parent / output_name
        sf.write(str(output_path), segment, sample_rate)
        written_files.append(output_path)
    return written_files


def build_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    return embeddings @ embeddings.T


def find_components(similarity: np.ndarray, threshold: float) -> list[list[int]]:
    size = similarity.shape[0]
    visited = np.zeros(size, dtype=bool)
    components: list[list[int]] = []

    for index in range(size):
        if visited[index]:
            continue
        stack = [index]
        visited[index] = True
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            neighbors = np.where(similarity[current] >= threshold)[0]
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(int(neighbor))
        components.append(sorted(component))
    return components


def choose_primary_component(components: list[list[int]], file_ids: list[str]) -> list[int]:
    def score(component: list[int]) -> tuple[int, int]:
        unique_files = len({file_ids[index] for index in component})
        return (len(component), unique_files)

    return max(components, key=score)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    raw_dir = output_dir / "raw" / args.speaker_name
    report_path = output_dir / "strict_filter_report.tsv"

    if output_dir.exists():
        shutil.rmtree(output_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    audio_files = collect_audio_files(input_dir)
    if not audio_files:
        raise SystemExit(f"No audio files found under {input_dir}")

    sliced_files: list[Path] = []
    source_ids: list[str] = []
    for audio_file in audio_files:
        current = slice_audio_file(
            audio_file=audio_file,
            source_root=input_dir,
            raw_dir=raw_dir,
            min_sec=args.min_sec,
            max_sec=args.max_sec,
            min_silence_dur_ms=args.min_silence_dur_ms,
            margin_ms=args.margin_ms,
        )
        sliced_files.extend(current)
        source_ids.extend([audio_file.stem] * len(current))

    if not sliced_files:
        raise SystemExit("No usable segments were created from the source audio")

    inference = build_inference()
    embeddings = np.vstack([compute_embedding(inference, wav_path) for wav_path in sliced_files])
    similarity = build_similarity_matrix(embeddings)
    components = find_components(similarity, args.cluster_similarity)
    primary_component = choose_primary_component(components, source_ids)
    primary_set = set(primary_component)

    centroid = embeddings[primary_component].mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm != 0:
        centroid = centroid / centroid_norm
    similarities = embeddings @ centroid
    primary_similarities = similarities[primary_component]
    dynamic_threshold = max(float(np.quantile(primary_similarities, args.similarity_quantile)), args.min_similarity)

    kept_files: list[Path] = []
    dropped_files: list[Path] = []
    cluster_sizes = Counter({index: len(component) for index, component in enumerate(components)})

    with report_path.open("w", encoding="utf-8", newline="") as report_file:
        writer = csv.writer(report_file, delimiter="\t")
        writer.writerow(
            [
                "path",
                "source_id",
                "cluster_id",
                "cluster_size",
                "similarity",
                "voiced_ratio",
                "contrast_db",
                "action",
                "reason",
            ]
        )

        cluster_lookup: dict[int, int] = {}
        for cluster_id, component in enumerate(components):
            for member in component:
                cluster_lookup[member] = cluster_id

        for index, wav_path in enumerate(sliced_files):
            audio, sample_rate = sf.read(wav_path, always_2d=False)
            voiced_ratio, contrast_db = compute_voiced_stats(audio, sample_rate)
            cluster_id = cluster_lookup[index]

            keep = True
            reasons: list[str] = []
            if index not in primary_set:
                keep = False
                reasons.append("non_primary_speaker_cluster")
            if similarities[index] < dynamic_threshold:
                keep = False
                reasons.append("low_speaker_similarity")
            if voiced_ratio < args.min_voiced_ratio:
                keep = False
                reasons.append("low_voiced_ratio")
            if contrast_db < args.min_contrast_db:
                keep = False
                reasons.append("low_dynamic_contrast")

            writer.writerow(
                [
                    wav_path.relative_to(output_dir).as_posix(),
                    source_ids[index],
                    cluster_id,
                    cluster_sizes[cluster_id],
                    f"{similarities[index]:.6f}",
                    f"{voiced_ratio:.4f}",
                    f"{contrast_db:.2f}",
                    "keep" if keep else "drop",
                    ",".join(reasons) if reasons else "primary_speaker_clean_segment",
                ]
            )

            if keep:
                kept_files.append(wav_path)
            else:
                dropped_files.append(wav_path)

    for wav_path in dropped_files:
        wav_path.unlink(missing_ok=True)

    print(f"input_files={len(audio_files)}")
    print(f"sliced_files={len(sliced_files)}")
    print(f"clusters={len(components)}")
    print(f"primary_cluster_size={len(primary_component)}")
    print(f"dynamic_threshold={dynamic_threshold:.6f}")
    print(f"kept_files={len(kept_files)}")
    print(f"dropped_files={len(dropped_files)}")
    print(f"output_dir={output_dir}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()