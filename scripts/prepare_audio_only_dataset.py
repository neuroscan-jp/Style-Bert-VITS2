import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from pyannote.audio import Inference, Model


SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare audio-only corpus by slicing audio and filtering speaker outliers."
    )
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--speaker_name", type=str, required=True)
    parser.add_argument("--min_sec", type=float, default=1.5)
    parser.add_argument("--max_sec", type=float, default=12.0)
    parser.add_argument("--min_silence_dur_ms", type=int, default=500)
    parser.add_argument("--keep_ratio", type=float, default=0.8)
    parser.add_argument("--num_processes", type=int, default=4)
    return parser.parse_args()


def is_audio_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def collect_audio_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*") if is_audio_file(path))


def detect_speech_regions(
    audio_file: Path,
    min_silence_dur_ms: int,
    min_sec: float,
    max_sec: float,
) -> list[tuple[float, float]]:
    waveform, sample_rate = sf.read(audio_file, dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    total_sec = len(waveform) / sample_rate

    frame_ms = 20
    frame_length = max(1, int(sample_rate * frame_ms / 1000))
    frame_count = int(np.ceil(len(waveform) / frame_length))
    padded = np.pad(waveform, (0, frame_count * frame_length - len(waveform)))
    frames = padded.reshape(frame_count, frame_length)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))

    threshold = max(float(rms.max()) * 0.08, 0.005)
    min_silence_frames = max(1, int(np.ceil(min_silence_dur_ms / frame_ms)))
    min_speech_frames = max(1, int(np.ceil(min_sec * 1000 / frame_ms)))
    max_speech_frames = max(1, int(np.floor(max_sec * 1000 / frame_ms)))

    voiced = rms > threshold
    ranges: list[tuple[int, int]] = []
    start = None
    silence_run = 0
    for index, is_voiced in enumerate(voiced):
        if is_voiced:
            if start is None:
                start = index
            silence_run = 0
            continue
        if start is None:
            continue
        silence_run += 1
        if silence_run >= min_silence_frames:
            end = index - silence_run + 1
            if end - start >= min_speech_frames:
                ranges.append((start, end))
            start = None
            silence_run = 0

    if start is not None:
        end = len(voiced)
        if end - start >= min_speech_frames:
            ranges.append((start, end))

    split_ranges: list[tuple[int, int]] = []
    for start_frame, end_frame in ranges:
        cursor = start_frame
        while end_frame - cursor > max_speech_frames:
            split_ranges.append((cursor, cursor + max_speech_frames))
            cursor += max_speech_frames
        if end_frame - cursor >= min_speech_frames:
            split_ranges.append((cursor, end_frame))

    results = [
        (start_frame * frame_ms / 1000, end_frame * frame_ms / 1000)
        for start_frame, end_frame in split_ranges
    ]

    if results:
        return results

    if total_sec < min_sec:
        return []

    fallback_ranges: list[tuple[float, float]] = []
    cursor = 0.0
    while total_sec - cursor > max_sec:
        fallback_ranges.append((cursor, cursor + max_sec))
        cursor += max_sec
    if total_sec - cursor >= min_sec:
        fallback_ranges.append((cursor, total_sec))
    return fallback_ranges


def slice_audio_file(
    audio_file: Path,
    source_root: Path,
    raw_dir: Path,
    min_sec: float,
    max_sec: float,
    min_silence_dur_ms: int,
) -> list[Path]:
    timestamps = detect_speech_regions(audio_file, min_silence_dur_ms, min_sec, max_sec)
    data, sample_rate = sf.read(audio_file)
    total_ms = len(data) / sample_rate * 1000
    margin_ms = 200
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


def build_inference() -> Inference:
    model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
    inference = Inference(model, window="whole")
    inference.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return inference


def compute_embedding(inference: Inference, wav_path: Path) -> np.ndarray:
    waveform, sample_rate = sf.read(wav_path, dtype="float32")
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    else:
        waveform = waveform.T
    audio = {
        "waveform": torch.from_numpy(waveform),
        "sample_rate": sample_rate,
    }
    embedding = inference(audio)
    embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


def pick_primary_indices(embeddings: np.ndarray, keep_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    centroid = embeddings.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm != 0:
        centroid = centroid / centroid_norm
    similarities = embeddings @ centroid
    keep_count = max(1, int(np.ceil(len(similarities) * keep_ratio)))
    order = np.argsort(similarities)[::-1]
    keep_indices = np.sort(order[:keep_count])
    drop_indices = np.sort(order[keep_count:])
    return keep_indices, drop_indices


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    raw_dir = output_dir / "raw" / args.speaker_name
    report_path = output_dir / "speaker_filter_report.tsv"
    manifest_path = output_dir / "audio_segments.tsv"

    if output_dir.exists():
        shutil.rmtree(output_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    audio_files = collect_audio_files(input_dir)
    if not audio_files:
        raise SystemExit(f"No audio files found under {input_dir}")

    sliced_files: list[Path] = []
    for audio_file in audio_files:
        sliced_files.extend(
            slice_audio_file(
                audio_file=audio_file,
                source_root=input_dir,
                raw_dir=raw_dir,
                min_sec=args.min_sec,
                max_sec=args.max_sec,
                min_silence_dur_ms=args.min_silence_dur_ms,
            )
        )

    if not sliced_files:
        raise SystemExit("No usable segments were created from the source audio")

    inference = build_inference()
    embeddings = np.vstack([compute_embedding(inference, wav_path) for wav_path in sliced_files])
    keep_indices, drop_indices = pick_primary_indices(embeddings, args.keep_ratio)

    kept_files = [sliced_files[index] for index in keep_indices]
    dropped_files = [sliced_files[index] for index in drop_indices]

    with report_path.open("w", encoding="utf-8", newline="") as report_file:
        writer = csv.writer(report_file, delimiter="\t")
        writer.writerow(["path", "action", "reason"])
        for wav_path in kept_files:
            writer.writerow([wav_path.relative_to(output_dir).as_posix(), "keep", "primary_speaker_cluster"])
        for wav_path in dropped_files:
            writer.writerow([wav_path.relative_to(output_dir).as_posix(), "drop", "speaker_outlier"])

    for wav_path in dropped_files:
        wav_path.unlink(missing_ok=True)

    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.writer(manifest_file, delimiter="\t")
        writer.writerow(["audio_path", "speaker"])
        for wav_path in kept_files:
            writer.writerow([wav_path.relative_to(output_dir).as_posix(), args.speaker_name])

    print(f"input_files={len(audio_files)}")
    print(f"sliced_files={len(sliced_files)}")
    print(f"kept_files={len(kept_files)}")
    print(f"dropped_files={len(dropped_files)}")
    print(f"output_dir={output_dir}")
    print(f"report={report_path}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()