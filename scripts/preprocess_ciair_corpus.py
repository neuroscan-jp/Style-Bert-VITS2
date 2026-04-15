import argparse
import csv
import os
import re
import shutil
import wave
from dataclasses import dataclass
from pathlib import Path


TAGGED_INLINE_PATTERN = re.compile(r"\(([A-Z]+)\s*([^)]*)\)")
ANGLE_TAG_PATTERN = re.compile(r"<([^>]+)>")


@dataclass
class TranscriptEntry:
    surface_text: str
    pronunciation_tagged: str
    spoken_text: str
    tag_summary: str


@dataclass
class SilenceInterval:
    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass
class SegmentPlan:
    start_ms: int
    end_ms: int
    method: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess CIAIR-VCV style wav/txt pairs into a Style-Bert-VITS2 dataset."
    )
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=Path("Data"))
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--speaker_name", type=str)
    parser.add_argument("--language", choices=["JP", "EN", "ZH"], default="JP")
    parser.add_argument("--text_encoding", type=str, default="cp932")
    parser.add_argument("--copy_mode", choices=["hardlink", "copy"], default="hardlink")
    parser.add_argument("--frame_ms", type=int, default=20)
    parser.add_argument("--min_silence_ms", type=int, default=180)
    parser.add_argument("--long_silence_ms", type=int, default=400)
    parser.add_argument("--silence_ratio", type=float, default=0.08)
    parser.add_argument("--min_segment_ms", type=int, default=180)
    parser.add_argument("--keep_existing", action="store_true")
    return parser.parse_args()


def parse_transcript_file(txt_path: Path, encoding: str) -> list[TranscriptEntry]:
    entries: list[TranscriptEntry] = []
    with txt_path.open("r", encoding=encoding) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or "&" not in line:
                continue
            surface_text, pronunciation_tagged = [part.strip() for part in line.split("&", 1)]
            spoken_text, tag_summary = flatten_pronunciation(pronunciation_tagged)
            if not spoken_text:
                continue
            entries.append(
                TranscriptEntry(
                    surface_text=surface_text,
                    pronunciation_tagged=pronunciation_tagged,
                    spoken_text=spoken_text,
                    tag_summary=tag_summary,
                )
            )
    return entries


def flatten_pronunciation(pronunciation_tagged: str) -> tuple[str, str]:
    tag_counts: dict[str, int] = {}

    def replace_inline(match: re.Match[str]) -> str:
        tag_name = match.group(1)
        tag_counts[tag_name] = tag_counts.get(tag_name, 0) + 1
        return match.group(2).strip()

    def replace_angle(match: re.Match[str]) -> str:
        tag_name = match.group(1).strip()
        tag_counts[f"<{tag_name}>"] = tag_counts.get(f"<{tag_name}>", 0) + 1
        if tag_name == "H":
            return "ー"
        return ""

    spoken_text = TAGGED_INLINE_PATTERN.sub(replace_inline, pronunciation_tagged)
    spoken_text = ANGLE_TAG_PATTERN.sub(replace_angle, spoken_text)
    spoken_text = re.sub(r"\s+", "", spoken_text)
    tag_summary = ",".join(f"{name}:{count}" for name, count in sorted(tag_counts.items()))
    return spoken_text, tag_summary


def read_wave_bytes(wav_path: Path) -> tuple[wave._wave_params, bytes]:
    with wave.open(str(wav_path), "rb") as wav_file:
        params = wav_file.getparams()
        frames = wav_file.readframes(params.nframes)
    return params, frames


def write_wave_segment(destination: Path, params: wave._wave_params, frame_bytes: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as wav_file:
        wav_file.setparams(params)
        wav_file.writeframes(frame_bytes)


def pcm_slice(frame_bytes: bytes, params: wave._wave_params, start_ms: int, end_ms: int) -> bytes:
    bytes_per_frame = params.sampwidth * params.nchannels
    start_frame = max(0, round(start_ms * params.framerate / 1000))
    end_frame = min(params.nframes, round(end_ms * params.framerate / 1000))
    start_offset = start_frame * bytes_per_frame
    end_offset = end_frame * bytes_per_frame
    return frame_bytes[start_offset:end_offset]


def frame_rms_values(frame_bytes: bytes, params: wave._wave_params, frame_ms: int) -> list[int]:
    bytes_per_frame = params.sampwidth * params.nchannels
    frames_per_window = max(1, round(params.framerate * frame_ms / 1000))
    window_size = frames_per_window * bytes_per_frame
    values: list[int] = []
    for offset in range(0, len(frame_bytes), window_size):
        window = frame_bytes[offset : offset + window_size]
        if not window:
            continue
        values.append(audio_rms(window, params.sampwidth))
    return values


def audio_rms(frame_bytes: bytes, sample_width: int) -> int:
    if sample_width == 1:
        midpoint = 128
        total = 0
        count = len(frame_bytes)
        if count == 0:
            return 0
        for value in frame_bytes:
            delta = value - midpoint
            total += delta * delta
        return int((total / count) ** 0.5)

    sample_count = len(frame_bytes) // sample_width
    if sample_count == 0:
        return 0

    total = 0
    for index in range(0, len(frame_bytes), sample_width):
        sample = int.from_bytes(
            frame_bytes[index : index + sample_width],
            byteorder="little",
            signed=True,
        )
        total += sample * sample
    return int((total / sample_count) ** 0.5)


def detect_silences(
    frame_bytes: bytes,
    params: wave._wave_params,
    frame_ms: int,
    min_silence_ms: int,
    silence_ratio: float,
) -> list[SilenceInterval]:
    rms_values = frame_rms_values(frame_bytes, params, frame_ms)
    if not rms_values:
        return []

    peak_rms = max(rms_values)
    silence_threshold = max(peak_rms * silence_ratio, 48)
    min_silence_frames = max(1, round(min_silence_ms / frame_ms))

    silences: list[SilenceInterval] = []
    current_start: int | None = None

    for frame_index, rms_value in enumerate(rms_values):
        if rms_value <= silence_threshold:
            if current_start is None:
                current_start = frame_index
            continue
        if current_start is None:
            continue
        if frame_index - current_start >= min_silence_frames:
            silences.append(
                SilenceInterval(
                    start_ms=current_start * frame_ms,
                    end_ms=frame_index * frame_ms,
                )
            )
        current_start = None

    if current_start is not None and len(rms_values) - current_start >= min_silence_frames:
        silences.append(
            SilenceInterval(
                start_ms=current_start * frame_ms,
                end_ms=len(rms_values) * frame_ms,
            )
        )

    return silences


def plan_segments(
    entries: list[TranscriptEntry],
    params: wave._wave_params,
    silences: list[SilenceInterval],
    min_segment_ms: int,
) -> tuple[list[SegmentPlan], str]:
    total_ms = round(params.nframes * 1000 / params.framerate)
    lead_trim = 0
    tail_trim = total_ms

    if silences and silences[0].start_ms == 0:
        lead_trim = silences[0].end_ms
    if silences and silences[-1].end_ms >= total_ms - 1:
        tail_trim = silences[-1].start_ms
    if tail_trim - lead_trim < min_segment_ms:
        lead_trim = 0
        tail_trim = total_ms

    if len(entries) == 1:
        return [SegmentPlan(start_ms=lead_trim, end_ms=tail_trim, method="trimmed_single")], "ok"

    internal_silences = [
        interval
        for interval in silences
        if interval.start_ms > lead_trim and interval.end_ms < tail_trim
    ]
    required_boundaries = len(entries) - 1

    if len(internal_silences) >= required_boundaries:
        chosen = sorted(
            sorted(internal_silences, key=lambda interval: interval.duration_ms, reverse=True)[
                :required_boundaries
            ],
            key=lambda interval: interval.start_ms,
        )
        plans = plans_from_silences(chosen, lead_trim, tail_trim)
        if segments_are_valid(plans, min_segment_ms):
            method = "silence_split_exact" if len(internal_silences) == required_boundaries else "silence_split_top"
            return plans, method

    proportional = proportional_boundaries(entries, lead_trim, tail_trim)
    plans = plans_from_boundaries(proportional, lead_trim, tail_trim)
    return plans, "proportional_fallback"


def proportional_boundaries(entries: list[TranscriptEntry], start_ms: int, end_ms: int) -> list[int]:
    total_chars = sum(max(1, len(entry.spoken_text)) for entry in entries)
    usable_ms = max(1, end_ms - start_ms)
    boundaries: list[int] = []
    consumed_chars = 0
    for entry in entries[:-1]:
        consumed_chars += max(1, len(entry.spoken_text))
        boundaries.append(start_ms + round(usable_ms * consumed_chars / total_chars))
    return boundaries


def plans_from_boundaries(boundaries: list[int], start_ms: int, end_ms: int) -> list[SegmentPlan]:
    points = [start_ms] + boundaries + [end_ms]
    plans: list[SegmentPlan] = []
    for left, right in zip(points, points[1:]):
        plans.append(SegmentPlan(start_ms=left, end_ms=right, method="split"))
    return plans


def plans_from_silences(
    silences: list[SilenceInterval], start_ms: int, end_ms: int
) -> list[SegmentPlan]:
    plans: list[SegmentPlan] = []
    current_start = start_ms
    for silence in silences:
        plans.append(SegmentPlan(start_ms=current_start, end_ms=silence.start_ms, method="split"))
        current_start = silence.end_ms
    plans.append(SegmentPlan(start_ms=current_start, end_ms=end_ms, method="split"))
    return plans


def segments_are_valid(plans: list[SegmentPlan], min_segment_ms: int) -> bool:
    return all(plan.end_ms - plan.start_ms >= min_segment_ms for plan in plans)


def safe_link_or_copy(source: Path, destination: Path, copy_mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    if copy_mode == "copy":
        shutil.copy2(source, destination)
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def build_output_name(source_wav: Path, segment_index: int, total_segments: int) -> str:
    if total_segments == 1:
        return source_wav.name
    return f"{source_wav.stem}_{segment_index:02d}.wav"


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    model_name = args.model_name or input_dir.name
    speaker_name = args.speaker_name or input_dir.name
    model_dir = args.dataset_root.resolve() / model_name
    raw_dir = model_dir / "raw"
    esd_path = model_dir / "esd.list"
    manifest_path = model_dir / "ciair_manifest.tsv"
    silence_report_path = model_dir / "silence_report.tsv"
    review_report_path = model_dir / "review_needed.tsv"

    if model_dir.exists() and not args.keep_existing:
        shutil.rmtree(model_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(path for path in input_dir.rglob("*.wav") if path.is_file())
    written_rows = 0

    with (
        esd_path.open("w", encoding="utf-8", newline="") as esd_file,
        manifest_path.open("w", encoding="utf-8", newline="") as manifest_file,
        silence_report_path.open("w", encoding="utf-8", newline="") as silence_file,
        review_report_path.open("w", encoding="utf-8", newline="") as review_file,
    ):
        manifest_writer = csv.writer(manifest_file, delimiter="\t")
        silence_writer = csv.writer(silence_file, delimiter="\t")
        review_writer = csv.writer(review_file, delimiter="\t")
        manifest_writer.writerow(
            [
                "source_wav",
                "source_txt",
                "style",
                "segment_index",
                "segment_start_ms",
                "segment_end_ms",
                "split_method",
                "surface_text",
                "pronunciation_tagged",
                "spoken_text",
                "tag_summary",
                "output_wav",
            ]
        )
        silence_writer.writerow(
            [
                "source_wav",
                "segment_count",
                "silence_start_ms",
                "silence_end_ms",
                "silence_duration_ms",
                "is_long_silence",
            ]
        )
        review_writer.writerow(
            [
                "source_wav",
                "source_txt",
                "segment_count",
                "reason",
                "silence_start_ms",
                "silence_end_ms",
                "silence_duration_ms",
                "pronunciation_tagged",
                "spoken_text",
            ]
        )

        for source_wav in wav_files:
            source_txt = source_wav.with_suffix(".txt")
            if not source_txt.exists():
                continue

            entries = parse_transcript_file(source_txt, args.text_encoding)
            if not entries:
                continue

            params, frame_bytes = read_wave_bytes(source_wav)
            silences = detect_silences(
                frame_bytes=frame_bytes,
                params=params,
                frame_ms=args.frame_ms,
                min_silence_ms=args.min_silence_ms,
                silence_ratio=args.silence_ratio,
            )
            plans, split_method = plan_segments(
                entries=entries,
                params=params,
                silences=silences,
                min_segment_ms=args.min_segment_ms,
            )

            style_name = source_wav.parent.name
            relative_style_dir = Path(style_name)

            for silence in silences:
                silence_writer.writerow(
                    [
                        source_wav.relative_to(input_dir).as_posix(),
                        len(entries),
                        silence.start_ms,
                        silence.end_ms,
                        silence.duration_ms,
                        int(silence.duration_ms >= args.long_silence_ms),
                    ]
                )

            if len(entries) == 1:
                for silence in silences:
                    if silence.duration_ms < args.long_silence_ms:
                        continue
                    if silence.start_ms <= plans[0].start_ms or silence.end_ms >= plans[0].end_ms:
                        continue
                    review_writer.writerow(
                        [
                            source_wav.relative_to(input_dir).as_posix(),
                            source_txt.relative_to(input_dir).as_posix(),
                            len(entries),
                            "single_segment_internal_long_silence",
                            silence.start_ms,
                            silence.end_ms,
                            silence.duration_ms,
                            entries[0].pronunciation_tagged,
                            entries[0].spoken_text,
                        ]
                    )

            if len(entries) != len(plans):
                raise ValueError(
                    f"Segment planning failed for {source_wav}: entries={len(entries)} plans={len(plans)}"
                )

            for index, (entry, plan) in enumerate(zip(entries, plans), start=1):
                output_name = build_output_name(source_wav, index, len(entries))
                output_wav = raw_dir / relative_style_dir / output_name
                segment_bytes = pcm_slice(frame_bytes, params, plan.start_ms, plan.end_ms)
                write_wave_segment(output_wav, params, segment_bytes)
                relative_output = output_wav.relative_to(raw_dir).as_posix()
                esd_file.write(
                    f"{relative_output}|{speaker_name}|{args.language}|{entry.spoken_text}\n"
                )
                manifest_writer.writerow(
                    [
                        source_wav.relative_to(input_dir).as_posix(),
                        source_txt.relative_to(input_dir).as_posix(),
                        style_name,
                        index,
                        plan.start_ms,
                        plan.end_ms,
                        split_method,
                        entry.surface_text,
                        entry.pronunciation_tagged,
                        entry.spoken_text,
                        entry.tag_summary,
                        relative_output,
                    ]
                )
                for silence in silences:
                    if silence.duration_ms < args.long_silence_ms:
                        continue
                    if silence.start_ms <= plan.start_ms or silence.end_ms >= plan.end_ms:
                        continue
                    review_writer.writerow(
                        [
                            source_wav.relative_to(input_dir).as_posix(),
                            source_txt.relative_to(input_dir).as_posix(),
                            len(entries),
                            "residual_internal_long_silence",
                            silence.start_ms,
                            silence.end_ms,
                            silence.duration_ms,
                            entry.pronunciation_tagged,
                            entry.spoken_text,
                        ]
                    )
                written_rows += 1

    print(f"model_name={model_name}")
    print(f"speaker_name={speaker_name}")
    print(f"written_rows={written_rows}")
    print(f"dataset={model_dir}")
    print(f"esd={esd_path}")
    print(f"manifest={manifest_path}")
    print(f"silence_report={silence_report_path}")
    print(f"review_report={review_report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())