import argparse
import csv
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a training dataset from audio-only preprocessing outputs."
    )
    parser.add_argument("--source_dataset", type=Path, required=True)
    parser.add_argument("--output_dataset", type=Path, required=True)
    parser.add_argument("--speaker_name", type=str, required=True)
    parser.add_argument("--copy_mode", choices=["hardlink", "copy"], default="hardlink")
    parser.add_argument("--language", type=str, default="JP")
    parser.add_argument(
        "--exclude_audio",
        action="append",
        default=[],
        help="Relative audio path from transcriptions.tsv to exclude. Can be passed multiple times.",
    )
    return parser.parse_args()


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


def main() -> int:
    args = parse_args()
    source_dataset = args.source_dataset.resolve()
    output_dataset = args.output_dataset.resolve()
    raw_source_dir = source_dataset / "raw"
    transcripts_path = source_dataset / "transcriptions.tsv"
    output_raw_dir = output_dataset / "raw"
    output_esd_path = output_dataset / "esd.list"
    output_report_path = output_dataset / "excluded_entries.tsv"

    if output_dataset.exists():
        shutil.rmtree(output_dataset)
    output_raw_dir.mkdir(parents=True, exist_ok=True)

    exclude_audio = set(args.exclude_audio)
    kept_count = 0
    excluded_count = 0

    with (
        transcripts_path.open("r", encoding="utf-8", newline="") as transcripts_file,
        output_esd_path.open("w", encoding="utf-8", newline="") as esd_file,
        output_report_path.open("w", encoding="utf-8", newline="") as report_file,
    ):
        reader = csv.DictReader(transcripts_file, delimiter="\t")
        report_writer = csv.writer(report_file, delimiter="\t")
        report_writer.writerow(["audio_path", "text", "reason"])

        for row in reader:
            audio_path = row["audio_path"]
            text = row["text"].strip()
            if audio_path in exclude_audio:
                report_writer.writerow([audio_path, text, "manual_exclusion"])
                excluded_count += 1
                continue
            if not text:
                report_writer.writerow([audio_path, text, "empty_text"])
                excluded_count += 1
                continue

            source_wav = raw_source_dir / Path(audio_path)
            target_wav = output_raw_dir / Path(audio_path)
            safe_link_or_copy(source_wav, target_wav, args.copy_mode)
            esd_file.write(f"{audio_path}|{args.speaker_name}|{args.language}|{text}\n")
            kept_count += 1

    print(f"source_dataset={source_dataset}")
    print(f"output_dataset={output_dataset}")
    print(f"kept_entries={kept_count}")
    print(f"excluded_entries={excluded_count}")
    print(f"esd={output_esd_path}")
    print(f"excluded_report={output_report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())