import argparse
import csv
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a filtered training dataset from CIAIR preprocessing artifacts."
    )
    parser.add_argument("--source_dataset", type=Path, required=True)
    parser.add_argument("--output_dataset", type=Path, required=True)
    parser.add_argument("--copy_mode", choices=["hardlink", "copy"], default="hardlink")
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


def has_angle_tag(tag_summary: str) -> bool:
    return "<" in tag_summary and ">" in tag_summary


def load_long_silence_sources(silence_report_path: Path) -> set[str]:
    long_silence_sources: set[str] = set()
    with silence_report_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row["is_long_silence"] == "1":
                long_silence_sources.add(row["source_wav"])
    return long_silence_sources


def main() -> int:
    args = parse_args()
    source_dataset = args.source_dataset.resolve()
    output_dataset = args.output_dataset.resolve()

    manifest_path = source_dataset / "ciair_manifest.tsv"
    silence_report_path = source_dataset / "silence_report.tsv"
    source_raw_dir = source_dataset / "raw"
    output_raw_dir = output_dataset / "raw"
    output_esd_path = output_dataset / "esd.list"
    output_manifest_path = output_dataset / "ciair_manifest.tsv"
    exclusion_report_path = output_dataset / "excluded_entries.tsv"

    if output_dataset.exists():
        shutil.rmtree(output_dataset)
    output_raw_dir.mkdir(parents=True, exist_ok=True)

    long_silence_sources = load_long_silence_sources(silence_report_path)

    kept_rows = 0
    excluded_rows = 0
    excluded_source_count = 0
    excluded_sources_seen: set[str] = set()

    with (
        manifest_path.open("r", encoding="utf-8", newline="") as manifest_file,
        output_esd_path.open("w", encoding="utf-8", newline="") as esd_file,
        output_manifest_path.open("w", encoding="utf-8", newline="") as output_manifest_file,
        exclusion_report_path.open("w", encoding="utf-8", newline="") as exclusion_file,
    ):
        reader = csv.DictReader(manifest_file, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        manifest_writer = csv.DictWriter(output_manifest_file, fieldnames=fieldnames, delimiter="\t")
        exclusion_writer = csv.writer(exclusion_file, delimiter="\t")

        manifest_writer.writeheader()
        exclusion_writer.writerow(
            [
                "source_wav",
                "output_wav",
                "pronunciation_tagged",
                "spoken_text",
                "tag_summary",
                "reason",
            ]
        )

        for row in reader:
            source_wav = row["source_wav"]
            output_wav = row["output_wav"]
            tag_summary = row["tag_summary"]

            exclusion_reason = None
            if source_wav in long_silence_sources:
                exclusion_reason = "source_has_long_silence"
                if source_wav not in excluded_sources_seen:
                    excluded_sources_seen.add(source_wav)
                    excluded_source_count += 1
            elif has_angle_tag(tag_summary):
                exclusion_reason = "contains_type2_angle_tag"

            if exclusion_reason is not None:
                exclusion_writer.writerow(
                    [
                        source_wav,
                        output_wav,
                        row["pronunciation_tagged"],
                        row["spoken_text"],
                        tag_summary,
                        exclusion_reason,
                    ]
                )
                excluded_rows += 1
                continue

            source_wav_path = source_raw_dir / Path(output_wav)
            target_wav_path = output_raw_dir / Path(output_wav)
            safe_link_or_copy(source_wav_path, target_wav_path, args.copy_mode)

            esd_file.write(
                f"{output_wav}|{source_dataset.name}|JP|{row['spoken_text']}\n"
            )
            manifest_writer.writerow(row)
            kept_rows += 1

    print(f"source_dataset={source_dataset}")
    print(f"output_dataset={output_dataset}")
    print(f"kept_rows={kept_rows}")
    print(f"excluded_rows={excluded_rows}")
    print(f"excluded_source_wavs={excluded_source_count}")
    print(f"esd={output_esd_path}")
    print(f"manifest={output_manifest_path}")
    print(f"excluded_report={exclusion_report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())