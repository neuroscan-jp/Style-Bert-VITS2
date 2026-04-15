import argparse
import csv
import re
import unicodedata
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an esd.list from wav/txt pairs under a corpus root."
    )
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_path", type=Path)
    parser.add_argument("--language", type=str, default="JP", choices=["JP", "EN", "ZH"])
    parser.add_argument("--text_encoding", type=str, default="cp932")
    parser.add_argument(
        "--speaker_part_index",
        type=int,
        default=1,
        help="0-based path part index, relative to input_dir, used as speaker name.",
    )
    parser.add_argument(
        "--missing_report_path",
        type=Path,
        help="Optional TSV report path for wav files that do not have a sibling txt.",
    )
    return parser.parse_args()


def strip_reference_annotations(text: str) -> str:
    text = re.sub(r"\(([A-Z]+)\s*([^)]*)\)", lambda match: match.group(2), text)
    text = re.sub(r"<([A-Z]+)\s*([^>]*)>", lambda match: match.group(2), text)
    return text


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(normalized.split())


def extract_reference_text(reference_path: Path, encoding: str) -> str:
    content = reference_path.read_text(encoding=encoding).strip()
    if "&" in content:
        _, right = content.split("&", maxsplit=1)
        content = right.strip()
    content = strip_reference_annotations(content).strip()
    return normalize_text(content)


def get_speaker_name(relative_wav_path: Path, speaker_part_index: int) -> str:
    parts = relative_wav_path.parts
    if not parts:
        raise ValueError(f"Invalid relative path: {relative_wav_path}")
    if speaker_part_index < len(parts):
        return parts[speaker_part_index]
    return parts[0]


def main() -> int:
    args = parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_path = args.output_path.resolve() if args.output_path else input_dir / "esd.list"
    missing_report_path = (
        args.missing_report_path.resolve()
        if args.missing_report_path
        else input_dir / "esd_missing.tsv"
    )

    wav_files = sorted(path for path in input_dir.rglob("*.wav") if path.is_file())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    missing_report_path.parent.mkdir(parents=True, exist_ok=True)

    written_count = 0
    missing_count = 0

    with output_path.open("w", encoding="utf-8", newline="") as esd_file, missing_report_path.open(
        "w", encoding="utf-8", newline=""
    ) as missing_file:
        missing_writer = csv.writer(missing_file, delimiter="\t")
        missing_writer.writerow(["wav_path", "expected_txt_path", "reason"])

        for wav_path in wav_files:
            txt_path = wav_path.with_suffix(".txt")
            relative_wav_path = wav_path.relative_to(input_dir)
            if not txt_path.exists():
                missing_writer.writerow(
                    [
                        relative_wav_path.as_posix(),
                        txt_path.relative_to(input_dir).as_posix(),
                        "missing_txt",
                    ]
                )
                missing_count += 1
                continue

            speaker_name = get_speaker_name(relative_wav_path, args.speaker_part_index)
            text = extract_reference_text(txt_path, args.text_encoding)
            esd_file.write(
                f"{relative_wav_path.as_posix()}|{speaker_name}|{args.language}|{text}\n"
            )
            written_count += 1

    print(f"wrote={written_count}")
    print(f"missing={missing_count}")
    print(f"esd={output_path}")
    print(f"missing_report={missing_report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())