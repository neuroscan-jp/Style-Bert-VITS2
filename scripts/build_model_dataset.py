import argparse
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a single Style-Bert-VITS2 dataset from a corpus subgroup."
    )
    parser.add_argument("--corpus_root", type=Path, required=True)
    parser.add_argument("--source_esd", type=Path, required=True)
    parser.add_argument("--group_path", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=Path("Data"))
    parser.add_argument("--model_name", type=str)
    parser.add_argument(
        "--copy_mode",
        choices=["hardlink", "copy"],
        default="hardlink",
        help="How to place audio files into Data/{model_name}/raw.",
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


def flatten_relative_path(relative_path: Path) -> str:
    stem_parts = list(relative_path.parts[:-1]) + [relative_path.stem]
    return "__".join(stem_parts) + ".wav"


def main() -> int:
    args = parse_args()

    corpus_root = args.corpus_root.resolve()
    source_esd = args.source_esd.resolve()
    dataset_root = args.dataset_root.resolve()
    group_path = Path(args.group_path.as_posix().strip("/"))
    model_name = args.model_name or group_path.name

    model_dir = dataset_root / model_name
    raw_dir = model_dir / "raw"
    esd_path = model_dir / "esd.list"
    source_map_path = model_dir / "source_map.tsv"

    model_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    prefix = group_path.as_posix().rstrip("/") + "/"
    written_count = 0

    with source_esd.open("r", encoding="utf-8") as src, esd_path.open(
        "w", encoding="utf-8", newline=""
    ) as dst, source_map_path.open("w", encoding="utf-8", newline="") as map_file:
        map_file.write("source_wav\ttarget_wav\n")

        for line in src:
            line = line.strip()
            if not line:
                continue
            audio_path, speaker_name, language_id, text = line.split("|", maxsplit=3)
            if not audio_path.startswith(prefix):
                continue

            source_wav = corpus_root / Path(audio_path)
            subgroup_relative_path = Path(audio_path).relative_to(group_path)
            target_name = flatten_relative_path(subgroup_relative_path)
            target_wav = raw_dir / target_name

            safe_link_or_copy(source_wav, target_wav, args.copy_mode)
            dst.write(f"{target_name}|{model_name}|{language_id}|{text}\n")
            map_file.write(f"{audio_path}\traw/{target_name}\n")
            written_count += 1

    print(f"model_name={model_name}")
    print(f"written={written_count}")
    print(f"dataset={model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())