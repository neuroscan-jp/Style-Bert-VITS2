import argparse
import csv
import importlib.util
import importlib
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf
from tqdm import tqdm


DEFAULT_INITIAL_PROMPT = "こんにちは。元気、ですかー？ふふっ、私は……ちゃんと元気だよ！"
LOGGER = logging.getLogger("transcribe_folder")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe all audio files under a folder and save text files."
    )
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--manifest_path", type=Path)
    parser.add_argument("--extensions", nargs="+", default=[".wav"])
    parser.add_argument("--language", choices=["ja", "en", "zh"], default="ja")
    parser.add_argument("--initial_prompt", type=str, default=DEFAULT_INITIAL_PROMPT)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--engine",
        choices=["auto", "faster-whisper", "hf-whisper"],
        default="auto",
    )
    parser.add_argument("--model", type=str, default="large-v3")
    parser.add_argument("--compute_type", type=str, default="auto")
    parser.add_argument("--hf_repo_id", type=str, default="openai/whisper-large-v3-turbo")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=10)
    parser.add_argument(
        "--output_suffix",
        type=str,
        default=".whisper.txt",
        help="Used only when --output_dir is not specified.",
    )
    parser.add_argument("--encoding", type=str, default="utf-8")
    parser.add_argument("--reference_encoding", type=str, default="cp932")
    parser.add_argument(
        "--ng_list_path",
        type=Path,
        help="Path to write mismatched comparison results as TSV.",
    )
    parser.add_argument(
        "--skip_comparison",
        action="store_true",
        help="Skip comparing generated transcripts with sibling reference .txt files.",
    )
    parser.add_argument(
        "--keep_transcript_text",
        action="store_true",
        help="When comparison finds mismatches, keep the generated transcript text instead of replacing it with the reference text.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def is_module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if is_module_available("torch"):
        import torch

        if torch.cuda.is_available():
            return "cuda"
    return "cpu"


def resolve_engine(engine: str) -> str:
    if engine != "auto":
        return engine
    if is_module_available("faster_whisper"):
        return "faster-whisper"
    return "hf-whisper"


def collect_audio_files(input_dir: Path, extensions: list[str]) -> list[Path]:
    normalized_extensions = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    files = [
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in normalized_extensions
    ]
    return sorted(files)


def build_output_path(
    audio_file: Path,
    input_dir: Path,
    output_dir: Optional[Path],
    output_suffix: str,
) -> Path:
    relative_path = audio_file.relative_to(input_dir)
    if output_dir is not None:
        return output_dir / relative_path.with_suffix(".txt")
    return audio_file.with_suffix(output_suffix)


def write_manifest_row(
    writer: Optional[Any],
    audio_file: Path,
    text_file: Path,
    text: str,
    input_dir: Path,
) -> None:
    if writer is None:
        return
    writer.writerow(
        [
            audio_file.relative_to(input_dir).as_posix(),
            text_file.relative_to(input_dir.parent).as_posix(),
            text,
        ]
    )


def rebuild_manifest(
    manifest_path: Path,
    input_dir: Path,
    output_paths: dict[Path, Path],
    encoding: str,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding=encoding, newline="") as manifest_file:
        writer = csv.writer(manifest_file, delimiter="\t")
        writer.writerow(["audio_path", "text_path", "text"])
        for audio_file, text_file in sorted(output_paths.items(), key=lambda item: str(item[0])):
            if not text_file.exists():
                continue
            text = text_file.read_text(encoding=encoding).strip()
            write_manifest_row(writer, audio_file, text_file, text, input_dir)


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(normalized.split())


def strip_reference_annotations(text: str) -> str:
    text = re.sub(r"\(([A-Z]+)\s*([^)]*)\)", lambda match: match.group(2), text)
    text = re.sub(r"<([A-Z]+)\s*([^>]*)>", lambda match: match.group(2), text)
    return text


def extract_reference_text(reference_path: Path, encoding: str) -> str:
    content = reference_path.read_text(encoding=encoding).strip()
    if "&" in content:
        _, right = content.split("&", maxsplit=1)
        content = right.strip()
    return strip_reference_annotations(content).strip()


def compare_with_reference(
    input_dir: Path,
    output_paths: dict[Path, Path],
    transcript_encoding: str,
    reference_encoding: str,
    ng_list_path: Path,
    prefer_reference_text: bool,
) -> tuple[int, int]:
    ng_list_path.parent.mkdir(parents=True, exist_ok=True)
    checked_count = 0
    mismatch_count = 0

    with ng_list_path.open("w", encoding="utf-8", newline="") as report_file:
        writer = csv.writer(report_file, delimiter="\t")
        writer.writerow(
            [
                "audio_path",
                "reference_path",
                "transcript_path",
                "reference_text",
                "transcript_text",
                "reason",
            ]
        )

        for audio_file, transcript_path in sorted(output_paths.items(), key=lambda item: str(item[0])):
            reference_path = audio_file.with_suffix(".txt")
            if not transcript_path.exists() or not reference_path.exists():
                continue

            checked_count += 1
            reference_text = extract_reference_text(reference_path, reference_encoding)
            transcript_text = transcript_path.read_text(encoding=transcript_encoding).strip()

            if normalize_text(reference_text) == normalize_text(transcript_text):
                continue

            mismatch_count += 1
            writer.writerow(
                [
                    audio_file.relative_to(input_dir).as_posix(),
                    reference_path.relative_to(input_dir).as_posix(),
                    transcript_path.relative_to(input_dir).as_posix(),
                    reference_text,
                    transcript_text,
                    "text_mismatch",
                ]
            )
            if prefer_reference_text:
                transcript_path.write_text(reference_text + "\n", encoding=transcript_encoding)

    return checked_count, mismatch_count


def transcribe_with_faster_whisper(
    audio_files: list[Path],
    input_dir: Path,
    output_paths: dict[Path, Path],
    manifest_writer: Optional[Any],
    language: str,
    initial_prompt: Optional[str],
    device: str,
    model_name: str,
    compute_type: str,
    num_beams: int,
    no_repeat_ngram_size: int,
    encoding: str,
) -> None:
    WhisperModel = importlib.import_module("faster_whisper").WhisperModel

    LOGGER.info("Loading faster-whisper model: %s", model_name)
    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
    except ValueError:
        LOGGER.warning("Failed to use compute_type=%s, falling back to default.", compute_type)
        model = WhisperModel(model_name, device=device)

    for audio_file in tqdm(audio_files, dynamic_ncols=True):
        segments, _ = model.transcribe(
            str(audio_file),
            beam_size=num_beams,
            language=language,
            initial_prompt=initial_prompt,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        text = "".join(segment.text for segment in segments).strip()
        output_path = output_paths[audio_file]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding=encoding)
        write_manifest_row(manifest_writer, audio_file, output_path, text, input_dir)


def transcribe_with_hf_whisper(
    audio_files: list[Path],
    input_dir: Path,
    output_paths: dict[Path, Path],
    manifest_writer: Optional[Any],
    language: str,
    initial_prompt: Optional[str],
    device: str,
    model_id: str,
    batch_size: int,
    num_beams: int,
    no_repeat_ngram_size: int,
    encoding: str,
) -> None:
    import torch
    from transformers import WhisperProcessor, pipeline

    LOGGER.info("Loading Hugging Face Whisper model: %s", model_id)
    processor: Any = WhisperProcessor.from_pretrained(model_id)
    target_device = 0 if device == "cuda" else -1
    pipe: Any = pipeline(
        task="automatic-speech-recognition",
        model=model_id,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=batch_size,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device=target_device,
        trust_remote_code=True,
    )

    generate_kwargs = {
        "language": language,
        "do_sample": False,
        "num_beams": num_beams,
        "no_repeat_ngram_size": no_repeat_ngram_size,
    }
    if initial_prompt:
        prompt_ids = pipe.tokenizer.get_prompt_ids(initial_prompt, return_tensors="pt")
        if device == "cuda":
            prompt_ids = prompt_ids.to("cuda")
        generate_kwargs["prompt_ids"] = prompt_ids

    def load_audio_input(audio_file: Path) -> dict[str, Any]:
        audio_array, sample_rate = sf.read(audio_file)
        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=1)
        return {
            "array": np.asarray(audio_array, dtype=np.float32),
            "sampling_rate": sample_rate,
        }

    file_list: Any = [load_audio_input(audio_file) for audio_file in audio_files]
    for audio_file, result in zip(
        audio_files,
        tqdm(pipe(file_list, generate_kwargs=generate_kwargs), total=len(audio_files), dynamic_ncols=True),
    ):
        text = result["text"].strip()
        if initial_prompt and text.startswith(f" {initial_prompt}"):
            text = text[len(f" {initial_prompt}") :].lstrip()
        output_path = output_paths[audio_file]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding=encoding)
        write_manifest_row(manifest_writer, audio_file, output_path, text, input_dir)


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    input_dir = args.input_dir.resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    audio_files = collect_audio_files(input_dir, args.extensions)
    if not audio_files:
        LOGGER.error("No audio files found under %s", input_dir)
        return 1

    output_dir = args.output_dir.resolve() if args.output_dir else None
    output_paths = {
        audio_file: build_output_path(audio_file, input_dir, output_dir, args.output_suffix)
        for audio_file in audio_files
    }
    pending_audio_files = audio_files
    if not args.overwrite:
        pending_audio_files = [audio_file for audio_file in audio_files if not output_paths[audio_file].exists()]

    manifest_file = None
    manifest_writer = None
    if args.manifest_path is not None:
        manifest_path = args.manifest_path.resolve()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_file = manifest_path.open("w", encoding=args.encoding, newline="")
        manifest_writer = csv.writer(manifest_file, delimiter="\t")
        manifest_writer.writerow(["audio_path", "text_path", "text"])

    device = resolve_device(args.device)
    engine = resolve_engine(args.engine)
    LOGGER.info("Found %d audio files", len(audio_files))
    LOGGER.info("Need to transcribe %d audio files", len(pending_audio_files))
    LOGGER.info("Using device=%s engine=%s", device, engine)

    try:
        if pending_audio_files:
            if engine == "faster-whisper":
                transcribe_with_faster_whisper(
                    audio_files=pending_audio_files,
                    input_dir=input_dir,
                    output_paths=output_paths,
                    manifest_writer=manifest_writer,
                    language=args.language,
                    initial_prompt=args.initial_prompt,
                    device=device,
                    model_name=args.model,
                    compute_type=args.compute_type,
                    num_beams=args.num_beams,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    encoding=args.encoding,
                )
            else:
                transcribe_with_hf_whisper(
                    audio_files=pending_audio_files,
                    input_dir=input_dir,
                    output_paths=output_paths,
                    manifest_writer=manifest_writer,
                    language=args.language,
                    initial_prompt=args.initial_prompt,
                    device=device,
                    model_id=args.hf_repo_id,
                    batch_size=args.batch_size,
                    num_beams=args.num_beams,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    encoding=args.encoding,
                )
        else:
            LOGGER.info("All transcript output files already exist. Skipping transcription step.")
    finally:
        if manifest_file is not None:
            manifest_file.close()

    if not args.skip_comparison:
        ng_list_path = args.ng_list_path or (input_dir / "ng_list.tsv")
        prefer_reference_text = not args.keep_transcript_text
        checked_count, mismatch_count = compare_with_reference(
            input_dir=input_dir,
            output_paths=output_paths,
            transcript_encoding=args.encoding,
            reference_encoding=args.reference_encoding,
            ng_list_path=ng_list_path,
            prefer_reference_text=prefer_reference_text,
        )
        LOGGER.info(
            "Comparison completed. checked=%d mismatches=%d report=%s",
            checked_count,
            mismatch_count,
            ng_list_path,
        )

    if args.manifest_path is not None:
        rebuild_manifest(
            manifest_path=args.manifest_path.resolve(),
            input_dir=input_dir,
            output_paths=output_paths,
            encoding=args.encoding,
        )

    LOGGER.info("Transcription completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())