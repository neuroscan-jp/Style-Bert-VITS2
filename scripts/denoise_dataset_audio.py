from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


def highpass_filter(audio: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    if cutoff_hz <= 0:
        return audio
    sos = signal.butter(2, cutoff_hz, btype="highpass", fs=sample_rate, output="sos")
    return signal.sosfiltfilt(sos, audio).astype(np.float32)


def spectral_denoise(
    audio: np.ndarray,
    sample_rate: int,
    noise_quantile: float,
    reduction_strength: float,
    floor_ratio: float,
) -> np.ndarray:
    nperseg = min(1024, max(256, 2 ** int(np.floor(np.log2(len(audio) // 8 or 256)))))
    noverlap = nperseg // 4 * 3
    _, _, zxx = signal.stft(
        audio,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary="zeros",
        padded=True,
    )
    magnitude = np.abs(zxx)
    phase = np.angle(zxx)

    frame_energy = np.sqrt(np.mean(magnitude**2, axis=0))
    threshold = np.quantile(frame_energy, noise_quantile)
    noise_frames = frame_energy <= threshold
    if not np.any(noise_frames):
        noise_frames = np.ones_like(frame_energy, dtype=bool)

    noise_profile = np.median(magnitude[:, noise_frames], axis=1, keepdims=True)
    reduced = magnitude - reduction_strength * noise_profile
    reduced = np.maximum(reduced, magnitude * floor_ratio)
    cleaned = reduced * np.exp(1j * phase)

    _, restored = signal.istft(
        cleaned,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        input_onesided=True,
        boundary=True,
    )
    return restored[: len(audio)].astype(np.float32)


def denoise_audio(
    audio: np.ndarray,
    sample_rate: int,
    cutoff_hz: float,
    noise_quantile: float,
    reduction_strength: float,
    floor_ratio: float,
) -> np.ndarray:
    if audio.ndim == 1:
        audio = audio[:, None]

    cleaned_channels: list[np.ndarray] = []
    for channel in range(audio.shape[1]):
        channel_audio = audio[:, channel].astype(np.float32)
        filtered = highpass_filter(channel_audio, sample_rate, cutoff_hz)
        cleaned = spectral_denoise(
            filtered,
            sample_rate,
            noise_quantile=noise_quantile,
            reduction_strength=reduction_strength,
            floor_ratio=floor_ratio,
        )
        peak = np.max(np.abs(cleaned))
        if peak > 0.999:
            cleaned = cleaned / peak * 0.999
        cleaned_channels.append(cleaned)

    stacked = np.stack(cleaned_channels, axis=1)
    if stacked.shape[1] == 1:
        return stacked[:, 0]
    return stacked


def backup_directory(src: Path, backup: Path) -> None:
    if backup.exists():
        return
    shutil.copytree(src, backup)


def process_directory(
    input_dir: Path,
    backup_dir: Path | None,
    cutoff_hz: float,
    noise_quantile: float,
    reduction_strength: float,
    floor_ratio: float,
) -> int:
    wav_files = sorted(input_dir.rglob("*.wav"))
    if backup_dir is not None:
        backup_directory(input_dir, backup_dir)

    processed = 0
    for wav_path in wav_files:
        audio, sample_rate = sf.read(wav_path, always_2d=False)
        cleaned = denoise_audio(
            audio,
            sample_rate,
            cutoff_hz=cutoff_hz,
            noise_quantile=noise_quantile,
            reduction_strength=reduction_strength,
            floor_ratio=floor_ratio,
        )
        sf.write(wav_path, cleaned, sample_rate, subtype="PCM_16")
        processed += 1
    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description="Denoise dataset wav files in-place with backup.")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--backup_dir", type=Path, default=None)
    parser.add_argument("--cutoff_hz", type=float, default=80.0)
    parser.add_argument("--noise_quantile", type=float, default=0.2)
    parser.add_argument("--reduction_strength", type=float, default=1.25)
    parser.add_argument("--floor_ratio", type=float, default=0.08)
    args = parser.parse_args()

    processed = process_directory(
        input_dir=args.input_dir,
        backup_dir=args.backup_dir,
        cutoff_hz=args.cutoff_hz,
        noise_quantile=args.noise_quantile,
        reduction_strength=args.reduction_strength,
        floor_ratio=args.floor_ratio,
    )
    print(f"processed={processed}")


if __name__ == "__main__":
    main()