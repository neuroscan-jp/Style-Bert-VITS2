from __future__ import annotations

import argparse
from pathlib import Path

from scipy.io import wavfile

from style_bert_vits2.constants import Languages
from style_bert_vits2.tts_model import TTSModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model = TTSModel(
        model_path=model_dir / args.model_file,
        config_path=model_dir / "config.json",
        style_vec_path=model_dir / "style_vectors.npy",
        device=args.device,
    )

    output_dir = model_dir / Path(args.model_file).stem.replace(".safetensors", "")
    output_dir = model_dir / f"samples_{Path(args.model_file).stem}"
    output_dir.mkdir(parents=True, exist_ok=True)

    texts = {
        "sample_01": "そのとき、まちのひがしに小さな音が聞こえた。",
        "sample_02": "きょうは落ち着いた調子で、文章を自然に読み上げます。",
        "sample_03": "短い単語だけでなく、少し長い文でも発音を確認します。",
    }

    for name, text in texts.items():
        sample_rate, audio = model.infer(
            text=text,
            language=Languages.JP,
            speaker_id=0,
            style="Neutral",
            style_weight=1.0,
            sdp_ratio=0.2,
            noise=0.6,
            noise_w=0.8,
            length=1.0,
            line_split=False,
        )
        output_path = output_dir / f"{name}.wav"
        wavfile.write(output_path, sample_rate, audio)
        print(output_path)


if __name__ == "__main__":
    main()
