# 任意フォルダの文字起こし

学習用の Data 配下ではなく、任意のフォルダを直接再帰走査して文字起こししたい場合は scripts/transcribe_folder.py を使います。

## 例

f0600101 配下の wav を文字起こしして、既存の txt を壊さないように .whisper.txt を作る:

```bash
d:/Style-Bert-VITS2/.venv/Scripts/python.exe scripts/transcribe_folder.py \
  --input_dir CIAIR-VCV/f06/f06001/f0600101 \
  --language ja \
  --device cpu \
  --engine hf-whisper \
  --manifest_path CIAIR-VCV/f06/f06001/f0600101/transcriptions.tsv
```

この実行では、同じフォルダにある既存の `001.txt` のような参照ファイルとも比較し、差分があるものを `ng_list.tsv` に出力します。差分があった場合、出力される `.whisper.txt` は参照 txt 側の文を正として上書きします。

別フォルダに結果を集約したい場合:

```bash
d:/Style-Bert-VITS2/.venv/Scripts/python.exe scripts/transcribe_folder.py \
  --input_dir CIAIR-VCV/f06 \
  --output_dir outputs/ciair_f06_text \
  --language ja \
  --device cpu
```

## 主なオプション

- --input_dir: 文字起こし対象のルートフォルダ
- --output_dir: 出力先ルート。未指定時は各 wav の横に .whisper.txt を出力
- --manifest_path: まとめ用 TSV を出力
- --ng_list_path: 参照 txt と比較して差分があったものだけを出す TSV を出力。未指定時は input_dir/ng_list.tsv
- --engine: auto / faster-whisper / hf-whisper
- --device: auto / cuda / cpu
- --reference_encoding: 参照 txt の文字コード。既定は cp932
- --skip_comparison: 参照 txt との比較を無効化
- --keep_transcript_text: 差分があっても `.whisper.txt` を参照 txt で上書きしない
- --overwrite: 既存の出力を上書き

## 補足

- 既定では .wav のみ対象です。拡張子を増やす場合は --extensions .wav .mp3 のように指定します。
- 既存の 001.txt などは上書きしません。未指定時の出力名は 001.whisper.txt のようになります。