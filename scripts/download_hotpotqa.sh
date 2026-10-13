#!/usr/bin/env bash
# Downloads the HotpotQA dev set (distractor setting) from the HuggingFace mirror.
#
# Distractor rather than fullwiki: each question ships with its 2 gold paragraphs
# plus 8 retrieved distractors, which is a bounded corpus we can treat as memory.
# Fullwiki would make this a web-scale retrieval problem instead of a memory one.
#
# The original CMU host (curtis.ml.cmu.edu) is frequently unavailable; the HF
# mirror carries the same data as parquet.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

URL="https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet"
OUT_DIR="$REPO_ROOT/data/raw/hotpotqa"
OUT_FILE="$OUT_DIR/validation_distractor.parquet"

mkdir -p "$OUT_DIR"

echo "Downloading HotpotQA dev (distractor) to $OUT_FILE ..."
curl -fSL "$URL" -o "$OUT_FILE"

SIZE=$(wc -c < "$OUT_FILE")
echo "Done. Downloaded $SIZE bytes."
echo "HotpotQA is CC BY-SA 4.0 — see https://hotpotqa.github.io/ before redistributing."
