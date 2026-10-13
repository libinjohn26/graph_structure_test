#!/usr/bin/env bash
# Downloads the LoCoMo benchmark (Maharana et al., ACL 2024) from its
# source repo: https://github.com/snap-research/locomo
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

URL="https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
OUT_DIR="$REPO_ROOT/data/raw/locomo"
OUT_FILE="$OUT_DIR/locomo10.json"

mkdir -p "$OUT_DIR"

echo "Downloading LoCoMo dataset to $OUT_FILE ..."
curl -fSL "$URL" -o "$OUT_FILE"

SIZE=$(wc -c < "$OUT_FILE")
echo "Done. Downloaded $SIZE bytes."
echo "Check dataset license terms at https://github.com/snap-research/locomo/blob/main/LICENSE.txt before redistributing."
