#!/usr/bin/env python
"""Run this first on a fresh clone or a new machine.

Checks, in order:
  1. Python version is new enough.
  2. Required third-party packages import.
  3. The `gms` package itself imports.
  4. The raw LoCoMo data file is present (points at the download script if not).
  5. The loader actually loads it — not a mock — and the result looks sane.

Exits 0 and prints "All checks passed" if everything works; exits 1 with a
specific, actionable message at the first failure otherwise.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

MIN_PYTHON = (3, 10)


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    sys.exit(1)


def ok(message: str) -> None:
    print(f"[ OK ] {message}")


def check_python_version() -> None:
    if sys.version_info < MIN_PYTHON:
        fail(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required, "
            f"found {sys.version_info.major}.{sys.version_info.minor}."
        )
    ok(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")


def check_dependencies() -> None:
    required = ["yaml", "tqdm", "numpy", "sentence_transformers", "networkx", "pyarrow"]
    missing = []
    for mod in required:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        fail(f"Missing packages: {missing}. Run `pip install -r requirements.txt`.")
    ok(f"Dependencies importable: {required}")


def check_gms_package():
    try:
        import gms  # noqa: F401
        from gms.data import load_locomo
    except ImportError as e:
        fail(f"Could not import gms package: {e}")
    ok(f"gms package importable (version {gms.__version__})")
    return load_locomo


def check_data_present() -> Path:
    data_path = REPO_ROOT / "data" / "raw" / "locomo" / "locomo10.json"
    if not data_path.exists():
        fail(
            f"LoCoMo data not found at {data_path}. "
            f"Run `bash scripts/download_locomo.sh` first."
        )
    size_kb = data_path.stat().st_size / 1024
    ok(f"LoCoMo data file present ({size_kb:.0f} KB)")
    return data_path


def check_loader(load_locomo, data_path: Path) -> None:
    conversations = load_locomo(data_path)

    if not conversations:
        fail("Loader returned zero conversations — data file may be malformed.")

    n_convs = len(conversations)
    n_sessions = sum(c.num_sessions() for c in conversations)
    n_turns = sum(c.num_turns() for c in conversations)
    n_qa = sum(len(c.qa_items) for c in conversations)
    categories = Counter(qa.category for c in conversations for qa in c.qa_items)

    ok(f"Loaded {n_convs} conversations via the real loader (no mocking)")

    print("\n--- LoCoMo dataset summary ---")
    print(f"  conversations : {n_convs}")
    print(f"  sessions      : {n_sessions}  ({n_sessions / n_convs:.1f} avg/conversation)")
    print(f"  turns         : {n_turns}  ({n_turns / n_convs:.1f} avg/conversation)")
    print(f"  QA items      : {n_qa}  ({n_qa / n_convs:.1f} avg/conversation)")
    print("  QA category counts (raw labels from the dataset):")
    for cat, count in sorted(categories.items(), key=lambda kv: str(kv[0])):
        print(f"    category {cat}: {count}")

    # Soft sanity checks — warn rather than hard-fail, since these are
    # expectations about a public dataset's shape, not our own invariants.
    first = conversations[0]
    if not first.sessions:
        print("[WARN] first conversation has no sessions — check parsing logic.")
    elif not first.sessions[0].turns:
        print("[WARN] first session has no turns — check parsing logic.")
    else:
        example_turn = first.sessions[0].turns[0]
        print(f"\n  example turn: [{example_turn.speaker}] {example_turn.text[:80]!r}")

    if first.qa_items:
        example_qa = first.qa_items[0]
        print(f"  example QA  : Q: {example_qa.question[:80]!r}")
        print(f"                A: {example_qa.answer[:80]!r}")


def main() -> None:
    print("Running gms setup check...\n")
    check_python_version()
    check_dependencies()
    load_locomo = check_gms_package()
    data_path = check_data_present()
    check_loader(load_locomo, data_path)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
