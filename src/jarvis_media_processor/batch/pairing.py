"""Folder layout conventions and base/donor pair discovery."""

from __future__ import annotations

import re
from pathlib import Path

from jarvis_media_processor.batch.models import MediaPair, PairLayout

# Common container / elementary media extensions used as inputs.
MEDIA_EXTENSIONS = frozenset(
    {
        ".mkv",
        ".mp4",
        ".m4v",
        ".mov",
        ".avi",
        ".webm",
        ".ts",
        ".m2ts",
        ".mpg",
        ".mpeg",
    }
)

_BASE_NAME = re.compile(r"^base$", re.IGNORECASE)
_DONOR_NAME = re.compile(r"^donor$", re.IGNORECASE)
_FLAT_BASE = re.compile(r"^(?P<stem>.+)\.base$", re.IGNORECASE)
_FLAT_DONOR = re.compile(r"^(?P<stem>.+)\.donor$", re.IGNORECASE)


def is_media_file(path: Path) -> bool:
    """True when ``path`` looks like a supported media file."""
    return path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS


def output_path_for_pair(output_dir: Path, pair_id: str) -> Path:
    """Deterministic output MKV path under ``output_dir`` for ``pair_id``.

    Nested pair ids (split-tree relative paths) keep their relative folders.
    The file name is always ``<leaf>.mkv``. Pair ids may contain dots
    (e.g. ``show.s01e01``); we append ``.mkv`` instead of ``Path.with_suffix``.
    """
    # Normalize separators; reject traversal.
    cleaned = pair_id.replace("\\", "/").strip("/")
    if not cleaned or ".." in cleaned.split("/"):
        raise ValueError(f"Invalid pair id: {pair_id!r}")
    rel = Path(cleaned)
    return Path(output_dir) / rel.parent / f"{rel.name}.mkv"


def discover_pairs(input_dir: Path, output_dir: Path) -> list[MediaPair]:
    """Discover base/donor pairs under ``input_dir``.

    Supported layouts (see ``docs/batch-workflows.md``):

    1. **subdir** — ``<input>/<pair_id>/base.<ext>`` + ``donor.<ext>``
    2. **flat** — ``<input>/<pair_id>.base.<ext>`` + ``<pair_id>.donor.<ext>``
    3. **split** — ``<input>/base/<rel>.<ext>`` + ``<input>/donor/<rel>.<ext>``

    Pair ids are unique across layouts. When the same id appears in more than
    one layout, the first win order is subdir, then flat, then split.
    Sources are never modified; outputs always resolve under ``output_dir``.
    """
    root = Path(input_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {root}")

    out_root = Path(output_dir)
    by_id: dict[str, MediaPair] = {}

    for pair in _discover_subdir_pairs(root, out_root):
        by_id.setdefault(pair.pair_id, pair)
    for pair in _discover_flat_pairs(root, out_root):
        by_id.setdefault(pair.pair_id, pair)
    for pair in _discover_split_pairs(root, out_root):
        by_id.setdefault(pair.pair_id, pair)

    return sorted(by_id.values(), key=lambda p: p.pair_id.lower())


def _discover_subdir_pairs(root: Path, out_root: Path) -> list[MediaPair]:
    pairs: list[MediaPair] = []
    for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir():
            continue
        if child.name.lower() in {"base", "donor"}:
            # Reserved for the split-tree layout.
            continue
        base = _find_role_file(child, _BASE_NAME)
        donor = _find_role_file(child, _DONOR_NAME)
        if base is None or donor is None:
            continue
        pair_id = child.name
        pairs.append(
            MediaPair(
                pair_id=pair_id,
                base_path=base,
                donor_path=donor,
                output_path=output_path_for_pair(out_root, pair_id),
                layout=PairLayout.SUBDIR,
            )
        )
    return pairs


def _discover_flat_pairs(root: Path, out_root: Path) -> list[MediaPair]:
    bases: dict[str, Path] = {}
    donors: dict[str, Path] = {}
    for child in root.iterdir():
        if not is_media_file(child):
            continue
        stem = child.stem  # e.g. "ep01.base"
        base_match = _FLAT_BASE.match(stem)
        if base_match:
            bases[base_match.group("stem")] = child
            continue
        donor_match = _FLAT_DONOR.match(stem)
        if donor_match:
            donors[donor_match.group("stem")] = child

    pairs: list[MediaPair] = []
    for pair_id in sorted(set(bases) & set(donors), key=str.lower):
        pairs.append(
            MediaPair(
                pair_id=pair_id,
                base_path=bases[pair_id],
                donor_path=donors[pair_id],
                output_path=output_path_for_pair(out_root, pair_id),
                layout=PairLayout.FLAT,
            )
        )
    return pairs


def _discover_split_pairs(root: Path, out_root: Path) -> list[MediaPair]:
    base_root = root / "base"
    donor_root = root / "donor"
    if not base_root.is_dir() or not donor_root.is_dir():
        return []

    bases = _index_media_by_rel_stem(base_root)
    donors = _index_media_by_rel_stem(donor_root)
    pairs: list[MediaPair] = []
    for pair_id in sorted(set(bases) & set(donors), key=str.lower):
        pairs.append(
            MediaPair(
                pair_id=pair_id,
                base_path=bases[pair_id],
                donor_path=donors[pair_id],
                output_path=output_path_for_pair(out_root, pair_id),
                layout=PairLayout.SPLIT,
            )
        )
    return pairs


def _index_media_by_rel_stem(tree: Path) -> dict[str, Path]:
    """Map relative stem (posix, no suffix) → media file under ``tree``."""
    indexed: dict[str, Path] = {}
    for path in sorted(tree.rglob("*")):
        if not is_media_file(path):
            continue
        rel = path.relative_to(tree)
        pair_id = rel.with_suffix("").as_posix()
        # Prefer the first extension if duplicates exist.
        indexed.setdefault(pair_id, path)
    return indexed


def _find_role_file(directory: Path, name_re: re.Pattern[str]) -> Path | None:
    matches = [
        p for p in directory.iterdir() if is_media_file(p) and name_re.match(p.stem) is not None
    ]
    if not matches:
        return None
    # Prefer .mkv when multiple extensions exist for the same role.
    matches.sort(key=lambda p: (0 if p.suffix.lower() == ".mkv" else 1, p.name.lower()))
    return matches[0]
