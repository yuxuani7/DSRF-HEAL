import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from xml.sax.saxutils import escape

from src.io_utils import ensure_dir, now_iso


HALL_TYPES = ("o1", "s", "o2", "r", "o3")
SCENE_ORDER = ("behavior", "virtualhome", "unknown")


@dataclass
class StatBucket:
    samples: int = 0
    hallucination_samples: int = 0
    counts: Dict[str, int] = field(
        default_factory=lambda: {key: 0 for key in HALL_TYPES}
    )
    denominators: Dict[str, int] = field(
        default_factory=lambda: {key: 0 for key in HALL_TYPES}
    )

    def merge(self, other: "StatBucket") -> None:
        self.samples += other.samples
        self.hallucination_samples += other.hallucination_samples
        for key in HALL_TYPES:
            self.counts[key] += other.counts[key]
            self.denominators[key] += other.denominators[key]


def summarize_validation_run(validation_run_dir: Path, results_dir: Path) -> Path:
    if not validation_run_dir.exists():
        raise FileNotFoundError(
            "Validation run dir not found: {0}".format(validation_run_dir)
        )
    if not validation_run_dir.is_dir():
        raise ValueError(
            "Validation run path is not a directory: {0}".format(validation_run_dir)
        )

    jsonl_files = sorted(validation_run_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise ValueError(
            "No jsonl files found in validation dir: {0}".format(validation_run_dir)
        )

    by_scene_category: Dict[Tuple[str, str], StatBucket] = {}
    parse_errors = 0

    for jsonl_file in jsonl_files:
        with jsonl_file.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    parse_errors += 1
                    continue

                source = record.get("source") or {}
                dataset_name = str(source.get("source_dataset") or "").strip()
                if not dataset_name:
                    dataset_name = _dataset_from_filename(jsonl_file.name)
                scene, category = _split_scene_category(dataset_name)
                key = (scene, category)
                bucket = by_scene_category.setdefault(key, StatBucket())
                _accumulate_record(bucket, record)

    rows = _build_rows(by_scene_category)
    svg_text = _build_svg_table(
        run_dir_name=validation_run_dir.name,
        rows=rows,
        parse_errors=parse_errors,
    )

    ensure_dir(results_dir)
    output_path = results_dir / "{0}.svg".format(validation_run_dir.name)
    output_path.write_text(svg_text, encoding="utf-8")
    return output_path


def _accumulate_record(bucket: StatBucket, record: Dict) -> None:
    bucket.samples += 1
    if bool(record.get("has_hallucination")):
        bucket.hallucination_samples += 1

    metrics = record.get("metrics") or {}
    for key in HALL_TYPES:
        item = metrics.get(key) or {}
        bucket.counts[key] += _to_int(item.get("count"))
        bucket.denominators[key] += _to_int(item.get("denominator"))


def _to_int(value) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _dataset_from_filename(filename: str) -> str:
    base = filename
    if base.endswith(".jsonl"):
        base = base[: -len(".jsonl")]
    # dataset__llm
    if "__" in base:
        return base.split("__", 1)[0]
    return base


def _split_scene_category(dataset_name: str) -> Tuple[str, str]:
    normalized = dataset_name.strip().lower()
    if normalized.startswith("behavior_"):
        return "behavior", normalized[len("behavior_") :]
    if normalized.startswith("virtualhome_"):
        return "virtualhome", normalized[len("virtualhome_") :]
    if "_" in normalized:
        first, rest = normalized.split("_", 1)
        return first or "unknown", rest or "unknown"
    return "unknown", normalized or "unknown"


def _build_rows(by_scene_category: Dict[Tuple[str, str], StatBucket]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    sorted_items = sorted(
        by_scene_category.items(),
        key=lambda item: (_scene_sort_key(item[0][0]), item[0][1]),
    )
    for (scene, category), bucket in sorted_items:
        rows.append(_row_from_bucket("{0}/{1}".format(scene, category), bucket))

    scene_totals: Dict[str, StatBucket] = {}
    scene_non_baseline_totals: Dict[str, StatBucket] = {}
    for (scene, _), bucket in by_scene_category.items():
        target = scene_totals.setdefault(scene, StatBucket())
        target.merge(bucket)
    for (scene, category), bucket in by_scene_category.items():
        if category == "baseline":
            continue
        target = scene_non_baseline_totals.setdefault(scene, StatBucket())
        target.merge(bucket)

    for scene in sorted(scene_totals.keys(), key=_scene_sort_key):
        non_baseline_bucket = scene_non_baseline_totals.get(scene)
        if non_baseline_bucket is not None and non_baseline_bucket.samples > 0:
            rows.append(
                _row_from_bucket(
                    "{0}/ALL_EXCEPT_BASELINE".format(scene),
                    non_baseline_bucket,
                    True,
                )
            )
        rows.append(_row_from_bucket("{0}/ALL".format(scene), scene_totals[scene], True))

    overall = StatBucket()
    for bucket in by_scene_category.values():
        overall.merge(bucket)
    rows.append(_row_from_bucket("ALL/ALL", overall, True))
    return rows


def _scene_sort_key(scene: str) -> int:
    if scene in SCENE_ORDER:
        return SCENE_ORDER.index(scene)
    return len(SCENE_ORDER) + 1


def _row_from_bucket(scope: str, bucket: StatBucket, highlight: bool = False) -> Dict[str, str]:
    return {
        "scope": scope,
        "samples": str(bucket.samples),
        "hall_rate": _format_rate(bucket.hallucination_samples, bucket.samples),
        "o1_rate": _format_rate(bucket.counts["o1"], bucket.denominators["o1"]),
        "s_rate": _format_rate(bucket.counts["s"], bucket.denominators["s"]),
        "o2_rate": _format_rate(bucket.counts["o2"], bucket.denominators["o2"]),
        "r_rate": _format_rate(bucket.counts["r"], bucket.denominators["r"]),
        "o3_rate": _format_rate(bucket.counts["o3"], bucket.denominators["o3"]),
        "is_highlight": "1" if highlight else "0",
    }


def _format_rate(count: int, denominator: int) -> str:
    if denominator <= 0:
        return "N/A (0/0)"
    ratio = float(count) / float(denominator)
    return "{0:.2f}% ({1}/{2})".format(ratio * 100.0, count, denominator)


def _build_svg_table(run_dir_name: str, rows: List[Dict[str, str]], parse_errors: int) -> str:
    run_ts, model_name = _parse_run_dir_name(run_dir_name)
    title = "Hallucination Summary | model={0} | run={1}".format(model_name, run_ts)
    subtitle = "Generated at {0} | parse_errors={1}".format(now_iso(), parse_errors)

    columns = [
        ("Category", "scope", 320),
        ("Samples", "samples", 90),
        ("Total Hall Rate", "hall_rate", 170),
        ("O1 Rate", "o1_rate", 150),
        ("S Rate", "s_rate", 150),
        ("O2 Rate", "o2_rate", 150),
        ("R Rate", "r_rate", 150),
        ("O3 Rate", "o3_rate", 150),
    ]

    left_margin = 20
    top_margin = 20
    title_height = 54
    row_height = 34
    table_width = sum(item[2] for item in columns)
    table_height = row_height * (1 + len(rows))
    width = left_margin * 2 + table_width
    height = top_margin + title_height + table_height + 20

    lines: List[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append(
        '<svg xmlns="http://www.w3.org/2000/svg" width="{0}" height="{1}" viewBox="0 0 {0} {1}">'.format(
            width, height
        )
    )
    lines.append('<rect x="0" y="0" width="{0}" height="{1}" fill="#ffffff"/>'.format(width, height))

    lines.append(
        '<text x="{0}" y="{1}" font-family="Arial, Helvetica, sans-serif" font-size="20" font-weight="700" fill="#111111">{2}</text>'.format(
            left_margin,
            top_margin + 24,
            escape(title),
        )
    )
    lines.append(
        '<text x="{0}" y="{1}" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="#555555">{2}</text>'.format(
            left_margin,
            top_margin + 44,
            escape(subtitle),
        )
    )

    table_x = left_margin
    table_y = top_margin + title_height

    # Header background
    lines.append(
        '<rect x="{0}" y="{1}" width="{2}" height="{3}" fill="#f0f4f8" stroke="#9aa5b1" stroke-width="1"/>'.format(
            table_x, table_y, table_width, row_height
        )
    )

    # Header text + vertical lines
    x_cursor = table_x
    for col_name, _, col_width in columns:
        lines.append(
            '<line x1="{0}" y1="{1}" x2="{0}" y2="{2}" stroke="#9aa5b1" stroke-width="1"/>'.format(
                x_cursor,
                table_y,
                table_y + table_height,
            )
        )
        lines.append(
            '<text x="{0}" y="{1}" font-family="Arial, Helvetica, sans-serif" font-size="13" font-weight="700" fill="#222222">{2}</text>'.format(
                x_cursor + 8,
                table_y + 22,
                escape(col_name),
            )
        )
        x_cursor += col_width
    lines.append(
        '<line x1="{0}" y1="{1}" x2="{0}" y2="{2}" stroke="#9aa5b1" stroke-width="1"/>'.format(
            table_x + table_width,
            table_y,
            table_y + table_height,
        )
    )

    # Data rows
    for row_idx, row in enumerate(rows):
        y = table_y + row_height * (row_idx + 1)
        is_highlight = row.get("is_highlight") == "1"
        bg = "#f8fafc" if row_idx % 2 == 0 else "#ffffff"
        if is_highlight:
            bg = "#eef6ff"
        lines.append(
            '<rect x="{0}" y="{1}" width="{2}" height="{3}" fill="{4}" stroke="#d2d6dc" stroke-width="1"/>'.format(
                table_x,
                y,
                table_width,
                row_height,
                bg,
            )
        )
        x_cursor = table_x
        for _, key, col_width in columns:
            raw = str(row.get(key, ""))
            text = _truncate(raw, 38 if key == "scope" else 20)
            lines.append(
                '<text x="{0}" y="{1}" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#1f2933">{2}</text>'.format(
                    x_cursor + 8,
                    y + 22,
                    escape(text),
                )
            )
            x_cursor += col_width

    # Outer border
    lines.append(
        '<rect x="{0}" y="{1}" width="{2}" height="{3}" fill="none" stroke="#9aa5b1" stroke-width="1"/>'.format(
            table_x,
            table_y,
            table_width,
            table_height,
        )
    )
    lines.append("</svg>")
    return "\n".join(lines)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3] + "..."


def _parse_run_dir_name(run_dir_name: str) -> Tuple[str, str]:
    if "__" not in run_dir_name:
        return run_dir_name, "unknown"
    raw_ts, model = run_dir_name.split("__", 1)
    ts = raw_ts
    if re.match(r"^\d{8}_\d{6}$", raw_ts):
        try:
            dt = datetime.strptime(raw_ts, "%Y%m%d_%H%M%S")
            ts = dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            ts = raw_ts
    return ts, model
