"""
prediction_log.py: schema and file handling for the prediction log.

The prediction log is appended to across long runs that are resumed after
interruptions, after code changes, and after migrations from earlier schemas.
Those are exactly the conditions under which a CSV's columns drift, and a CSV
whose header is narrower than its rows cannot be read back at all: the writer
accepts every row, and the file only breaks when something later tries to parse
it, by which point the run that produced it is long finished.

This module keeps the schema explicit rather than inferred from whichever row
happened to be written first:

  * `canonical_log_fields` states the full column order in advance, so a log can
    be created with its final header before any prediction exists.
  * `append_row` writes against the file's own header and widens the file when a
    row carries new columns, so mixed call shapes stay readable.
  * `inspect_log` and `repair_log` diagnose and rebuild a log that was written
    before those guarantees existed.

Free of torch and of the model, so the schema arithmetic is unit-testable without
a GPU. `model.append_prediction_log` builds the row and delegates the file
handling here.
"""

from __future__ import annotations

import csv
import os

from action_bins import DOF, readout_log_fields

# Columns every prediction carries, in the order they are written.
LEADING_LOG_FIELDS = ("timestamp", "instruction", "unnorm_key", "do_sample")

# Keys of `model.run_metadata`, declared here so the schema can be built without
# importing torch. A test asserts the two agree.
RUN_METADATA_FIELDS = ("gpu_name", "gpu_capability", "dtype", "seed", "torch",
                       "transformers", "bitsandbytes")

# The probe's `**extra` keys, in the order the sweep passes them. Declared here
# rather than in a notebook cell because three places need to agree on it: the
# header the migration writes, the rows the sweep appends, and the schema a repair
# reads an existing log under. A copy in a notebook would be a fourth declaration
# able to drift from the other three.
#
# Two properties of this list are load-bearing for the constructed set. It carries
# `base_scene_id`, the frame a constructed stimulus was composited from, because the
# frozen set pairs each same-side scene with the opposite scene built from that same
# frame and the decisive contrast is meant to be read within a frame; a log without
# it cannot express the pairing at all. And the geometry columns hold the recorded
# positions in image coordinates, unconverted. The convention relating image
# position to the sign of the lateral action is applied in `analysis.py` instead, so
# revising it is a re-read of the log rather than a repeat of every prediction in it.
PROBE_EXTRA_FIELDS = (
    "scene_id", "pair_id", "base_scene_id", "role", "frame", "scene_source",
    "condition", "image_transform", "image_scene_id", "configuration",
    "expected_sign_image", "target_sign_a_image", "target_sign_b_image",
    "spatial_term", "axis", "axis_index", "category", "feasible_both",
    "duplicate_target", "sample_idx",
)

# Resolved header per log path, so the header is not re-read from a mounted drive
# before each of several thousand appends.
_SCHEMA_CACHE: dict = {}


def canonical_log_fields(extra_fields=(), *, with_readout: bool = True,
                         n_dims: int = DOF) -> list[str]:
    """Full column order a prediction row produces for a given call shape.

    `extra_fields` are the caller's `**extra` keys, in the order they are passed.

    Knowing the complete schema in advance is what lets a log be created with its
    final header, and what lets a log written under an earlier schema be widened
    to the current one before anything is appended to it.
    """
    fields = list(LEADING_LOG_FIELDS) + list(RUN_METADATA_FIELDS)
    fields += [f"a{i}" for i in range(n_dims)]
    if with_readout:
        fields += readout_log_fields(n_dims)
    fields += [f for f in extra_fields if f not in fields]
    return fields


def probe_log_fields() -> list[str]:
    """Full column order of the contrastive probe's log."""
    return canonical_log_fields(PROBE_EXTRA_FIELDS)


# Extra keys for the first-action object-tracking log, written by Notebook 03.
# A separate file from the probe log: that log is the language experiment, this
# one is the instrument check that licenses reading it.
TRACKING_EXTRA_FIELDS = (
    "scene_id", "episode_index", "role", "condition", "image_transform",
    "noun", "spatial_term", "sample_idx",
)


def tracking_log_fields() -> list[str]:
    """Full column order of the object-tracking instrument-check log."""
    return canonical_log_fields(TRACKING_EXTRA_FIELDS)


def widen_log(csv_path: str, extra_fields, verbose: bool = False) -> list[str]:
    """Rewrite `csv_path` with `extra_fields` appended to its header.

    Existing rows keep their values and take an empty string in the new columns.
    The rewrite goes to a temporary file that then replaces the original, so an
    interruption leaves either the old log or the new one and never a half-written
    file. That matters on a mounted drive, where a long sweep is the thing most
    likely to be interrupted.
    """
    with open(csv_path, newline="") as f:
        header = next(csv.reader(f), [])
    missing = [f for f in extra_fields if f not in header]
    if not missing:
        return list(header)

    widened = list(header) + missing
    temp_path = f"{csv_path}.widening"
    with open(csv_path, newline="") as src, open(temp_path, "w", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=widened, restval="")
        writer.writeheader()
        for row in reader:
            writer.writerow(row)
    os.replace(temp_path, csv_path)
    _SCHEMA_CACHE[csv_path] = widened
    if verbose:
        print(f"[widen_log] added {len(missing)} columns to {csv_path}: {missing}")
    return widened


def resolve_schema(csv_path: str, wanted) -> list[str]:
    """Header to write `wanted` against, creating or widening the file as needed."""
    cached = _SCHEMA_CACHE.get(csv_path)
    if cached is not None and not set(wanted) - set(cached):
        return cached

    if not (os.path.exists(csv_path) and os.path.getsize(csv_path) > 0):
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=list(wanted)).writeheader()
        schema = list(wanted)
    else:
        schema = widen_log(csv_path, wanted)

    _SCHEMA_CACHE[csv_path] = schema
    return schema


def append_row(csv_path: str, row: dict) -> dict:
    """Append one row, aligning it to the file's header.

    A row carrying columns the header lacks widens the file rather than being
    written at its own width. Columns absent from the row are written empty, so a
    log may hold rows from different call shapes: predictions logged without a
    continuous readout, including rows migrated from an earlier schema, sit in the
    same file as rows that have one.
    """
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    fields = resolve_schema(csv_path, list(row.keys()))
    with open(csv_path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=fields, restval="").writerow(row)
    return row


def inspect_log(csv_path: str) -> dict:
    """Field counts present in a log, without parsing it as a table.

    A log whose rows are wider than its header cannot be read by a CSV parser at
    all, so the failure reports a line number and nothing else. This reads the file
    as raw rows and reports how many fields each line actually has and where each
    width first appears, which is what identifies the point the schema changed.
    """
    widths: dict = {}
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        for line_no, row in enumerate(reader, start=2):
            if not row:
                continue
            entry = widths.setdefault(len(row), {"rows": 0, "first_line": line_no})
            entry["rows"] += 1
    return {
        "header_fields": len(header),
        "header": list(header),
        "widths": dict(sorted(widths.items())),
        "consistent": set(widths) <= {len(header)},
    }


def ensure_readable(csv_path: str, schemas=None, verbose: bool = True) -> dict:
    """Make a log parseable, repairing it only when nothing would be lost.

    Called before reading the log in any notebook that reads it. A log written
    before the header was declared in full can hold rows wider than its own
    header, which no CSV reader will parse, and repairing it is a file operation
    that re-runs nothing on GPU. On a consistent log this does nothing.

    The repair goes ahead only when every field count present is accounted for by
    the file's header or one of `schemas`, which defaults to the probe schema. An
    unaccounted width would be dropped by `repair_log`, so rather than rewrite the
    file and lose those rows, this reports what it found and leaves the log alone
    for the caller to look at.
    """
    if not os.path.exists(csv_path):
        return {"status": "missing", "path": csv_path}

    report = inspect_log(csv_path)
    if report["consistent"]:
        if verbose:
            print(f"{csv_path}\n  {report['header_fields']} columns, "
                  f"{sum(e['rows'] for e in report['widths'].values())} rows, "
                  "consistent")
        return {"status": "consistent", "path": csv_path, "report": report}

    schemas = [list(s) for s in (schemas if schemas is not None
                                 else [probe_log_fields()])]
    if verbose:
        print(f"{csv_path}\n  header declares {report['header_fields']} columns "
              "but the rows disagree:")
        for width, entry in report["widths"].items():
            print(f"    {entry['rows']:6} rows of {width:3} fields "
                  f"(first at line {entry['first_line']})")

    known = {report["header_fields"]} | {len(s) for s in schemas}
    unknown = {w: e["rows"] for w, e in report["widths"].items() if w not in known}
    if unknown:
        raise ValueError(
            f"{csv_path} holds rows of unrecognised width {unknown}, matching "
            f"neither its {report['header_fields']}-column header nor any known "
            f"schema (widths {sorted(len(s) for s in schemas)}). Repairing would "
            "drop them, so the log is unchanged. Inspect those lines, then call "
            "repair_log directly with a schema for them."
        )

    summary = repair_log(csv_path, schemas, verbose=verbose)
    if not inspect_log(csv_path)["consistent"]:
        raise ValueError(f"{csv_path} is still inconsistent after repair")
    summary["status"] = "repaired"
    return summary


def repair_log(csv_path: str, schemas, out_path: str | None = None,
               verbose: bool = True) -> dict:
    """Rewrite a log whose rows were written against differing headers.

    `schemas` are candidate field orders, each matched to a row by its field count.
    The file's own header is always a candidate. Rows are re-read under the schema
    of their own width and written out under the union of all columns, so no value
    moves to a different column and none is discarded.

    Matching on width is sound only because the widths are distinct and each
    corresponds to a known call shape, which is why the schemas are supplied by the
    caller rather than guessed. A row whose width matches nothing is dropped and
    counted: placing its values under the wrong names would corrupt the analysis
    silently, which is worse than losing the row visibly.
    """
    with open(csv_path, newline="") as f:
        header = next(csv.reader(f), [])

    by_width = {len(header): list(header)}
    for schema in schemas:
        by_width.setdefault(len(schema), list(schema))

    union: list = []
    for schema in [list(header)] + [by_width[w] for w in sorted(by_width)]:
        union += [name for name in schema if name not in union]

    out_path = out_path or csv_path
    temp_path = f"{out_path}.repairing"
    counts: dict = {}
    unmatched: dict = {}
    with open(csv_path, newline="") as src, open(temp_path, "w", newline="") as dst:
        reader = csv.reader(src)
        next(reader, None)
        writer = csv.DictWriter(dst, fieldnames=union, restval="")
        writer.writeheader()
        for row in reader:
            if not row:
                continue
            schema = by_width.get(len(row))
            if schema is None:
                unmatched[len(row)] = unmatched.get(len(row), 0) + 1
                continue
            writer.writerow(dict(zip(schema, row)))
            counts[len(row)] = counts.get(len(row), 0) + 1
    os.replace(temp_path, out_path)
    _SCHEMA_CACHE[out_path] = union

    summary = {
        "path": out_path,
        "columns": len(union),
        "rows_by_width": counts,
        "rows_written": sum(counts.values()),
        "rows_dropped": sum(unmatched.values()),
        "unmatched_widths": unmatched,
    }
    if verbose:
        print(f"[repair_log] {summary['rows_written']} rows -> {out_path} under "
              f"{len(union)} columns, by original width {counts}")
        if unmatched:
            print(f"[repair_log] dropped {summary['rows_dropped']} rows of "
                  f"unrecognised width {unmatched}; these match no supplied call "
                  "shape and were not guessed at")
    return summary
