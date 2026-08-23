"""
tests/test_prediction_log.py: unit tests for the prediction log's schema, the
self-widening append, and the repair of a log written under mixed schemas.

Free of torch and of the model: the module under test handles only files and
column names, which is why it was separated from `model.py`.
"""

from __future__ import annotations

import csv

import pytest

from action_bins import DOF, readout_log_fields
from prediction_log import (
    LEADING_LOG_FIELDS,
    PROBE_EXTRA_FIELDS,
    RUN_METADATA_FIELDS,
    append_row,
    canonical_log_fields,
    ensure_readable,
    inspect_log,
    probe_log_fields,
    repair_log,
    resolve_schema,
    widen_log,
)

PROBE_EXTRA = list(PROBE_EXTRA_FIELDS)


def _write(path, fields, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fill(fields, tag):
    return {name: f"{name}_{tag}" for name in fields}


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- #
# Declared schema
# --------------------------------------------------------------------------- #
def test_canonical_fields_cover_every_part_of_a_row_in_order():
    fields = canonical_log_fields(PROBE_EXTRA)
    assert fields[:len(LEADING_LOG_FIELDS)] == list(LEADING_LOG_FIELDS)
    assert fields[len(LEADING_LOG_FIELDS):][:len(RUN_METADATA_FIELDS)] == list(
        RUN_METADATA_FIELDS)
    assert "a0" in fields and f"a{DOF - 1}" in fields
    assert set(readout_log_fields()) <= set(fields)
    assert fields[-len(PROBE_EXTRA):] == PROBE_EXTRA
    assert len(fields) == len(set(fields))


def test_the_probe_row_is_the_width_the_broken_log_reported():
    """A regression pinned to the observed failure.

    The log that could not be parsed had a 34-column header and 66-column rows.
    The 66 is the full probe schema, which is what confirms the rows were correct
    and the header was the part that was wrong.
    """
    assert len(probe_log_fields()) == 66
    assert len(canonical_log_fields(PROBE_EXTRA, with_readout=False)) == 37


def test_canonical_fields_without_a_readout_omit_only_the_readout():
    with_readout = canonical_log_fields(PROBE_EXTRA)
    without = canonical_log_fields(PROBE_EXTRA, with_readout=False)
    assert set(with_readout) - set(without) == set(readout_log_fields())


def test_declared_metadata_keys_match_what_run_metadata_returns():
    """The declaration lives apart from the function, so it is pinned to it.

    Skipped where torch is absent, since it is the only check here that needs the
    model module.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import torch

    from model import run_metadata

    assert list(run_metadata(torch.bfloat16)) == list(RUN_METADATA_FIELDS)


def test_readout_field_names_match_what_a_readout_actually_writes():
    """The declared names have to be the written names, or the header is a lie."""
    import numpy as np

    from action_bins import ActionReadout

    readout = ActionReadout(
        action=np.zeros(DOF), expected=np.zeros(DOF),
        bin_index=np.zeros(DOF, dtype=int), top_prob=np.zeros(DOF),
        entropy=np.zeros(DOF), action_mass=np.ones(DOF),
    )
    assert list(readout.to_log_fields()) == readout_log_fields()


# --------------------------------------------------------------------------- #
# Appending
# --------------------------------------------------------------------------- #
def test_append_creates_the_file_with_its_own_header(tmp_path):
    path = str(tmp_path / "log.csv")
    append_row(path, {"timestamp": "t0", "a0": 1.0})
    rows = _read(path)
    assert rows == [{"timestamp": "t0", "a0": "1.0"}]


def test_append_widens_the_file_when_a_row_brings_new_columns(tmp_path):
    """The failure this module exists to prevent.

    Writing the row at its own width leaves a file whose rows are wider than its
    header. Nothing complains at write time and every later read fails.
    """
    path = str(tmp_path / "log.csv")
    _write(path, ["timestamp", "a0"], [{"timestamp": "t0", "a0": "0.1"}])

    append_row(path, {"timestamp": "t1", "a0": 0.2, "c0": 0.25})

    rows = _read(path)
    assert [len(r) for r in rows] == [3, 3]
    assert rows[0] == {"timestamp": "t0", "a0": "0.1", "c0": ""}
    assert rows[1]["c0"] == "0.25"
    assert inspect_log(path)["consistent"]


def test_append_leaves_missing_columns_empty_rather_than_shifting_values(tmp_path):
    """A narrower row must not slide its values into the wrong columns."""
    path = str(tmp_path / "log.csv")
    append_row(path, {"timestamp": "t0", "a0": 0.1, "c0": 0.5})
    append_row(path, {"timestamp": "t1", "a0": 0.2})

    rows = _read(path)
    assert rows[1] == {"timestamp": "t1", "a0": "0.2", "c0": ""}


def test_append_keeps_the_existing_column_order(tmp_path):
    """New columns go on the end, so existing readers keep working."""
    path = str(tmp_path / "log.csv")
    _write(path, ["b", "a"], [{"b": "1", "a": "2"}])
    append_row(path, {"a": "3", "b": "4", "c": "5"})

    with open(path, newline="") as f:
        assert next(csv.reader(f)) == ["b", "a", "c"]


def test_appending_a_full_probe_row_to_a_migrated_log_recovers(tmp_path):
    """End to end on the shape that actually broke.

    A log migrated from the earlier schema, then appended to with full probe rows,
    stays readable instead of becoming unparseable at the first appended row.
    """
    path = str(tmp_path / "probe.csv")
    migrated = canonical_log_fields(PROBE_EXTRA[:5], with_readout=False)
    _write(path, migrated, [_fill(migrated, i) for i in range(3)])

    full = canonical_log_fields(PROBE_EXTRA)
    for i in range(2):
        append_row(path, _fill(full, i))

    report = inspect_log(path)
    assert report["consistent"]
    assert report["header_fields"] == len(full)
    rows = _read(path)
    assert len(rows) == 5
    # The migrated rows keep their values and are blank in the readout columns.
    assert rows[0]["a0"] == "a0_0"
    assert rows[0]["c0"] == ""
    assert rows[3]["c0"] == "c0_0"


def test_widening_is_atomic_enough_to_leave_no_partial_file(tmp_path):
    path = str(tmp_path / "log.csv")
    _write(path, ["timestamp"], [{"timestamp": "t0"}])
    widen_log(path, ["timestamp", "c0"])
    assert [p.name for p in tmp_path.iterdir()] == ["log.csv"]


def test_resolve_schema_does_not_rewrite_when_nothing_is_missing(tmp_path):
    path = str(tmp_path / "log.csv")
    _write(path, ["timestamp", "a0"], [{"timestamp": "t0", "a0": "1"}])
    before = open(path).read()
    assert resolve_schema(path, ["a0"]) == ["timestamp", "a0"]
    assert open(path).read() == before


# --------------------------------------------------------------------------- #
# Inspection and repair
# --------------------------------------------------------------------------- #
def _broken_log(path):
    """A header narrower than the rows appended after it, as observed."""
    narrow = canonical_log_fields(PROBE_EXTRA[:5], with_readout=False)
    wide = canonical_log_fields(PROBE_EXTRA)
    _write(path, narrow, [_fill(narrow, i) for i in range(3)])
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=wide)
        for i in range(2):
            writer.writerow(_fill(wide, i))
    return narrow, wide


def test_inspect_reports_the_widths_and_where_they_change(tmp_path):
    """A parser reports a line number and nothing else, which does not say why."""
    path = str(tmp_path / "log.csv")
    narrow, wide = _broken_log(path)

    report = inspect_log(path)
    assert report["consistent"] is False
    assert report["header_fields"] == len(narrow)
    assert set(report["widths"]) == {len(narrow), len(wide)}
    assert report["widths"][len(wide)]["rows"] == 2
    assert report["widths"][len(wide)]["first_line"] == 5


def test_inspect_reports_a_clean_log_as_consistent(tmp_path):
    path = str(tmp_path / "log.csv")
    fields = canonical_log_fields(PROBE_EXTRA)
    _write(path, fields, [_fill(fields, i) for i in range(3)])
    assert inspect_log(path)["consistent"]


def test_repair_reads_each_row_under_the_schema_of_its_own_width(tmp_path):
    path = str(tmp_path / "log.csv")
    narrow, wide = _broken_log(path)

    summary = repair_log(path, [wide], verbose=False)
    assert summary["rows_written"] == 5
    assert summary["rows_dropped"] == 0
    assert summary["columns"] == len(wide)

    rows = _read(path)
    # Values stay under their own names in both groups, which is the whole point.
    assert rows[0]["scene_id"] == "scene_id_0"
    assert rows[0]["c0"] == ""
    assert rows[3]["scene_id"] == "scene_id_0"
    assert rows[3]["c0"] == "c0_0"
    assert rows[3]["axis"] == "axis_0"
    assert inspect_log(path)["consistent"]


def test_repair_drops_unrecognised_widths_rather_than_guessing(tmp_path):
    """Values under the wrong names would corrupt the analysis silently."""
    path = str(tmp_path / "log.csv")
    narrow, wide = _broken_log(path)
    with open(path, "a", newline="") as f:
        f.write(",".join(["stray"] * 11) + "\n")

    summary = repair_log(path, [wide], verbose=False)
    assert summary["rows_written"] == 5
    assert summary["rows_dropped"] == 1
    assert summary["unmatched_widths"] == {11: 1}


def test_repair_can_write_to_a_new_path_leaving_the_original(tmp_path):
    path = str(tmp_path / "log.csv")
    out = str(tmp_path / "repaired.csv")
    narrow, wide = _broken_log(path)
    before = open(path).read()

    repair_log(path, [wide], out_path=out, verbose=False)
    assert open(path).read() == before
    assert inspect_log(out)["consistent"]


def test_repair_is_idempotent(tmp_path):
    path = str(tmp_path / "log.csv")
    _, wide = _broken_log(path)

    repair_log(path, [wide], verbose=False)
    once = open(path).read()
    repair_log(path, [wide], verbose=False)
    assert open(path).read() == once


def test_repair_needs_distinct_widths_to_be_sound(tmp_path):
    """Two schemas of equal width cannot be told apart by counting fields.

    Recorded as a constraint on the method rather than a defect: the first
    candidate wins, so a caller must not supply two same-width schemas.
    """
    path = str(tmp_path / "log.csv")
    fields = ["a", "b", "c"]
    _write(path, fields, [{"a": "1", "b": "2", "c": "3"}])

    summary = repair_log(path, [["x", "y", "z"]], verbose=False)
    assert summary["rows_written"] == 1
    assert _read(path)[0]["a"] == "1"


# --------------------------------------------------------------------------- #
# ensure_readable, the entry point every reader of the log goes through
# --------------------------------------------------------------------------- #
def test_ensure_readable_repairs_the_probe_log_without_being_told_the_schema(tmp_path):
    """Readers call this with a path alone, so the schema has to come from code."""
    path = str(tmp_path / "probe_predictions_v4.csv")
    _broken_log(path)

    summary = ensure_readable(path, verbose=False)
    assert summary["status"] == "repaired"
    assert summary["rows_written"] == 5
    assert summary["rows_dropped"] == 0
    assert inspect_log(path)["consistent"]


def test_ensure_readable_is_a_no_op_on_a_consistent_log(tmp_path):
    path = str(tmp_path / "log.csv")
    fields = probe_log_fields()
    _write(path, fields, [_fill(fields, i) for i in range(3)])
    before = open(path).read()

    assert ensure_readable(path, verbose=False)["status"] == "consistent"
    assert open(path).read() == before


def test_ensure_readable_refuses_to_repair_when_rows_would_be_lost(tmp_path):
    """The repair drops widths it cannot name, so it must not run unasked here.

    Rewriting the file would discard those rows silently. Reporting them and
    leaving the log untouched keeps the decision with whoever can look at them.
    """
    path = str(tmp_path / "log.csv")
    _broken_log(path)
    with open(path, "a", newline="") as f:
        f.write(",".join(["stray"] * 11) + "\n")
    before = open(path).read()

    with pytest.raises(ValueError, match="unrecognised width"):
        ensure_readable(path, verbose=False)
    assert open(path).read() == before


def test_ensure_readable_tolerates_a_missing_log(tmp_path):
    summary = ensure_readable(str(tmp_path / "absent.csv"), verbose=False)
    assert summary["status"] == "missing"


def test_ensure_readable_accepts_explicit_schemas(tmp_path):
    path = str(tmp_path / "log.csv")
    _write(path, ["a", "b"], [{"a": "1", "b": "2"}])
    with open(path, "a", newline="") as f:
        f.write("1,2,3\n")

    summary = ensure_readable(path, schemas=[["a", "b", "c"]], verbose=False)
    assert summary["status"] == "repaired"
    assert _read(path) == [{"a": "1", "b": "2", "c": ""},
                           {"a": "1", "b": "2", "c": "3"}]


@pytest.mark.parametrize("width", [0, 1])
def test_inspect_handles_a_header_only_or_empty_log(tmp_path, width):
    path = str(tmp_path / "log.csv")
    with open(path, "w", newline="") as f:
        if width:
            f.write("timestamp\n")
    report = inspect_log(path)
    assert report["widths"] == {}
    assert report["consistent"]
