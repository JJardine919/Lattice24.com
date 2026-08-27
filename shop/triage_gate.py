#!/usr/bin/env python3
"""
Pre-flight triage gate.

Runs BEFORE the engine. Decides the branch from the file itself in seconds.

Three outcomes, exactly one per upload:
  INSTANT  — the data clears the gate; engine runs, report comes back
  REVIEW   — gate is ambiguous; a person looks, Jim answers
  CANNOT   — gate fails; plain explanation + the specific thing that would fix it

75% of domains are unreadable to the screen (73 of 97 at block cosine >= 0.9,
see ~/BUILDS/INDEX.md RESEARCH). REVIEW and CANNOT are the normal path.

Every check says which branch it forces and why, in words a customer could read.

2026-08-26: initial version. Gates only — does not run the engine.
"""

import csv
import io
import statistics
from pathlib import Path

import numpy as np

import encoder_gate as _gate   # the engine's own per-channel gate; analyze.py calls this too


# ---------------------------------------------------------------------------
# tunable thresholds (all verified against disk: BUILDS/INDEX.md, ramp-regime
# census, and the existing analyze.py constants)
# ---------------------------------------------------------------------------

MIN_ROWS = 200          # below this: CANNOT (too few rows, nothing to read)
NCH = 16                # channels per 128-vector (matches analyze.py: NCH = 16)
WIN = 8                 # consecutive time steps per channel (matches analyze.py: WIN = 8)
SAT_LIMIT = 0.90        # end-bin mass above this: saturated (matches analyze.py)
TREND_THRESHOLD = 0.85  # |mean|/sd above this: trend-shaped regime, order verdict can fire
BLOCK_COSINE_CLEAR = 0.90  # block cosine above this: data is not readable as-is


#: Files that are not CSVs, recognised by their own first bytes.
#:
#: This exists because a PDF was dropped on the page and the site answered
#: "Estimated saved: $13 + sensor report (16 channels)". Nothing had read the
#: PDF -- the figure came from its byte count and the channel count was a
#: hardcoded string. The gate's job here is to say what the file IS, in one
#: sentence, and ask for the thing that would work. Telling somebody who
#: uploaded a PDF that his file "has 3 data rows, send at least 200" is a
#: technically-true answer to a question he did not ask.
_MAGIC = [
    (b"%PDF-", "a PDF"),
    (b"PK\x03\x04", "a ZIP, XLSX or ODS archive"),
    (b"\xd0\xcf\x11\xe0", "an old-format Excel or Word document"),
    (b"\x89PNG", "a PNG image"),
    (b"\xff\xd8\xff", "a JPEG image"),
    (b"\x1f\x8b", "a gzip archive"),
    (b"SQLite format 3", "a SQLite database"),
    (b"{", "JSON"),
    (b"[", "JSON"),
    (b"<?xml", "XML"),
    (b"<", "HTML or XML"),
]


def looks_like_csv(raw_or_text):
    """(is_csv, what_it_looks_like). Reads the file's own bytes, not its name."""
    if isinstance(raw_or_text, str):
        raw = raw_or_text.encode("utf-8", errors="replace")
    else:
        raw = raw_or_text
    head = raw[:2048].lstrip()
    for sig, what in _MAGIC:
        if head.startswith(sig):
            return False, what
    # A CSV is text. A binary file that dodged the signature list still has
    # bytes no text file has.
    sample = raw[:8192]
    if sample and sum(1 for b in sample if b < 9 or (13 < b < 32)) > len(sample) * 0.02:
        return False, "a binary file, not text"
    # Text, but is it delimited? A CSV's first line has a separator in it.
    try:
        first = raw[:8192].decode("utf-8", errors="replace").splitlines()
    except Exception:  # noqa: BLE001
        return False, "a file we could not decode as text"
    first = [ln for ln in first if ln.strip()]
    if not first:
        return False, "an empty file"
    if not any(d in first[0] for d in (",", ";", "\t", "|")):
        return False, "a text file with no comma, tab or semicolon in its first line"
    return True, "CSV"


#: Header names that declare a column to be the time axis.
_TIME_NAMES = ("time", "timestamp", "date", "datetime", "ts", "epoch",
               "sample", "index", "seconds", "second", "minute", "minutes",
               "hour", "hours", "period", "step", "reading_time", "utc")

#: Every timestamp layout seen on this machine, plus the one Python's own
#: datetime.isoformat() writes. The gate used to accept ONLY
#: "%Y-%m-%d %H:%M:%S", so an ISO-8601 stamp with a T separator -- the format
#: every machine-written log uses -- was reported to the customer as "the first
#: column does not look like a time stamp". He would then have been asked to
#: add a time column to a file that already had one.
_TIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d",
    "%Y/%m/%d %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S",
    "%d-%m-%Y %H:%M:%S", "%Y%m%d%H%M%S", "%Y%m%d",
    "%H:%M:%S",
)


def _parse_stamp(v):
    """One cell -> float, datetime, or None. Tries the fractional-second form
    by trimming it, so 2026-01-01T00:00:00.250 parses like its whole second."""
    from datetime import datetime
    v = v.strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        pass
    base = v.split(".")[0].replace("+00:00", "").rstrip("Z") if "." in v or v.endswith("Z") else v
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(v, fmt)
        except (ValueError, TypeError):
            pass
        if base != v:
            try:
                return datetime.strptime(base, fmt)
            except (ValueError, TypeError):
                pass
    return None


def _must_have_rows(rows):
    """Return (ok, reason) for the row-count gate."""
    if len(rows) < MIN_ROWS:
        return False, (
            f"Your file has {len(rows)} data rows. "
            f"The engine needs at least {MIN_ROWS} to read anything meaningful. "
            f"Send the same columns with at least {MIN_ROWS} rows "
            f"(a few thousand is better)."
        )
    return True, ""


def _numeric_columns(rows):
    """
    Return list of (name, np.array) for every numeric column.

    Non-numeric columns are silently dropped — the gate reads whatever header
    it is given and reports what it found; it never requires a fixed layout.
    """
    if not rows:
        return []
    header = [h.strip() for h in rows[0]]
    ncol = len(rows[1]) if len(rows) > 1 else 0
    cols = [[] for _ in range(ncol)]
    names = header[:ncol] if header and len(header) == ncol else [f"col{i}" for i in range(ncol)]
    for r in rows[1:]:
        if len(r) != ncol:
            continue
        for i, v in enumerate(r):
            try:
                cols[i].append(float(v))
            except (ValueError, IndexError):
                pass
    out = []
    for i, c in enumerate(cols):
        if len(c) >= 60:
            arr = np.array(c, float)
            if np.nanstd(arr) >= 1e-12:   # constant columns carry nothing
                out.append((names[i], arr))
    return out


def _time_column_check(rows, col_names):
    """
    Is there a first column that is a time axis, and is it in order?

    Returns (has_time, is_monotone, reason).

    2026-08-26 -- two defects fixed here, both of which sent the customer a
    confidently WRONG instruction, which is worse than a vague one:

      * Only "%Y-%m-%d %H:%M:%S" parsed, so an ISO-8601 stamp with a T
        separator was reported as "the first column does not look like a time
        stamp" and the customer was asked to add a column his file already had.

      * ANY numeric first column was treated as the clock. A file whose first
        column is a sensor reading was told "the rows are not in time order --
        sort by the first column and send it again." Following that instruction
        sorts a time series by one of its own channels and destroys the very
        ordering the engine reads. A file with no time column must be told it
        has no time column.

    A column is the time axis if it is NAMED like one, or if it is monotone.
    Numbers that are neither are just numbers.
    """
    if not rows or len(rows) < 3:
        return False, False, "File has too few rows to check time ordering."
    name = (col_names[0].strip().lower() if col_names else "")
    named_time = any(t in name for t in _TIME_NAMES)
    first = [r[0].strip() for r in rows[1:] if r]
    if not first:
        return False, False, "No data rows found."

    parsed = [_parse_stamp(v) for v in first]
    clean = [p for p in parsed if p is not None]
    parse_rate = len(clean) / len(first) if first else 0.0
    all_numeric = bool(clean) and all(isinstance(p, float) for p in clean)

    if parse_rate < 0.8:
        cols = ", ".join(c.strip() for c in col_names[:8]) or "(no header row)"
        return False, False, (
            f"We could not find a time column. The first column is \"{col_names[0].strip()}\" "
            f"and its values do not read as timestamps. The columns we did read were: "
            f"{cols}. Send the same rows with a first column of timestamps "
            f"(2026-01-01 00:00:00 or 2026-01-01T00:00:00 both work), or a plain "
            f"increasing sample number, oldest first. The engine reads sequence, so "
            f"without that it cannot tell your rows apart from a shuffled copy."
        )

    # Monotone?
    if all_numeric:
        arr = np.array([float(p) for p in clean])
        diffs = np.diff(arr)
        n_back = int(np.sum(diffs < -1e-12))
    else:
        diffs = [(clean[i + 1] - clean[i]).total_seconds() for i in range(len(clean) - 1)]
        n_back = sum(1 for d in diffs if d < -1e-12)
    monotone = n_back == 0
    n_pairs = len(diffs)

    if monotone:
        return True, True, ""

    # Not monotone. Whether that is "unsorted timestamps" or "this was never a
    # clock" depends entirely on what the column is -- and getting that wrong is
    # how the customer gets told to sort by a pressure reading.
    if not named_time and all_numeric:
        cols = ", ".join(c.strip() for c in col_names[:8]) or "(no header row)"
        return False, False, (
            f"We could not find a time column. The first column is \"{col_names[0].strip()}\", "
            f"which is numeric but is not named like a time axis and does not increase "
            f"({n_back} of {n_pairs} consecutive pairs go backward) -- so it reads as a "
            f"measurement, not a clock. The columns we read were: {cols}. Add a first "
            f"column of timestamps or an increasing sample number, oldest first. "
            f"Do NOT sort your rows by \"{col_names[0].strip()}\" -- that would reorder the "
            f"measurements themselves, and the engine reads sequence."
        )
    return True, False, (
        f"The first column \"{col_names[0].strip()}\" is a time stamp, but the rows are not "
        f"in time order -- {n_back} of {n_pairs} consecutive pairs go backward. "
        f"Rows must be oldest first. Sort by that column and send it again."
    )


def _sorted_input_check(rows):
    """
    Detect whether rows are sorted ascending on the first numeric column.

    A file that is sorted on a value column (not time) is a VOID — the engine's
    ORDER-INVARIANT verdict cannot fire, and reporting it as "we found nothing"
    is misleading. Returns (is_sorted_values, reason).
    """
    if not rows or len(rows) < 20:
        return False, ""
    # look at the first numeric column that is NOT the time column
    header = rows[0]
    for col_idx in range(1, len(header)):
        try:
            vals = [float(r[col_idx]) for r in rows[1:] if len(r) > col_idx]
            if len(vals) < 20:
                continue
            arr = np.array(vals)
            diffs = np.diff(arr)
            # sorted ascending: all diffs >= 0 (with small tolerance)
            if np.all(diffs >= -1e-9):
                return True, (
                    f"The data in column {header[col_idx] if col_idx < len(header) else col_idx} "
                    f"is sorted ascending. That means the rows are ordered by value, not by "
                    f"time. The engine's order verdict cannot fire on sorted data — a VOID "
                    f"verdict is a CANNOT-READ with a known reason, not a null. Send the same "
                    f"columns with their timestamps and we can read it."
                )
            # sorted descending
            if np.all(diffs <= 1e-9):
                return True, (
                    f"The data in column {header[col_idx] if col_idx < len(header) else col_idx} "
                    f"is sorted descending. Same problem as sorted ascending — the order is "
                    f"by value, not time. Send with timestamps."
                )
        except (ValueError, IndexError):
            continue
    return False, ""


def _channel_shape_checks(channels):
    """
    Per-channel shape checks: row count, gaps, one bad cell silently deleting a channel.

    Returns list of (name, problem, reason) for channels with issues.
    """
    problems = []
    for name, arr in channels:
        n = len(arr)
        # gaps: large jumps in consecutive rows (more than 10x the median absolute diff)
        diffs = np.abs(np.diff(arr))
        median_diff = np.median(diffs) if len(diffs) else 0
        if median_diff > 0:
            gap_mask = diffs > 10 * median_diff
            n_gaps = int(np.sum(gap_mask))
            if n_gaps > n * 0.05:   # more than 5% of consecutive pairs are gaps
                problems.append(
                    (name, "gaps",
                     f"Column {name} has {n_gaps} large gaps between consecutive rows "
                     f"(more than 10x the typical step). That usually means the data was "
                     f"resampled or spliced. Send the raw unsampled series if you have it."
                    ))
        # NaN fraction
        nan_frac = np.isnan(arr).mean()
        if nan_frac > 0.5:
            problems.append(
                (name, "mostly-nan",
                 f"Column {name} is {nan_frac*100:.0f}% empty. We drop columns that are "
                 f"mostly missing, which means this column will not be read at all. Send it "
                 f"with the missing values filled or as a separate file."
                ))
    return problems


def _saturation_per_channel(channels):
    """
    Encoder health per channel, decided by the ENGINE's own gate.

    2026-08-26 -- this used to be a hand-rolled reimplementation of the fixed
    ladder in analyze.py, and it reproduced two defects that encoder_gate.py had
    already fixed. Measured on a 16-channel test file built for this prompt:

      * Twelve channels that sat at a constant 0.0 with a rare huge spike came
        back "0.0% of changes in extreme bins (readable)". `arr[arr != 0]`
        deleted 897 of 900 samples, and what was left was a constant. That is
        analyze.py's own documented failure mode -- "a flat channel returns a
        clean, believable null that means nothing" -- happening inside the check
        written to prevent it. encoder_gate catches it with a resolution test on
        distinct SAMPLE values, which runs before any early return.
      * Four ordinary sine channels came back 89-91% and one was called
        SATURATED. They are not: analyze.py refits the ladder to the channel's
        own change distribution when the fixed one saturates, and a plain
        sinusoid is the case that refit exists for. So the pre-flight was
        refusing channels the engine behind it reads happily.

    A pre-flight gate that disagrees with the engine it gates for is worse than
    no pre-flight, so this now calls `encoder_gate.gate_all` -- the same
    function analyze.py calls, on the same window layout. No mathematics here.

    Returns list of (name, end_bin_mass, readable), one per channel.
    """
    if not channels:
        return []
    names = [n for n, _ in channels]
    n = min(len(channels), NCH)
    length = min(len(a) for _, a in channels)
    if length < WIN * 2:
        return [(nm, float("nan"), False) for nm in names]
    X = np.column_stack([a[:length] for _, a in channels[:n]])
    # Same channel-major 128-vector layout analyze.py's windows() produces:
    # nch channels x WIN consecutive samples, so each 8-slice the entropy
    # function sees is one channel's own short series.
    vecs = []
    for st in range(0, length - WIN, WIN):
        w = X[st:st + WIN, :n]
        if np.all(np.isfinite(w)):
            vecs.append((st, w.T.reshape(-1)))
    if not vecs:
        return [(nm, float("nan"), False) for nm in names]
    gates = _gate.gate_all(vecs, names[:n], n)
    out = []
    for g in gates:
        end = g["end"]
        # "readable" and "refitted" are both readable -- refitted means the
        # ladder was refit to this channel and the engine went on to read it.
        out.append((g["channel"], float(end) if end == end else float("nan"),
                    g["status"] in ("readable", "refitted")))
    # Channels past NCH are not handed to the engine at all; say so rather than
    # silently dropping them from the report.
    for nm in names[n:]:
        out.append((nm, float("nan"), False))
    return out


def _trend_shape_check(channels):
    """
    |mean|/sd across channels — the two-number pre-flight.

    Above ~0.85: trend-shaped regime, the order verdict can fire.
    Below: the order verdict cannot fire (CANNOT on the order arm).

    Returns (trend_ratio, classification).
    """
    if not channels:
        return 0.0, "no channels"
    ratios = []
    for _, arr in channels:
        arr = arr[np.isfinite(arr)]
        if len(arr) < 10:
            continue
        s = np.nanstd(arr)
        if s > 1e-12:
            ratios.append(abs(arr.mean()) / s)
    if not ratios:
        return 0.0, "no readable channels"
    ratio = float(np.median(ratios))
    if ratio > TREND_THRESHOLD:
        return ratio, "trend"
    return ratio, "flat"


def _block_cosine(channels):
    """
    Block cosine: measure of how much the channels align in time.

    High block cosine (>= 0.9) means the data is not readable as independent
    channels — 73 of 97 domains at this machine hit this (BUILDS/INDEX.md).

    Returns (cosine, classification).
    """
    if len(channels) < 2:
        return 0.0, "need at least 2 channels"
    arrs = []
    for _, a in channels:
        a = a[np.isfinite(a)]
        if len(a) < 10:
            continue
        arrs.append(a - a.mean())
    if len(arrs) < 2:
        return 0.0, "not enough readable channels"
    # stack into matrix, compute mean pairwise cosine
    mat = np.column_stack(arrs)
    norms = np.linalg.norm(mat, axis=0)
    if np.any(norms < 1e-12):
        return 0.0, "zero-norm channel"
    normed = mat / norms
    # mean absolute cosine between all pairs
    n = normed.shape[1]
    cosines = []
    for i in range(n):
        for j in range(i + 1, n):
            cosines.append(abs(np.dot(normed[:, i], normed[:, j])))
    return float(np.mean(cosines)), "block_cosine"


def triage(csv_text):
    """
    Run the pre-flight gate on a CSV string.

    Returns dict with:
      branch: "INSTANT" | "REVIEW" | "CANNOT"
      reasons: list of (gate, verdict, explanation) — what each check said
      pre_flight: dict of the two-number pre-flight (block_cosine, trend_ratio)
      shape_problems: list of channel shape issues
      saturation: list of (channel, end_bin_mass, readable) per channel
      summary: one paragraph a customer could read
    """
    is_csv, what = looks_like_csv(csv_text)
    if not is_csv:
        ask = ("Export the table you want read as CSV -- one row per reading, "
               "one column per channel, a timestamp in the first column -- and send "
               "that. Every tool that produced this file can do it: in Excel or "
               "Sheets it is File > Save as / Download as > CSV; from a historian it "
               "is usually an export or trend-to-file option.")
        reason = (f"This file is {what}, not a CSV, so there were no numbers in it to "
                  f"read. Nothing about it was analysed. " + ask)
        return {
            "branch": "CANNOT",
            "reasons": [("file_type", "FAIL", reason)],
            "pre_flight": {"block_cosine": 0.0, "trend_ratio": 0.0,
                           "classification": f"not a CSV ({what})"},
            "shape_problems": [], "saturation": [],
            "summary": reason, "n_channels": 0, "n_rows": 0,
        }

    reader = csv.reader(io.StringIO(csv_text))
    rows = list(reader)
    if not rows or len(rows) < 2:
        return {
            "branch": "CANNOT",
            "reasons": [("row_count", "FAIL",
                        "The file appears to be empty or has no data rows.")],
            "pre_flight": {"block_cosine": 0.0, "trend_ratio": 0.0, "classification": "no data"},
            "shape_problems": [],
            "saturation": [],
            "summary": "The file is empty. Send a CSV with a header row and at least 200 data rows.",
            "n_channels": 0,
            "n_rows": 0,
        }

    n_rows = len(rows) - 1
    channels = _numeric_columns(rows)
    n_channels = len(channels)

    reasons = []

    # --- 1. row count ---
    row_ok, row_reason = _must_have_rows(rows[1:])
    if not row_ok:
        reasons.append(("row_count", "FAIL", row_reason))
        return {
            "branch": "CANNOT",
            "reasons": reasons,
            "pre_flight": {"block_cosine": 0.0, "trend_ratio": 0.0, "classification": "too few rows"},
            "shape_problems": [],
            "saturation": [],
            "summary": row_reason,
            "n_channels": n_channels,
            "n_rows": n_rows,
        }

    # --- 2. time column ---
    has_time, monotone, time_reason = _time_column_check(rows, [h.strip() for h in rows[0]])
    reasons.append(("time_column", "PASS" if (has_time and monotone) else "FAIL",
                    time_reason if time_reason else "Time column present and ordered."))
    if not (has_time and monotone):
        reasons.append(("time_column", "FAIL", time_reason))
        return {
            "branch": "CANNOT",
            "reasons": reasons,
            "pre_flight": {"block_cosine": 0.0, "trend_ratio": 0.0, "classification": "no time axis"},
            "shape_problems": [],
            "saturation": [],
            "summary": time_reason,
            "n_channels": n_channels,
            "n_rows": n_rows,
        }

    # --- 3. sorted-input control (VOID check) ---
    sorted_val, sorted_reason = _sorted_input_check(rows)
    if sorted_reason:
        reasons.append(("sorted_input", "VOID", sorted_reason))
        return {
            "branch": "CANNOT",
            "reasons": reasons,
            "pre_flight": {"block_cosine": 0.0, "trend_ratio": 0.0, "classification": "sorted_values"},
            "shape_problems": [],
            "saturation": [],
            "summary": sorted_reason,
            "n_channels": n_channels,
            "n_rows": n_rows,
        }

    # --- 4. channel shape checks ---
    shape_problems = _channel_shape_checks(channels)
    for name, kind, reason in shape_problems:
        reasons.append(("shape", "FAIL", reason))
    if shape_problems:
        # shape problems are CANNOT — the file has structural issues
        summary_parts = [r[2] for r in reasons if r[1] == "FAIL"]
        return {
            "branch": "CANNOT",
            "reasons": reasons,
            "pre_flight": {"block_cosine": 0.0, "trend_ratio": 0.0, "classification": "shape_problem"},
            "shape_problems": shape_problems,
            "saturation": [],
            "summary": " ".join(summary_parts[:2]) +  # first two failures as the headline
                       (f" Plus {len(shape_problems)-2} more channel issues." if len(shape_problems) > 2 else ""),
            "n_channels": n_channels,
            "n_rows": n_rows,
        }

    # --- 5. saturation per channel ---
    saturation = _saturation_per_channel(channels)
    n_saturated = sum(1 for _, m, r in saturation if not r)
    n_readable = sum(1 for _, m, r in saturation if r)
    for name, mass, readable in saturation:
        verdict = "PASS" if readable else "SATURATED"
        reasons.append(("saturation", verdict,
                        f"Channel {name}: {mass*100:.1f}% of changes in extreme bins "
                        f"({'readable' if readable else 'saturated — would return a confident-looking result that means nothing'})"))
    if n_saturated > n_readable:
        # majority saturated: CANNOT
        sat_reason = (
            f"{n_saturated} of {n_channels} channels are saturated on the standard scale. "
            f"A saturated channel returns a confident-looking result that means nothing — "
            f"consecutive rows are not consecutive measurements of the same thing, or the "
            f"columns are on wildly different scales. Try sending one or two key channels "
            f"on the same scale, or confirm the row ordering and column definitions."
        )
        reasons.append(("saturation_majority", "FAIL", sat_reason))
        return {
            "branch": "CANNOT",
            "reasons": reasons,
            "pre_flight": {"block_cosine": 0.0, "trend_ratio": 0.0, "classification": "saturated"},
            "shape_problems": [],
            "saturation": saturation,
            "summary": sat_reason,
            "n_channels": n_channels,
            "n_rows": n_rows,
        }

    # --- 6. two-number pre-flight: block cosine + |mean|/sd ---
    block_cos, block_label = _block_cosine(channels)
    trend_ratio, trend_class = _trend_shape_check(channels)
    pre_flight = {
        "block_cosine": block_cos,
        "trend_ratio": trend_ratio,
        "classification": f"{block_label} / {trend_class}",
    }
    reasons.append(("block_cosine", "PASS" if block_cos < BLOCK_COSINE_CLEAR else "HIGH",
                    f"Block cosine {block_cos:.3f} — "
                    + ("channels move independently enough to read" if block_cos < BLOCK_COSINE_CLEAR
                       else "channels move together too much to separate; 73 of 97 domains at this machine hit this level. Send fewer channels on the same scale, or separate them by run.")))
    trend_desc = ("trend-shaped, order verdict can fire"
                  if trend_class == "trend"
                  else "flat regime, the order verdict cannot fire — "
                       "this is a CANNOT-READ on the order arm, which is a result not a failure.")
    reasons.append(("trend_ratio", "PASS" if trend_class == "trend" else "FLAT",
                    f"|mean|/sd {trend_ratio:.3f} — {trend_desc}"))

    # --- branch decision ---
    # INSTANT: block cosine clear AND trend-shaped AND not saturated majority
    # REVIEW:  block cosine high OR trend flat (ambiguous — a person looks)
    # CANNOT:  anything that fails hard above
    if block_cos >= BLOCK_COSINE_CLEAR and trend_class == "flat":
        branch = "REVIEW"
        branch_reason = (
            f"Block cosine {block_cos:.3f} (high) and trend ratio {trend_ratio:.3f} (flat). "
            f"The data is both aligned in time and not trend-shaped, which means the engine's "
            f"order verdict cannot fire and the channels are not independent. A person will look "
            f"at this and decide if it can be read with a different framing, or if we need "
            f"different data."
        )
    elif block_cos >= BLOCK_COSINE_CLEAR:
        branch = "REVIEW"
        branch_reason = (
            f"Block cosine {block_cos:.3f} is high — the channels move together too much to "
            f"separate. A person will look at whether fewer channels on the same scale, or "
            f"separate runs, would let the engine read it."
        )
    elif trend_class == "flat":
        branch = "REVIEW"
        branch_reason = (
            f"Trend ratio {trend_ratio:.3f} is below {TREND_THRESHOLD} — the data is in the "
            f"flat regime where the order verdict cannot fire. That is a result, not a failure. "
            f"A person will confirm whether the flat verdict is the right answer or if there is "
            f"another framing."
        )
    else:
        branch = "INSTANT"
        branch_reason = (
            f"Block cosine {block_cos:.3f} is clear and trend ratio {trend_ratio:.3f} is above "
            f"{TREND_THRESHOLD}. The data clears the pre-flight gate — the engine will run and "
            f"the report will come back."
        )

    reasons.append(("branch", branch,
                    branch_reason + " This is the decision of the pre-flight gate, run before the engine."))

    # The summary is the DECISION, not whichever check happened to be listed
    # first. It used to take customer_lines[0], and saturation lines are appended
    # before the pre-flight ones -- so a file that sailed through the gate was
    # summarised to the customer as "Channel sensor_01: 97.3% ... saturated",
    # which is neither the branch nor the headline. Anything else worth saying
    # is appended after it, never in front of it.
    summary = branch_reason
    if n_saturated:
        # Minority saturation does not change the branch, but the INSTANT
        # message promises to say what was looked at and what was NOT, and a
        # saturated channel is precisely a channel that was not read.
        sat_names = ", ".join(n for n, _, r in saturation if not r)
        summary += (f" Note: {n_saturated} of {n_channels} channels are saturated on "
                    f"the standard scale and were NOT read -- {sat_names}. The verdict "
                    f"covers the other {n_readable}.")

    return {
        "branch": branch,
        "reasons": reasons,
        "pre_flight": pre_flight,
        "shape_problems": [],
        "saturation": saturation,
        "summary": summary,
        "n_channels": n_channels,
        "n_rows": n_rows,
        "n_saturated": n_saturated,
        "saturated_names": [n for n, _, r in saturation if not r],
        "readable_names": [n for n, _, r in saturation if r],
    }


# ---------------------------------------------------------------------------
# message templates
# ---------------------------------------------------------------------------

INSTANT_MSG = """\
Your data cleared the pre-flight gate. The engine ran and the report is below.

What was looked at: {n_channels} channels of {n_rows} rows, read as {n_windows} windows
of {win} consecutive samples per channel.

What was not looked at: nothing in the file was skipped on purpose. The engine reads
a random sample of windows (drawn once with a fixed seed, so re-running gives the same
windows and the same report), and reports what it finds in the order of your rows.

The report follows. If it says something you were not expecting, that is the finding —
not a mistake. The same pipeline runs on every file unchanged, and no number here was
fitted to your data.
"""

REVIEW_MSG = """\
Your data landed in the review queue. The pre-flight gate could not decide between
"read it" and "cannot read it" on the data as sent, so a person is looking at it.

What the gate found:
{reasons}

A person will look at this and either:
- run the engine with a different framing (fewer channels, separate runs, a refit scale),
- ask for different data, or
- confirm that the gate's conclusion is the right one.

You will hear back. The timeframe is: a person looks within one business day, and writes
back with the answer or the ask. No finding is promised — if the data cannot support an
answer, that is what you will get, with the reason.

Your upload is in the queue with file hash {file_hash}, submitted {submit_time}.
"""

CANNOT_MSG = """\
Your data could not be read as sent. Here is what the pre-flight gate found and why:

{reasons}

This is not a rejection — it is an answer plus a shopping list. Each check above names
the specific thing that would fix it. The most common fixes:

- Rows not in time order: sort by the first column (oldest first) and send again.
- No time column: send the file with a time stamp as the first column.
- Sorted by value, not time: send the same columns with their timestamps.
- Channels saturated: send one or two key channels on the same scale.
- Too few rows: send at least 200 rows; a few thousand is better.
- Columns with gaps or mostly empty: send the raw unsampled series, or fill the gaps.

If you send the thing the gate asked for, the engine can read it. If you are not sure
which column is which, tell us what each one measures and we will tell you how to send it.
"""


def _customer_reasons(triage_result):
    """The checks that decided this, in the customer's words.

    Not every check. The raw list carries one line per channel for saturation,
    so a 16-channel file rendered fifteen "readable" lines the customer did not
    ask for and could not act on, each prefixed with an internal gate slug
    ("saturation:", "block_cosine:"). What goes in the message is what failed,
    plus the one line that decided the branch, with the slugs stripped.
    """
    out, seen = [], set()
    for gate, verdict, text in triage_result.get("reasons", []):
        if verdict == "PASS" or gate == "branch":
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(f"- {text}")
    n_sat = triage_result.get("n_saturated", 0)
    if n_sat and len(triage_result.get("saturated_names", [])) > 3:
        # collapse a long saturation list into one line rather than one per channel
        out = [o for o in out if "extreme bins" not in o]
        out.insert(0, f"- {n_sat} of {triage_result.get('n_channels', 0)} channels are "
                      f"saturated on the standard measurement ladder and could not be "
                      f"read: {', '.join(triage_result['saturated_names'][:8])}"
                      + (", ..." if n_sat > 8 else ""))
    if not out:
        out.append(f"- {triage_result.get('summary', 'No specific check failed.')}")
    return "\n".join(out)


def render_message(branch, triage_result, meta):
    """
    Render the customer-facing message for the given branch.

    meta: dict with file_hash, submit_time (ISO string), email, notes_client.
    """
    if branch == "INSTANT":
        # count windows from the report if available, else estimate
        n_windows = triage_result.get("n_windows", "unknown")
        return INSTANT_MSG.format(
            n_channels=triage_result["n_channels"],
            n_rows=triage_result["n_rows"],
            n_windows=n_windows,
            win=WIN,
        )
    elif branch == "REVIEW":
        return REVIEW_MSG.format(
            reasons=_customer_reasons(triage_result),
            file_hash=meta.get("file_hash", "unknown")[:16],
            submit_time=meta.get("submit_time", "unknown"),
        )
    else:  # CANNOT
        return CANNOT_MSG.format(reasons=_customer_reasons(triage_result))


def branch_from_triage(triage_result):
    """Extract the branch from a triage result dict."""
    return triage_result["branch"]


# ---------------------------------------------------------------------------
# review queue
# ---------------------------------------------------------------------------

QUEUE_DIR = Path(__file__).resolve().parent / "review_queue"


def ensure_queue_dir():
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)


def enqueue(job_id, triage_result, meta, csv_text):
    """
    Write a review queue folder for REVIEW and CANNOT branches.

    Folder per job contains:
      - input.csv (the uploaded file)
      - meta.json (email, notes, file hash, submit time, client IP, branch, reasons)
      - pre_flight.json (the two-number pre-flight, saturation per channel, shape problems)
      - branch.txt (one line: INSTANT / REVIEW / CANNOT)
      - reasons.txt (every gate check with its verdict and explanation)

    Returns the folder path.
    """
    ensure_queue_dir()
    d = QUEUE_DIR / job_id
    d.mkdir(exist_ok=True)
    (d / "input.csv").write_text(csv_text)
    m = {
        "job": job_id,
        "branch": triage_result["branch"],
        "email": meta.get("email", ""),
        # Which of the sender's own clients this file is for. Without it, six
        # uploads from one contact are one pile -- which is the exact case
        # group_cannot_read_reasons() exists to answer.
        "client": meta.get("client", ""),
        "notes_client": meta.get("notes_client", ""),
        "notes_internal": meta.get("notes_internal", ""),
        "file_hash": meta.get("file_hash", ""),
        "submit_time": meta.get("submit_time", ""),
        "client_ip": meta.get("client_ip", ""),
        "n_channels": triage_result["n_channels"],
        "n_rows": triage_result["n_rows"],
        "reasons": triage_result["reasons"],
    }
    (d / "meta.json").write_text(json.dumps(m, indent=2))
    pf = {
        "pre_flight": triage_result["pre_flight"],
        "shape_problems": triage_result["shape_problems"],
        "saturation": triage_result["saturation"],
    }
    (d / "pre_flight.json").write_text(json.dumps(pf, indent=2))
    (d / "branch.txt").write_text(triage_result["branch"] + "\n")
    (d / "reasons.txt").write_text(
        "\n".join(f"{r[0]:20s} {r[1]:10s}  {r[2]}" for r in triage_result["reasons"])
    )
    return d


# ---------------------------------------------------------------------------
# Jim alert
# ---------------------------------------------------------------------------

ALERT_LOG = QUEUE_DIR / "alert_log.json"


def log_alert(job_id, triage_result, meta):
    """
    Append one alert line for REVIEW and CANNOT branches.

    Format: {job, branch, email, who, what, why, when}

    where:
      who = email or "no email"
      what = branch + n_channels + n_rows
      why = the branch reason (from the gate)
      when = submit_time
    """
    ensure_queue_dir()
    entry = {
        "job": job_id,
        "branch": triage_result["branch"],
        "email": meta.get("email", "") or "no email",
        "what": f"{triage_result['branch']} — {triage_result['n_channels']} channels, "
                f"{triage_result['n_rows']} rows",
        "why": triage_result["reasons"][-1][2] if triage_result["reasons"] else "",
        "when": meta.get("submit_time", ""),
        "file_hash": meta.get("file_hash", ""),
    }
    existing = {}
    if ALERT_LOG.exists():
        try:
            existing = json.loads(ALERT_LOG.read_text())
        except Exception:
            pass
    existing[job_id] = entry
    tmp = ALERT_LOG.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(existing, indent=2))
    import os
    os.replace(tmp, ALERT_LOG)
    return entry


def alert_summary():
    """
    Return a one-line-per-job summary of everything in the queue.

    One alert to Jim: who, what, which branch, and the single sentence explaining why.
    """
    if not ALERT_LOG.exists():
        return "No jobs in the review queue."
    try:
        data = json.loads(ALERT_LOG.read_text())
    except Exception:
        return "Alert log unreadable."
    lines = []
    for job_id, entry in data.items():
        lines.append(
            f"{entry['job']} | {entry['branch']:8s} | {entry['email']:30s} | "
            f"{entry['what']} | {entry['why'][:120]}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# expiry
# ---------------------------------------------------------------------------

import time as _time
import json  # late import to avoid top-level circularity


def expire_old_entries(max_age_days=7):
    """
    Nothing expires silently. A job with no answer after N days must resurface.

    This checks the alert log and returns jobs older than max_age_days that are still
    in REVIEW or CANNOT — those need a nudge.
    """
    if not ALERT_LOG.exists():
        return []
    try:
        data = json.loads(ALERT_LOG.read_text())
    except Exception:
        return []
    cutoff = _time.time() - max_age_days * 86400
    stale = []
    for job_id, entry in data.items():
        if entry["branch"] in ("REVIEW", "CANNOT"):
            try:
                when = _time.mktime(_time.strptime(entry["when"], "%Y-%m-%dT%H:%M:%S"))
            except (ValueError, TypeError):
                continue
            if when < cutoff:
                stale.append(entry)
    return stale


# ---------------------------------------------------------------------------
# multi-client: per-client labels
# ---------------------------------------------------------------------------

def extract_client_label(meta):
    """
    Extract a client label from the meta for multi-client handling.

    Craig Moody / Verdis Group case: a job carries a client label so multiple
    uploads from one contact do not become one undifferentiated pile.

    Returns the label string, or "unlabelled" if none was given.

    2026-08-26: this fell back to meta["label_col"], which is the name of the
    engine's LABEL COLUMN (e.g. "status") -- not a client. Craig's six uploads
    would all have grouped under "status", which is the undifferentiated pile
    this function exists to prevent, wearing a different name. There was also no
    `client` field on the upload form or on /api/run, so the label could never
    arrive at all; both now exist.
    """
    for key in ("client", "client_label"):
        v = (meta.get(key) or "").strip()
        if v:
            return v
    return "unlabelled"


def group_cannot_read_reasons(jobs):
    """
    For multi-client cases: group CANNOT-READ reasons by client label.

    Input: list of (job_id, triage_result, meta) tuples.
    Returns: dict client_label -> list of (job_id, reasons, channels, rows).

    One message, many reasons — if six files give six different CANNOT-READ
    reasons, that is a useful deliverable, not six separate messages.
    """
    groups = {}
    for job_id, triage_result, meta in jobs:
        label = extract_client_label(meta)
        if label not in groups:
            groups[label] = []
        groups[label].append({
            "job": job_id,
            "reasons": triage_result["reasons"],
            "n_channels": triage_result["n_channels"],
            "n_rows": triage_result["n_rows"],
            "branch": triage_result["branch"],
        })
    return groups


def render_multi_client_message(groups):
    """
    Render one message for a client with multiple CANNOT-READ uploads.

    Each group becomes one message listing all the jobs and their reasons.
    """
    parts = []
    for label, jobs in groups.items():
        reasons_set = set()
        for j in jobs:
            for r in j["reasons"]:
                if r[1] in ("FAIL", "VOID"):
                    reasons_set.add(r[2])
        parts.append(
            f"## {label}\n\n{j['n_channels']} channels across {len(jobs)} upload"
            f"{'s' if len(jobs) > 1 else ''}:\n\n"
            + "\n".join(
                f"- Job {j['job']}: {j['n_channels']} channels, {j['n_rows']} rows "
                f"({j['branch']})"
                for j in jobs
            )
            + "\n\nWhat is missing across these uploads:\n\n"
            + "\n".join(f"- {r}" for r in sorted(reasons_set))
            + "\n\nSend the specific thing each check above asks for, and we can re-read."
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# PDF test: confirm no dollar figure leaks from the engine
# ---------------------------------------------------------------------------

def check_no_dollar_in_report(report_text):
    """
    Confirm the report contains no dollar figure.

    Per prompt 07 and 09: if a dollar figure cannot be traced to the customer's
    own inputs, print no dollar figure. This is a gate, not a warning.
    """
    if "$" in report_text:
        # find the line
        for line in report_text.splitlines():
            if "$" in line:
                return False, f"Dollar figure found in report: {line.strip()}"
    return True, "No dollar figure in report."


if __name__ == "__main__":
    # quick self-test
    import sys
    if len(sys.argv) > 1:
        text = Path(sys.argv[1]).read_text()
        result = triage(text)
        print(f"branch: {result['branch']}")
        print(f"pre_flight: {result['pre_flight']}")
        print(f"n_channels: {result['n_channels']}, n_rows: {result['n_rows']}")
        print(f"summary: {result['summary'][:200]}")
