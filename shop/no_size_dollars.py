#!/usr/bin/env python3
"""Guard: no page may compute a number from a file's SIZE.

The defect this exists to make impossible
-----------------------------------------
A PDF was dropped on lattice24.com/send-data.html and the page answered
"Estimated saved: $13 + sensor report (16 channels)". Neither number came from
the file. Reproduced from the LIVE page's own source, 2026-08-26:

    function estimate(file){ // fake $ from rows*cols heuristic
      const rows=Math.min(8000, Math.max(300, (file.size||3000)/18|0));
      const kwh=rows*0.018;   // placeholder
      const price=0.14;       // $/kWh blended
      const ecr=0.72;         // conservative
      return Math.round(kwh*price*ecr*1.2);
    }

`file.size` is the ONLY input. Everything else is a constant. So:

  * any file between 103,356 and 111,600 bytes prints exactly $13 -- a PDF, a
    CSV, a JPEG, a photo of a cat;
  * `rows` saturates at 8000, so EVERY file above 144 KB prints $17. That is
    the measured "$17 for pure noise and $17 for real refinery data";
  * zipping a file halves its size and halves the quoted saving;
  * "16 channels" was never counted at all -- it is a hardcoded string in the
    HTML next to the figure.

Removing the function is not the fix, because the next person writes it again.
The fix is a check that fails the build if any served page contains an
arithmetic path from a file's size to a displayed number. Run it against every
page this project serves, on a timer or by hand:

    python3 no_size_dollars.py
    python3 no_size_dollars.py --url https://lattice24.com/send-data.html
"""
import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: Pages this project serves or owns.
TARGETS = [
    HERE / "public" / "upload.html",
    HERE / "public" / "compute.html",
    HERE / "send-data.html",
    Path("/home/voodoo/lattice24_site/send-data.html"),
    Path("/home/voodoo/lattice24_site/index.html"),
]

#: A size token reaching a number the page shows. Comments are stripped before
#: matching, so the removal note in the fixed file -- which quotes the defect on
#: purpose -- does not read as the defect.
#: A size token used as an INPUT, not as a limit. `f.size > 5_000_000` is a
#: cap on the upload and is fine; `file.size / 18` is the defect. The
#: difference is the operator that follows, so comparisons are excluded rather
#: than the check being loosened.
SIZE_TOKEN = re.compile(
    r"\b(?:file\.size|\w+\.size|byteLength|contentLength|content_length)\s*"
    r"(?![<>=!]|\s*\)?\s*[<>])")

#: Only CURRENCY sinks. This used to include `toLocaleString`, which is how a
#: row count gets a thousands separator -- so a page that capped upload size and
#: formatted a row count was flagged for a defect it did not have. A check that
#: cries wolf on the honest page gets switched off, and then the real one lands.
MONEY_SINK = re.compile(
    r"(Estimated saved|'\$'\s*\+|\"\$\"\s*\+|`\$\$\{|\$'\s*\+"
    r"|\bestimate\s*\(|per\s*tonne|savings?\s*[:=]\s*\$)", re.I)


def strip_comments(text):
    """// line comments, /* block */ comments, and <!-- html -->."""
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"(?m)^\s*//.*$", " ", text)
    text = re.sub(r"(?<=[;{}\s])//[^\n]*", " ", text)
    return text


def check(name, text):
    """Return a list of findings. Empty list means clean."""
    body = strip_comments(text)
    out = []
    # Line-level is too narrow: on the live page the size token and the money
    # sink are eleven lines apart, inside one function. Match at the level of a
    # <script> block -- if a script both reads a file's size and puts a figure
    # on the screen, that is the shape, wherever the two sit inside it.
    blocks = re.findall(r"<script\b[^>]*>(.*?)</script>", body, flags=re.S | re.I)
    if not blocks:
        blocks = [body]
    for blk in blocks:
        sz = SIZE_TOKEN.search(blk)
        mn = MONEY_SINK.search(blk)
        if sz and mn:
            line = body[:body.index(blk) + sz.start()].count("\n") + 1
            out.append((line,
                        "a script reads the file's SIZE and puts a figure on the "
                        "screen — no page may derive a number from how big a file is",
                        blk[max(0, sz.start() - 60):sz.start() + 100].strip()[:160]))
    for i, line in enumerate(body.splitlines(), 1):
        if SIZE_TOKEN.search(line) and MONEY_SINK.search(line):
            out.append((i, "size feeds a displayed figure", line.strip()[:160]))
    # The hardcoded channel count: a number of channels the page did not count.
    for m in re.finditer(r"sensor report \((\d+) channels\)", body):
        out.append((body[:m.start()].count("\n") + 1,
                    "channel count is a hardcoded string, not counted from the file",
                    m.group(0)))
    # A dollar figure shown before any server round-trip.
    if re.search(r"Estimated saved", body):
        out.append((body[:body.index("Estimated saved")].count("\n") + 1,
                    "page shows a saving before the file has been read",
                    "Estimated saved"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", action="append", default=[],
                    help="also fetch and check a live URL")
    args = ap.parse_args()
    bad = 0
    for t in TARGETS:
        if not t.exists():
            print(f"  skip   {t} (not on disk)")
            continue
        findings = check(t.name, t.read_text(errors="replace"))
        if findings:
            bad += 1
            print(f"  FAIL   {t}")
            for ln, why, snip in findings:
                print(f"           line {ln}: {why}\n             {snip}")
        else:
            print(f"  clean  {t}")
    for url in args.url:
        import urllib.request
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                text = r.read().decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR  {url}: {exc}")
            bad += 1
            continue
        findings = check(url, text)
        if findings:
            bad += 1
            print(f"  FAIL   {url}  (LIVE)")
            for ln, why, snip in findings:
                print(f"           line {ln}: {why}\n             {snip}")
        else:
            print(f"  clean  {url}  (LIVE)")
    print(f"\n{bad} page(s) failed.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
