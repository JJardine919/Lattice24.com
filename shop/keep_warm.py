#!/usr/bin/env python3
"""Keep the free-tier Render instance awake during the hours clicks arrive.

Why this exists
---------------
Measured 2026-08-26 by curl: https://lattice24-com.onrender.com/shop/upload.html
returns 200 in 21.3 s cold and 0.20 s warm. Render's free tier suspends a web
service after ~15 minutes with no request; the next request pays a full
container cold start. A prospect who clicks a link in one of Jim's emails and
watches a blank tab for twenty seconds concludes the thing is broken, and he is
not wrong to -- twenty seconds is what broken looks like.

Why not simply ping around the clock
------------------------------------
The free tier is metered in instance-hours (750/month, account-wide). Pinging
24/7 holds the container up for ~730 h and spends essentially the entire free
allowance on hours when nobody is clicking -- and if the cap is reached the
service stops, which is a worse outage than a cold start. So this runs on a
business-hours window only (see lattice24-warm.timer): ~12 h x ~30 d = ~360 h,
under half the budget, covering the hours outreach is actually read.

Outside the window the first visitor still pays the cold start, and nothing
served by this host can warn him: the wait happens BEFORE upload.html loads, so
an interstitial on that page is too late by definition. The warning lives in the
drip templates instead (outreach/send_drip.py), which is the only surface that
reaches him earlier.

Costs nothing: one HTTP GET, no paid tier, no new host.
"""
import datetime as dt
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

URL = "https://lattice24-com.onrender.com/shop/upload.html"
LOG = Path(__file__).resolve().parent / "logs" / "keep_warm.log"
TIMEOUT = 90  # a cold start is ~22 s; give a stalled one room before calling it


def main():
    started = time.monotonic()
    try:
        req = urllib.request.Request(URL, method="GET",
                                     headers={"User-Agent": "lattice24-keepwarm/1"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            code, note = r.status, ""
    except urllib.error.HTTPError as exc:
        code, note = exc.code, "http-error"
    except Exception as exc:  # noqa: BLE001 -- a dead network is a logged fact
        code, note = 0, f"{type(exc).__name__}: {exc}"
    elapsed = time.monotonic() - started
    # A ping over ~5 s means it was asleep and this call paid the cold start --
    # worth seeing in the log, because a window full of them means the schedule
    # is not actually holding it up.
    state = "warm" if elapsed < 5 else ("cold-start-paid" if code == 200 else "fail")
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a") as fh:
        fh.write(f"{dt.datetime.now().isoformat(timespec='seconds')}  "
                 f"{code}  {elapsed:6.2f}s  {state}  {note}\n")
    print(f"{code} {elapsed:.2f}s {state} {note}".strip())
    return 0 if code == 200 else 1


if __name__ == "__main__":
    sys.exit(main())
