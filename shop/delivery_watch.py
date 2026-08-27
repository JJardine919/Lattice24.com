#!/usr/bin/env python3
"""Watchdog over the delivery path. Silence is the thing it is looking for.

notify.py queues customer mail as a mail_gate draft. The gate is autonomous and
normally empties within a minute, but a draft can sit: a hold rule can catch it,
the daily cap can defer it, the gate's own service can be down. A receipt that
never leaves is exactly the failure this whole build exists to remove, and it
looks like nothing at all from the server's side -- the POST returned 200 and
the log says "queued".

So this checks the one thing the server cannot see: did the draft actually go?

  python3 delivery_watch.py            # report, and mail Jim if anything is stuck
  python3 delivery_watch.py --quiet    # report only, send nothing

Stuck means: queued longer than STUCK_MINUTES and still not in drafts/sent/.
Each stuck draft is reported ONCE per state change, tracked in
delivery_watch_state.json -- a watchdog that mails every run is a watchdog that
gets filtered.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import notify

ROOT = Path(__file__).resolve().parent
STATE = ROOT / "delivery_watch_state.json"
STUCK_MINUTES = 10


def read_log():
    if not notify.LOG.exists():
        return []
    out = []
    for line in notify.LOG.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


UNKNOWN_AGE = float("inf")


def age_minutes(stamp):
    """Minutes since `stamp`, or UNKNOWN_AGE if it cannot be read.

    This returned 0.0 on a parse failure, which meant an unreadable timestamp
    read as "queued just now" and could never be flagged STUCK -- a control that
    defaults to pass, exactly the shape of feedback_controls_default_open. A
    record whose age is unknown is the LAST record to assume is fine: the log
    line exists, so a message was queued, and nothing here can show it left.
    Fail towards the alarm, not away from it.
    """
    if not stamp:
        return UNKNOWN_AGE
    try:
        return (time.time()
                - datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S").timestamp()) / 60
    except (ValueError, TypeError):
        return UNKNOWN_AGE


def survey():
    """Every queued customer message and where it actually ended up."""
    rows = []
    for rec in read_log():
        if rec.get("event") not in ("receipt_queued", "outcome_queued"):
            continue
        did = rec.get("draft")
        state = notify.draft_state(did)
        mins = age_minutes(rec.get("at"))
        rows.append({
            "draft": did,
            "job": rec.get("job"),
            "to": rec.get("to"),
            "event": rec["event"],
            "queued_at": rec.get("at"),
            "age_min": None if mins == UNKNOWN_AGE else round(mins, 1),
            "state": state,
            "stuck": state == "queued" and mins > STUCK_MINUTES,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description="Watch the shop's delivery path.")
    ap.add_argument("--quiet", action="store_true", help="report only, send nothing")
    args = ap.parse_args()

    rows = survey()
    for r in rows:
        flag = "  <-- STUCK" if r["stuck"] else ""
        age = "  unreadable" if r["age_min"] is None else f"{r['age_min']:8.1f} min"
        print(f"  {r['draft']}  {r['state']:8s} {age}  "
              f"job {r['job']}  {r['event']}{flag}")
    stuck = [r for r in rows if r["stuck"]]
    rejected = [r for r in rows if r["state"] == "rejected"]
    gone = [r for r in rows if r["state"] == "gone"]
    alarm = notify.ALARM.exists()

    print(f"-- {len(rows)} customer message(s): "
          f"{sum(1 for r in rows if r['state'] == 'sent')} sent, "
          f"{len(stuck)} stuck, {len(rejected)} rejected, {len(gone)} unaccounted")
    if alarm:
        print(f"-- DELIVERY_ALARM.txt EXISTS: {notify.ALARM}")

    problems = stuck + rejected + gone
    if not problems and not alarm:
        print("-- delivery path clean")
        return 0

    key = json.dumps(sorted(r["draft"] for r in problems) + [str(alarm)])
    try:
        seen = json.loads(STATE.read_text()).get("last")
    except Exception:  # noqa: BLE001
        seen = None
    if args.quiet or key == seen:
        print("-- already reported; not re-sending")
        return 1

    lines = [f"  {r['draft']}  {r['state']}  "
             f"{'age unreadable' if r['age_min'] is None else format(r['age_min'], '.0f') + ' min'}  "
             f"job {r['job']}  -> {r['to']}" for r in problems]
    body = ("The shop's delivery path has messages that did not go out.\n\n"
            + "\n".join(lines) +
            ("\n\nDELIVERY_ALARM.txt exists — read it:\n"
             f"  {notify.ALARM}\n" if alarm else "\n") +
            f"\nQueue: {notify.GATE_DRAFTS}\n"
            f"Log  : {notify.LOG}\n\n"
            "A customer is waiting on each of these.\n")
    try:
        notify._smtp_send(notify.JIM_ADDRESS,
                          f"[SHOP] {len(problems)} customer message(s) did not go out",
                          body, {"X-Lattice24-Shop": "watchdog"})
        notify.log(event="watchdog_alerted", stuck=len(stuck),
                   rejected=len(rejected), gone=len(gone), alarm=alarm)
        STATE.write_text(json.dumps({"last": key,
                                     "at": time.strftime("%Y-%m-%dT%H:%M:%S")}))
        print("-- alerted Jim")
    except Exception as exc:  # noqa: BLE001
        print(f"-- COULD NOT ALERT: {exc}")
        notify.raise_alarm("watchdog", "watchdog could not reach Jim", str(exc))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
