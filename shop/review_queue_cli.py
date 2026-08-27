#!/usr/bin/env python3
"""Jim's side of the triage queue: see what is waiting, answer it, and never
let one go quiet.

Why this file exists
--------------------
triage_gate.py builds the queue and writes an alert entry per job. It also has
`expire_old_entries()`, whose docstring says "Nothing expires silently. A job
with no answer after N days must resurface." Nothing called it. A resurfacing
rule that nobody runs is the silence it was written to prevent, so the sweep is
now a command and a timer (lattice24-triage.timer) runs it every morning.

Four commands:

    list                  what is waiting, oldest first
    show <job>            the file's numbers, the branch, and why
    answer <job> "text"   queue the reply to the customer
    sweep [--days N]      anything unanswered past N days -> one mail to Jim

`answer` does NOT send. It queues a draft through notify.queue_via_gate(), so
the message lands in mail_gate marked public_origin and waits for Jim to reply
SEND -- the same review path every other customer-facing message takes since
the authorisation fix of 2026-08-26. Answering here is composing, not sending.
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/home/voodoo/lattice24_pipeline")

import triage_gate as T          # noqa: E402
import notify                    # noqa: E402

ANSWERED = T.QUEUE_DIR / "answered.json"


def _load(p, default):
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001 -- a missing or half-written log is empty
        return default


def _answered():
    return _load(ANSWERED, {})


def _age_days(when):
    try:
        return (time.time() - time.mktime(time.strptime(when, "%Y-%m-%dT%H:%M:%S"))) / 86400
    except (ValueError, TypeError):
        return None


def cmd_list(_args):
    alerts = _load(T.ALERT_LOG, {})
    answered = _answered()
    rows = [e for j, e in alerts.items() if j not in answered]
    rows.sort(key=lambda e: e.get("when") or "")
    if not rows:
        print("Queue is empty — every upload has been answered.")
        return 0
    print(f"{len(rows)} waiting (oldest first)\n")
    for e in rows:
        age = _age_days(e.get("when", ""))
        age_s = f"{age:.1f}d" if age is not None else "  ?"
        print(f"  {e['job']}  {e['branch']:7s} {age_s:>6s}  {e.get('email','no email')}")
        print(f"      {e.get('what','')}")
        print(f"      {(e.get('why') or '')[:110]}")
    print(f"\nAnswer one with:  python3 {Path(__file__).name} answer <job> \"your reply\"")
    return 0


def cmd_show(args):
    d = T.QUEUE_DIR / args.job
    if not d.exists():
        print(f"No queue folder for {args.job}")
        return 1
    print(f"=== {args.job} ===")
    print((d / "branch.txt").read_text().strip())
    m = _load(d / "meta.json", {})
    print(f"from    : {m.get('email') or 'no email given'}")
    print(f"when    : {m.get('submit_time') or '(not recorded)'}")
    print(f"hash    : {m.get('file_hash','')[:32]}")
    if m.get("notes_client"):
        print(f"their note (verbatim):\n  {m['notes_client']}")
    if m.get("notes_internal"):
        print(f"internal note (verbatim):\n  {m['notes_internal']}")
    print()
    print("--- pre-flight numbers, already computed ---")
    print(json.dumps(_load(d / "pre_flight.json", {}), indent=2)[:2000])
    print()
    print("--- every check ---")
    print((d / "reasons.txt").read_text()[:4000])
    return 0


def cmd_answer(args):
    d = T.QUEUE_DIR / args.job
    if not d.exists():
        print(f"No queue folder for {args.job}")
        return 1
    m = _load(d / "meta.json", {})
    to = (m.get("email") or "").strip()
    if not to:
        print("That upload carried no email address, so there is nobody to answer.")
        print("Mark it done with --close if you have replied some other way.")
        if not args.close:
            return 1
    body = args.text
    if not body.endswith("\n"):
        body += "\n"
    body += f"\nYour job page: {notify.SITE}/report/{args.job}\n\nJim Jardine\nLattice24\n"
    did = None
    if to and not args.close:
        did = notify.queue_via_gate(
            to, f"Lattice24 — about your file (job {args.job})", body,
            f"answer to triage job {args.job} ({m.get('branch')}), written by Jim")
        print(f"Queued draft {did} to {to}.")
        # Say what will actually happen to THIS draft rather than reciting the
        # general rule: mail to Jim's own address is exempt from the
        # public-origin hold (it cannot reach anyone else), so telling him it
        # is held when it is about to send would be wrong.
        try:
            import mail_gate as _g
            held = _g.hold_reason({"to": to, "subject": "x", "body": "x",
                                   "public_origin": True})
        except Exception:  # noqa: BLE001
            held = "could not ask the gate; assume it is held"
        if held:
            print("It is HELD in mail_gate until you reply SEND to the [GATE] mail —")
            print(f"customer mail from a public upload never auto-sends. ({held})")
        else:
            print("The gate will send it on its next pass (within a minute) — this "
                  "address is your own, which is the one case that does not wait.")
    answered = _answered()
    answered[args.job] = {"when": time.strftime("%Y-%m-%dT%H:%M:%S"),
                          "draft": did, "closed_without_reply": bool(args.close)}
    T.ensure_queue_dir()
    tmp = ANSWERED.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(answered, indent=2))
    tmp.replace(ANSWERED)
    print(f"{args.job} marked answered; it will not resurface.")
    return 0


def cmd_sweep(args):
    """Anything unanswered past --days resurfaces, as ONE mail to Jim."""
    alerts = _load(T.ALERT_LOG, {})
    answered = _answered()
    stale = []
    for job, e in alerts.items():
        if job in answered:
            continue
        age = _age_days(e.get("when", ""))
        if age is None:
            # An entry with no timestamp cannot be aged, so it can never expire
            # and would sit in the queue forever. Surface it rather than skip
            # it -- an unagavailable date is a reason to look, not to ignore.
            stale.append((999.0, e))
        elif age >= args.days:
            stale.append((age, e))
    if not stale:
        print(f"sweep: nothing unanswered past {args.days} day(s).")
        return 0
    stale.sort(reverse=True)
    lines = [f"{len(stale)} upload(s) have been waiting more than {args.days} day(s) "
             f"with no answer sent.\n"]
    for age, e in stale:
        age_s = "date not recorded" if age >= 999 else f"{age:.1f} days"
        lines.append(f"  {e['job']}  {e['branch']}  waiting {age_s}")
        lines.append(f"      from : {e.get('email','no email')}")
        lines.append(f"      what : {e.get('what','')}")
        lines.append(f"      why  : {(e.get('why') or '')[:160]}")
        lines.append("")
    lines.append(f"See one:  python3 {HERE}/review_queue_cli.py show <job>")
    lines.append(f"Answer:   python3 {HERE}/review_queue_cli.py answer <job> \"...\"")
    text = "\n".join(lines)
    print(text)
    if args.mail:
        # Straight to Jim's own inbox, not through the gate: this is the machine
        # telling Jim about his own queue, not customer mail.
        notify._smtp_send(notify.JIM_ADDRESS,
                          f"[TRIAGE] {len(stale)} upload(s) waiting more than {args.days}d",
                          text, {"X-Lattice24-Triage": "sweep"})
        print(f"\n(mailed to {notify.JIM_ADDRESS})")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    s = sub.add_parser("show"); s.add_argument("job"); s.set_defaults(fn=cmd_show)
    a = sub.add_parser("answer"); a.add_argument("job"); a.add_argument("text")
    a.add_argument("--close", action="store_true",
                   help="mark answered without queuing mail (already replied elsewhere)")
    a.set_defaults(fn=cmd_answer)
    w = sub.add_parser("sweep"); w.add_argument("--days", type=float, default=2.0)
    w.add_argument("--mail", action="store_true", help="send the summary to Jim")
    w.set_defaults(fn=cmd_sweep)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
