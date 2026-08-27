#!/usr/bin/env python3
"""Delivery: the customer hears back, and Jim finds out. Nothing is silent.

Before 2026-08-26 server.py wrote a job directory and stopped. The upload page
said "Reply comes by email" and the confirmation said "Watch your inbox"; no
message of any kind was generated, for the customer or for Jim. This module is
the part that was missing.

Three messages, in this order:

  1. receipt(job, meta)      -> the customer, the instant the file lands.
  2. alert_jim(job, meta)    -> Jim, same instant. A stranger uploading a file
                                is a sales event; he must not have to poll a
                                directory to learn about it.
  3. outcome(job, meta)      -> the customer, when the engine finishes. Sends
                                the report ONLY if REPORT_TRUSTWORTHY is true.
                                It is false today (see ~/BUILDS/SHOP_TEST_2026-08-26.md
                                §4.1: the Result section prints "no sequence
                                structure" for data saturated with it), so a
                                holding message goes instead. Do not flip that
                                flag until the report's headline is right.

ROUTING -- deliberate, and not symmetric:

  * Customer mail goes through mail_gate (~/lattice24_pipeline/mail_gate.py) by
    dropping a draft JSON in its drafts/ directory. Until 2026-08-26 those
    drafts carried "authorized", which cleared the never-re-mail hold and the
    daily cap -- for a recipient chosen by an unauthenticated public POST.
    They now carry "public_origin" instead, and the gate holds every
    public-origin draft for an explicit human SEND. Customer mail is therefore
    NOT autonomous: it waits for Jim. "FINAL" stays absolute in the gate;
    nothing here ever writes that word.
    The customer is not left waiting on that decision -- the browser gets the
    job page in the same response, and /report/<job> serves the report whether
    or not the mail has gone out. The email is a courtesy on top of a result
    they already hold, which is what makes holding it acceptable.

  * Jim's alert does NOT go through the gate. The gate exists to stop the
    machine mailing strangers unbidden; a notification from Jim's own account
    to Jim's own inbox is not that, and putting it behind a 60 s poll would
    mean the sales event arrives after the customer's receipt does.

  * Drafts are written to a NON-matching temp name and renamed into place.
    mail_gate.one_pass() json.loads() each d-*.json OUTSIDE its try block, so a
    half-written file would raise straight through serve()'s loop and kill the
    gate. Atomic rename or nothing.

FAILURE is the defect being fixed here, so it is never swallowed:

  * every attempt and every outcome appends to delivery_log.jsonl, written
    before anything that can throw;
  * a failure writes DELIVERY_ALARM.txt at the top of this directory and
    DELIVERY_FAILED.txt inside the job, and tries to reach Jim by mail;
  * if the mail path itself is down, the files remain and delivery_watch.py
    reports them.
"""

import json
import re
import os
import secrets
import smtplib
import time
import traceback
from email.message import EmailMessage
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "delivery_log.jsonl"
ALARM = ROOT / "DELIVERY_ALARM.txt"

GATE_DRAFTS = Path(os.environ.get(
    "LATTICE24_GATE_DRAFTS", "/home/voodoo/lattice24_pipeline/drafts"))

# ---------------------------------------------------------------- addresses
#
# The address a customer replies to. NOT jim@lattice24.com: that domain's MX is
# Namecheap's free forwarder with the Status toggle OFF and no active rule, so
# mail to it is refused at the remote server -- verified 2026-08-26 by sending
# to it and reading the bounce: "554 5.7.1 <jim@lattice24.com>: Relay access
# denied". The address below is the account this machine authenticates as, is
# the From header on every message it sends (confirmed in the Sent folder), and
# is the inbox mail_gate and watch_replies.py already poll.
#
# When Jim turns Namecheap's Mail Settings > Email Forwarding Status toggle ON
# (the forwarding target is already set -- the apex TXT record reads
# "jjj101147@gmail.com"), change this one line and the page copy follows.
REPLY_ADDRESS = os.environ.get("LATTICE24_REPLY_ADDRESS", "jjj101147@gmail.com")

# Where the sales alert lands. Jim's working inbox, same account.
JIM_ADDRESS = os.environ.get("LATTICE24_JIM_ADDRESS", "jjj101147@gmail.com")

SENDER_NAME = "Jim Jardine"
SMTP_HOST = os.environ.get("LATTICE24_SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("LATTICE24_SMTP_PORT", "465"))

# The report's headline statistic is inverted (SHOP_TEST_2026-08-26 §4.1) and
# three further claims in it are false (§4.4, §4.8, §4.11). Until prompt 07
# closes those, no report is attached to anything.
# Flipped to True 2026-08-26 on Jim's explicit instruction, after prompt 07
# closed all four SHOP_TEST defects and the fix was verified INDEPENDENTLY here
# on 2,000 rows x 16 channels rather than taken on the peer session's word:
#   sine period 8 + noise .... 16/16 channels read   (the accept path fires)
#   iid noise ................  0/16                 (not a pass-by-default control)
#   real refinery file, ordered  4/16
#   same file, rows SHUFFLED .  0/16                 (the headline is the right way round)
# It read p=1.0000 ordered vs 0.0250 shuffled before -- exactly backwards.
#
# Flipping this authorises attaching reports to outgoing customer mail AND opens
# the report page in server.py. Do not flip it back and forth casually: set it
# False again the moment any headline claim in the report is in doubt, and
# rewrite the message below to match, because a message that describes a defect
# that no longer exists is its own dishonesty.
REPORT_TRUSTWORTHY = True

SITE = os.environ.get("SHOP_SITE",
    "https://lattice24-com.onrender.com")
# NOT https://lattice24.com. The apex is GitHub Pages; /report/<job> there is a
# hard 404, so every job link this machine has ever emailed was dead. Browser
# uploads only survived because upload.html strips the prefix client-side, which
# does nothing for a link that is copied, forwarded or read out of an email.
# This is the base URL of the deployment SERVING the job -- override SHOP_SITE
# per deployment; a job written by one server does not exist on another.


# ---------------------------------------------------------------- log & alarm

def log(**rec):
    """Append one delivery record. Called before anything that can throw."""
    rec["at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    line = json.dumps(rec, ensure_ascii=False)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return rec


def raise_alarm(job, what, detail):
    """Put the failure somewhere Jim cannot miss it, on disk, unconditionally.

    Runs before any attempt to mail about it, because the mail path is the
    thing most likely to be broken when this is called.
    """
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    text = (f"[{stamp}] DELIVERY FAILED\n"
            f"  job    : {job}\n"
            f"  message: {what}\n"
            f"  detail : {detail}\n"
            f"  action : the customer has NOT heard back. Reply by hand, then\n"
            f"           delete this file.\n\n")
    with open(ALARM, "a", encoding="utf-8") as fh:
        fh.write(text)
    d = ROOT / "jobs" / job
    if d.is_dir():
        # Append, never write_text: a job can fail twice (receipt, then
        # outcome) and overwriting made the first failure vanish from the job
        # directory -- found in the deliberate-failure test, 2026-08-26.
        with open(d / "DELIVERY_FAILED.txt", "a", encoding="utf-8") as fh:
            fh.write(text)
    log(event="failure", job=job, message=what, detail=str(detail)[:2000])
    # Last resort: tell Jim by mail. If this throws too, the files above stand.
    try:
        _smtp_send(JIM_ADDRESS,
                   f"[SHOP FAILURE] {what} — job {job}",
                   text + f"Alarm file: {ALARM}\n")
    except Exception as exc:  # noqa: BLE001 - the alarm file is the guarantee
        log(event="failure_alert_failed", job=job, detail=str(exc)[:500])


# ---------------------------------------------------------------- smtp

def _credentials():
    """Credentials, environment first.

    This used to go straight to /home/voodoo/lattice24_pipeline and import
    mailbox from it. That path does not exist inside the Render container, so
    every send from the deployed server raised ImportError -- and because the
    caller swallowed it, mail silently went nowhere. Read the environment
    first, exactly as mailbox.load_credentials() does, so the container works
    with LATTICE24_GMAIL_USER + LATTICE24_GMAIL_APP_PASSWORD set as env vars
    and the laptop keeps working unchanged.
    """
    pw = os.environ.get("LATTICE24_GMAIL_APP_PASSWORD")
    if pw:
        return (os.environ.get("LATTICE24_GMAIL_USER", ""), pw)
    # The laptop fallback, guarded. Unguarded, this `import mailbox` resolved to
    # PYTHON'S STDLIB mailbox inside the container -- the inserted path does not
    # exist there, so nothing shadows it -- and the operator saw
    # "module 'mailbox' has no attribute 'load_credentials'", which names the
    # wrong cause. That is the same class of defect as the "check row ordering"
    # page: a true-sounding message pointing at something that is not broken.
    # Say what is actually wrong instead.
    import sys
    pipeline = "/home/voodoo/lattice24_pipeline"
    if not os.path.isfile(os.path.join(pipeline, "mailbox.py")):
        raise RuntimeError(
            "no mail credentials: set LATTICE24_GMAIL_USER and "
            "LATTICE24_GMAIL_APP_PASSWORD in this deployment's environment. "
            f"(The laptop fallback needs {pipeline}/mailbox.py, which is not "
            "present here -- this is a deployment that has never been given "
            "credentials, not a broken import.)")
    sys.path.insert(0, pipeline)
    import mailbox as mailbox_mod
    return mailbox_mod.load_credentials()


def _smtp_send(to, subject, body, headers=None):
    """One message, straight out. Used for Jim's own mail only."""
    user, password = _credentials()
    em = EmailMessage()
    em["From"] = f"{SENDER_NAME} <{user}>"
    em["To"] = to
    em["Subject"] = subject
    em["Reply-To"] = REPLY_ADDRESS
    for k, v in (headers or {}).items():
        em[k] = v
    em.set_content(body)
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=60) as smtp:
        smtp.login(user, password)
        smtp.send_message(em)
    return user


# ---------------------------------------------------------------- gate drafts

#: One address, and nothing that is not part of one.
#:
#: An allowlist, deliberately, because the thing being defended against is an
#: address that smuggles a second recipient or a header past the check. A
#: denylist of separators has to enumerate every encoding of every separator --
#: comma, semicolon, space, tab, CR, LF, %0A, %0D, angle brackets, a quoted
#: local part with a comma inside it -- and it is wrong the first time one is
#: missed. Nothing outside this character set can express any of them.
ADDRESS_RE = re.compile(r"^[A-Za-z0-9!#$%&'*+/=?^_`{|}~.-]{1,64}"
                        r"@[A-Za-z0-9.-]{1,255}\.[A-Za-z]{2,63}$")


def valid_single_address(addr):
    """True only for one ordinary address with no room for a second."""
    a = (addr or "").strip()
    if a != (addr or ""):
        # Leading/trailing whitespace is how a CRLF-injected address arrives
        # looking innocent after a naive .strip(). Refuse rather than repair.
        return False
    if len(a) > 254 or a.count("@") != 1:
        return False
    return bool(ADDRESS_RE.match(a))


def queue_via_gate(to, subject, body, why_queued):
    """Drop a draft for mail_gate, marked as public-origin. Returns the id.

    2026-08-26. This used to write `authorized`, which made mail_gate's
    hold_reason() return None above the never-re-mail check and above the daily
    cap. The recipient came from the body of an unauthenticated public
    POST /api/run -- so a stranger chose the address, and the machine wrote the
    flag that turned the protections off for it. That is a relay signed
    "Jim Jardine / Lattice24", not a receipt.

    Nothing here authorises anything now. `public_origin` tells the gate this
    draft came from an endpoint on the internet, and the gate holds every one
    of them for a human SEND. The flag is a confession, not a permission: it
    can only ever make the gate stricter, so a caller who could somehow set it
    gains nothing.

    The temp name deliberately does NOT match d-*.json, so the gate can never
    glob a partially written file.
    """
    if not valid_single_address(to):
        # Refuse to write the draft at all. A gate that is asked to send to a
        # malformed recipient has already been handed something it should never
        # have been asked to handle.
        raise ValueError(f"refusing to queue mail to a non-address: {to!r}")
    GATE_DRAFTS.mkdir(parents=True, exist_ok=True)
    did = f"d-{secrets.token_hex(4)}"
    payload = json.dumps({
        "to": to,
        "subject": subject,
        "body": body,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "public_origin": True,
        "origin": "lattice24_machine/notify.py via POST /api/run",
        "why_queued": why_queued,
        "source": "lattice24_machine/notify.py",
    }, indent=2)
    tmp = GATE_DRAFTS / f".staging-{did}.tmp"
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, GATE_DRAFTS / f"{did}.json")
    return did


def draft_state(did):
    """queued | sent | rejected | gone — where a draft actually ended up."""
    if (GATE_DRAFTS / f"{did}.json").exists():
        return "queued"
    if (GATE_DRAFTS / "sent" / f"{did}.json").exists():
        return "sent"
    if (GATE_DRAFTS / "rejected" / f"{did}.json").exists():
        return "rejected"
    return "gone"


# ---------------------------------------------------------------- file facts

def file_facts(meta):
    """What we can say about the upload without reading any engine output."""
    csv = meta.get("csv") or ""
    lines = [ln for ln in csv.splitlines() if ln.strip()]
    header = lines[0] if lines else ""
    sep = "\t" if header.count("\t") > header.count(",") else ","
    cols = [c.strip() for c in header.split(sep)] if header else []
    return {
        "bytes": len(csv.encode("utf-8")),
        "rows": max(0, len(lines) - 1),
        "cols": len(cols),
        "col_names": cols,
    }


def _facts_block(f):
    names = ", ".join(f["col_names"][:16]) or "(no header row read)"
    if len(f["col_names"]) > 16:
        names += f", +{len(f['col_names']) - 16} more"
    return (f"  rows     : {f['rows']:,}\n"
            f"  columns  : {f['cols']}\n"
            f"  size     : {f['bytes']:,} bytes\n"
            f"  columns read: {names}\n")


# ---------------------------------------------------------------- messages

def receipt(job, meta):
    """Message 1 -- the customer, immediately. This one must never fail."""
    to = (meta.get("email") or "").strip()
    f = file_facts(meta)
    subject = f"Lattice24 — your file is in (job {job})"
    body = (
        "Your file arrived and is being read now.\n\n"
        "What we received\n"
        + _facts_block(f) +
        f"  job id   : {job}\n"
        # The submitter's own text is NOT echoed into a message this machine
        # sends to an address the submitter also chose. That combination --
        # attacker picks the recipient AND the body, Jim's account signs it --
        # is a mail relay, not a receipt. Jim sees the note verbatim in
        # alert_jim(); the customer already knows what they typed.
        "\nWhat happens next\n"
        "  The engine runs on it — usually under a minute, at most three.\n"
        "  You get a second email from this address when it finishes.\n"
        f"  Your job page: {SITE}/report/{job}\n\n"
        "Nothing is charged, and your file is not shared with anyone.\n\n"
        f"Reply to this email and it reaches Jim directly: {REPLY_ADDRESS}\n\n"
        "Jim Jardine\nLattice24\n")
    if not to:
        log(event="receipt_skipped", job=job, reason="no email address given")
        return None
    try:
        did = queue_via_gate(
            to, subject, body,
            f"shop upload receipt for job {job} — transactional reply to a "
            "customer who just submitted a file, not outreach")
        log(event="receipt_queued", job=job, to=to, draft=did, subject=subject)
        return did
    except Exception as exc:  # noqa: BLE001
        raise_alarm(job, "customer receipt could not be queued",
                    f"{exc}\n{traceback.format_exc()}")
        return None



def _self_send_budget(kind, cap_env, default):
    """Shared daily ceiling for the direct-SMTP paths that never reach mail_gate.

    mail_gate has SELF_SEND_CAP, but neither alert_jim nor mail_upload_to_jim
    goes through mail_gate -- they call _smtp_send directly. Capping the gate
    on 2026-08-26 therefore fixed nothing: alert_jim stayed uncapped and
    delivery_log.jsonl recorded 270 alerts that day, inside the 549 sends that
    tripped Gmail's 500/day limit and took the account offline.

    Counts every ATTEMPT, not every success. Counting successes means a failing
    send never advances the counter and retries are unbounded -- which is the
    exact state a 550 puts this in.

    Returns (ok, n, cap). The counter file lives on the container's ephemeral
    disk, so it resets on a cold start: this is a brake on a runaway loop
    inside one uptime, not an accounting record.
    """
    cap = int(os.environ.get(cap_env, str(default)))
    stamp = time.strftime("%Y-%m-%d")
    path = Path(os.environ.get("SHOP_SELF_SEND_COUNTER",
                               "/tmp/lattice24_self_send")).with_suffix(f".{kind}")
    n = 0
    try:
        day, got = path.read_text().split(None, 1)
        if day == stamp:
            n = int(got)
    except Exception:  # noqa: BLE001 - no counter yet is n=0
        pass
    if n >= cap:
        return False, n, cap
    try:
        path.write_text(f"{stamp} {n + 1}")
    except Exception as exc:  # noqa: BLE001 - bookkeeping must not block the send
        print(f"self-send counter not written: {exc}", flush=True)
    return True, n + 1, cap


def alert_jim(job, meta, client_ip="unknown"):
    """Message 2 -- Jim, immediately. A stranger uploaded a file."""
    ok, n, cap = _self_send_budget("alert", "SHOP_ALERT_DAILY_CAP", 80)
    if not ok:
        print(f"ALERT: daily cap of {cap} reached — job {job} NOT alerted. "
              f"The upload is still on disk and on its report page.", flush=True)
        return None
    f = file_facts(meta)
    who = meta.get("email") or "(no address given)"
    subject = f"[SHOP] upload from {who} — job {job} ({n}/{cap} today)"
    body = (
        "Someone uploaded a file to the shop.\n\n"
        f"  from     : {who}\n"
        f"  ip       : {client_ip}\n"
        f"  job      : {job}\n"
        f"  market   : {meta.get('market', '')}\n"
        + _facts_block(f) +
        "\nTheir note, verbatim:\n"
        f"  {meta.get('notes_client') or '(none)'}\n"
        "\nInternal note, verbatim:\n"
        f"  {meta.get('notes_internal') or '(none)'}\n"
        f"\nReport: {SITE}/report/{job}\n"
        f"On disk: {ROOT / 'jobs' / job}\n\n"
        "They have been sent a receipt from this address. Their reply lands\n"
        f"in this inbox ({REPLY_ADDRESS}).\n")
    try:
        _smtp_send(JIM_ADDRESS, subject, body,
                   {"X-Lattice24-Shop": "upload"})
        log(event="jim_alert_sent", job=job, to=JIM_ADDRESS, subject=subject)
        return True
    except Exception as exc:  # noqa: BLE001
        raise_alarm(job, "Jim's upload alert could not be sent",
                    f"{exc}\n{traceback.format_exc()}")
        return False


def outcome(job, meta):
    """Message 3 -- the report, or a holding message that promises nothing."""
    to = (meta.get("email") or "").strip()
    if not to:
        log(event="outcome_skipped", job=job, reason="no email address given")
        return None
    f = file_facts(meta)
    engine_ok = bool(meta.get("ok"))

    if not engine_ok:
        subject = f"Lattice24 — your file could not be read (job {job})"
        body = (
            f"The engine could not read job {job}.\n\n"
            "What it said:\n"
            f"  {(meta.get('err') or 'no detail recorded').strip()[:600]}\n\n"
            "The most common cause is row ordering — the engine reads sequence,\n"
            "so rows have to be in time order.\n\n"
            f"Reply to this email with what the file is and Jim will look at it: "
            f"{REPLY_ADDRESS}\n\n"
            "Jim Jardine\nLattice24\n")
        why = "engine error"
    elif REPORT_TRUSTWORTHY:
        subject = f"Lattice24 — your report is ready (job {job})"
        body = (
            f"Your file ran clean. Job {job}.\n\n"
            "What we read\n"
            + _facts_block(f) +
            f"\nYour report: {SITE}/report/{job}\n\n"
            "How to read it\n"
            "  The report says which of your channels carry information in the\n"
            "  ORDER of your rows, not just in their values. It is checked against\n"
            "  your own data shuffled: anything that survives shuffling is not\n"
            "  reported as structure.\n\n"
            "  It states its own coverage — how many channels and what share of\n"
            "  your rows were actually screened — and it says when the reading\n"
            "  scale had to be refit to your data. Read those lines; they are the\n"
            "  honest limits of the read.\n\n"
            "  If it finds nothing, it says so. A null is a result here, not a\n"
            "  failed sale, and we would rather tell you that than dress it up.\n\n"
            f"Reply to this email with any question and it reaches Jim: {REPLY_ADDRESS}\n\n"
            "Jim Jardine\nLattice24\n")
        why = "report"
    else:
        subject = f"Lattice24 — your file ran, and Jim is reading it himself (job {job})"
        body = (
            f"Your file ran clean. Job {job}.\n\n"
            "What we can tell you for certain about the file itself:\n"
            + _facts_block(f) +
            "\nWhat we are NOT sending you\n"
            "  The automated write-up. We are holding it, on purpose.\n"
            "  An audit of our own report on 26 August found that one of its\n"
            "  headline statistics is reported from the wrong tail — it can\n"
            "  print 'no sequence structure' for data that is full of it. Three\n"
            "  further sentences in it overstate what was actually screened.\n"
            "  Until that is fixed, sending it to you would mean sending you a\n"
            "  conclusion we already know is wrong.\n\n"
            "  Nothing about your file caused this. It ran normally.\n\n"
            "What happens instead\n"
            "  Jim reads your file himself and replies to you personally. If\n"
            "  there is nothing in it, he will tell you that plainly — a null\n"
            "  is a result here, not a failed sale.\n\n"
            f"Reply to this email and it reaches him: {REPLY_ADDRESS}\n\n"
            "Jim Jardine\nLattice24\n")
        why = "holding (report withheld)"

    try:
        did = queue_via_gate(
            to, subject, body,
            f"shop outcome for job {job} ({why}) — transactional reply to a "
            "customer who submitted a file, not outreach")
        log(event="outcome_queued", job=job, to=to, draft=did,
            kind=why, subject=subject)
        return did
    except Exception as exc:  # noqa: BLE001
        raise_alarm(job, f"customer {why} message could not be queued",
                    f"{exc}\n{traceback.format_exc()}")
        return None


def accept_alert(job, meta, client_ip="unknown"):
    """The savings-split button. The hottest event this server can produce.

    Sends Jim the lead and the customer a confirmation, so the page's own
    "Received. Watch your inbox." stops being a lie.
    """
    who = meta.get("email") or "(no address given)"
    ok_jim = False
    try:
        _smtp_send(
            JIM_ADDRESS,
            f"[SHOP] SAVINGS-SPLIT ACCEPTED by {who} — job {job}",
            ("Someone clicked the savings-split button on their report.\n\n"
             f"  from   : {who}\n"
             f"  ip     : {client_ip}\n"
             f"  job    : {job}\n"
             f"  market : {meta.get('market', '')}\n"
             f"  note   : {meta.get('notes_client') or '(none)'}\n\n"
             f"Report: {SITE}/report/{job}\n\n"
             "This is a request for paperwork. Answer it yourself.\n"),
            {"X-Lattice24-Shop": "accept"})
        log(event="accept_alert_sent", job=job, to=JIM_ADDRESS, who=who)
        ok_jim = True
    except Exception as exc:  # noqa: BLE001
        raise_alarm(job, "savings-split acceptance did not reach Jim",
                    f"{exc}\n{traceback.format_exc()}")

    to = (meta.get("email") or "").strip()
    if to:
        try:
            queue_via_gate(
                to, f"Lattice24 — we have your request (job {job})",
                (f"We have your request on job {job}.\n\n"
                 "Jim writes the savings-split paperwork himself; there is no\n"
                 "sales team here. He will reply to this address.\n\n"
                 "To be plain about what is being agreed: we deploy at our cost,\n"
                 "and we are paid only out of savings that are verified after\n"
                 "the fact. If nothing is recoverable in your data, nothing is\n"
                 "owed and we will say so.\n\n"
                 f"Reply to this email and it reaches him: {REPLY_ADDRESS}\n\n"
                 "Jim Jardine\nLattice24\n"),
                f"shop savings-split acknowledgement for job {job} — "
                "transactional reply to a customer request, not outreach")
        except Exception as exc:  # noqa: BLE001
            raise_alarm(job, "savings-split acknowledgement could not be queued",
                        f"{exc}\n{traceback.format_exc()}")
    return ok_jim


# ------------------------------------------------------- collect-only intake
#
# Added 2026-08-26. The hosted free instance cannot run the engine inside
# Render's 180 s request limit -- a 600-row file measured 181 s and returned
# "engine timeout after 180s". Rather than advertise a row ceiling nobody has
# measured, the site can simply COLLECT the file and say so honestly.
#
# The file must LEAVE the container immediately. Render's free tier wipes the
# filesystem on redeploy and on every spin-down, so a job left on disk is gone
# before anyone reads it. Mailing it out is the storage.

def mail_upload_to_jim(job, csv_text, meta=None):
    """Send one customer upload to Jim as an attachment. Returns True if sent.

    Raises nothing: the caller must still answer the customer even if mail
    fails, and a silent failure here is exactly the defect this replaces --
    so it prints loudly and returns False instead.
    """
    meta = meta or {}

    # A per-day ceiling on this path too. It sends direct SMTP and never
    # touches mail_gate, so the gate's SELF_SEND_CAP cannot see it.
    #
    # On 2026-08-26 a soak run pushed 381 files through this server and each
    # one mailed an alert. 549 messages left the account that day and Gmail
    # refused everything past 500, which then blocked real customer uploads.
    # A test loop must not be able to spend the account's daily quota again.
    ok, n, cap = _self_send_budget("collect", "SHOP_COLLECT_DAILY_CAP", 100)
    if not ok:
        print(f"COLLECT: daily cap of {cap} reached — upload {job} NOT mailed. "
              f"The file is on disk and its report page still serves.",
              flush=True)
        return False

    try:
        user, password = _credentials()
    except Exception as exc:                     # noqa: BLE001
        print(f"COLLECT: no credentials, upload {job} NOT mailed: {exc}",
              flush=True)
        return False

    rows = max(0, len([ln for ln in csv_text.splitlines() if ln.strip()]) - 1)
    first = (csv_text.splitlines() or [""])[0][:400]
    who = meta.get("email") or "(no address given)"
    note = (meta.get("notes") or "").strip()

    body = (
        f"Upload {job}\n"
        f"{rows:,} rows\n"
        f"from: {who}\n"
        f"branch the gate assigned: {meta.get('branch', '(none)')}\n"
        f"\nfirst line:\n{first}\n"
    )
    if note:
        body += f"\ntheir notes:\n{note}\n"
    body += ("\nThe CSV is attached. It is NOT stored on the server -- the free "
             "instance wipes its disk on every restart, so this mail is the "
             "only copy.\n")

    em = EmailMessage()
    em["From"] = f"{SENDER_NAME} <{user}>"
    em["To"] = JIM_ADDRESS
    em["Subject"] = f"[UPLOAD] {rows:,} rows from {who}"
    em["Reply-To"] = REPLY_ADDRESS
    em.set_content(body)
    em.add_attachment(csv_text.encode("utf-8", "replace"),
                      maintype="text", subtype="csv",
                      filename=f"upload_{job}.csv")
    try:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=60) as smtp:
            smtp.login(user, password)
            smtp.send_message(em)
        print(f"COLLECT: upload {job} mailed to {JIM_ADDRESS} "
              f"({rows:,} rows, {n}/{cap} today)", flush=True)
        return True
    except Exception as exc:                     # noqa: BLE001
        print(f"COLLECT: SMTP FAILED for upload {job}: {exc}", flush=True)
        return False
