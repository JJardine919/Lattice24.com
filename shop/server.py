#!/usr/bin/env python3
"""
Self-serve shop server: customers upload a CSV, the engine runs, report comes
back. Nobody talks to Jim. This is the missing piece between shop/analyze.py
(manual CLI) and a product that runs while he sleeps.

Endpoints:
  GET  /shop/upload.html        upload page (static)
  POST /api/run                 body: JSON {csv, email?, notes_client?, notes_internal?, label_col?, group_col?}
                                 -> {job, url, pdf}
  GET  /report/{job}            teaser (first TEASER_LINES lines of report)
  GET  /report/{job}/{token}    full report once unlocked
  GET  /api/checkout/{job}      -> redirect to Stripe Checkout (needs STRIPE_SECRET_KEY, PRICE_ID)
  POST /api/stripe/webhook      verifies signature, unlocks job

Honest limits (v1): one worker, jobs under ~/lattice24_machine/jobs/,
5 MB cap, 180 s engine timeout. The engine refuses bad data on its own --
analyze.py's gates are the quality control, this server only moves bytes.

2026-08-25 additions:
  - notes_client / notes_internal fields on upload and /api/run
  - Verra MIN disclaimer on report PDF (not approved, determination 30 Sep 2026)
  - shop_submissions.json log: file hash + submit time + client IP
  - report.pdf rendered via Chrome headless, no new libs, no title rewrites
"""

import hashlib, hmac, json, os, subprocess, threading, time, uuid
import notify
import triage_gate
import urllib.request, urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT   = Path(__file__).resolve().parent
_envf  = ROOT / ".env"
if _envf.exists():
    for line in _envf.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
JOBS   = ROOT / "jobs"
STATIC = ROOT / "public"
JOBS.mkdir(exist_ok=True); STATIC.mkdir(exist_ok=True)
MAX_BYTES   = 5_000_000
ENGINE_SECS = 180
#: Row ceiling, refused at intake instead of discovered at 180 seconds.
#:
#: Measured 2026-08-26 on the deployed free-tier host: a 2,000 x 16 file
#: completes, and both 4,000 x 16 and a 17,518-row file come back as
#: "engine timeout after 180 s". Measured the same day on this laptop through
#: the full server path (engine + annex + PDF): 600 rows 16.1 s, 2,000 rows
#: 21.3 s, 4,000 rows 31.2 s -- about 11 s fixed plus 5 ms a row. Those two
#: measurements bracket the hosted machine at 6-8x slower than this one, which
#: puts its real ceiling between roughly 2,000 and 3,900 rows.
#:
#: 2,500 sits inside that bracket at the conservative end, and is the number
#: public/upload.html already states. The point of checking it HERE is that a
#: customer who sends 17,518 rows currently waits three minutes to be told it
#: failed, which reads as a broken product; this tells him in one second, with
#: the limit and his own row count in the message, so the next thing he does is
#: send a slice rather than give up.
# MEASURED against the live host, 2026-08-26. Not scaled from a laptop timing:
# that is how the 2,500 that was here before got here, and it was wrong.
#
# Binary search against https://lattice24-com.onrender.com, warm instance, the
# 16-channel refinery file truncated, total request time and whether the report
# was actually served:
#
#      600 rows  111.8 s  report served
#      600 rows  109.7 s  report served
#      900 rows  135.3 s  report served
#    1,200 rows  164.6 s  report served
#    2,400 rows  181.9 s  ENGINE TIMEOUT AFTER 180 s
#    4,000 rows  181.9 s  ENGINE TIMEOUT
#    8,000 rows  184.8 s  ENGINE TIMEOUT
#   17,518 rows  252.3 / 213.0 / 199.2 s  ENGINE TIMEOUT, three for three
#
# 600 is the largest value with more than one clean repeat behind it, and its
# worst run leaves the engine at roughly a third of the 180 s it is allowed.
# 900 and 1,200 each completed ONCE; on a shared free CPU one pass is a lucky
# draw, not a ceiling, and the third 600-row repeat came back 502 (see below),
# which is exactly the variance that makes a single pass untrustworthy.
#
# render.yaml is INERT for this service: it declares SHOP_MAX_ROWS and the
# throttles and none of it reaches the container. Render reads a blueprint's
# envVars at service creation; for a service wired up in the dashboard the file
# is documentation. Proven live -- the file said 600, then 20000, and the host
# went on answering "max_rows": 2500, which was this line's own default. The
# code default is the only lever a push controls.
MAX_ROWS = int(os.environ.get("SHOP_MAX_ROWS", "600"))
#: Collect the file instead of running the engine on it.
#:
#: Measured 2026-08-26 on the live free instance: a 600-row file took 181 s and
#: returned "engine timeout after 180s". The row ceiling that would make the
#: engine safe here has never been measured, and MIN_ROWS is 200 -- so the
#: honest window may be nearly empty. Collecting the file costs nothing, has no
#: ceiling, and is what the customer is actually promised: a read back from a
#: person. The engine then runs on a machine that can finish it.
COLLECT_ONLY = os.environ.get("SHOP_COLLECT_ONLY", "1") != "0"

#: Shared secret for the pull endpoints. UNSET = the endpoints do not exist.
#:
#: They serve customers' uploaded CSVs, so the default has to be off: a typo in
#: a deploy must not publish other people's data. With no key set, /api/pull
#: 404s exactly like any unknown path and gives away nothing about itself.
PULL_KEY = os.environ.get("SHOP_PULL_KEY", "")


def _netcheck():
    """Can this container reach the outside world, and on what?

    Outbound SMTP is blocked here; whether 443 is open decides whether the
    container can ever push anything itself. The build reaching PyPI does not
    answer it -- that is the build network, not the runtime network -- so this
    asks at runtime and reports what it got.
    """
    out = {}
    for name, url in (("https_pypi", "https://pypi.org/simple/"),
                      ("https_github", "https://api.github.com/"),
                      ("https_google", "https://www.google.com/")):
        t0 = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "lattice24-netcheck"})
            with urllib.request.urlopen(req, timeout=15) as r:
                out[name] = {"ok": True, "status": r.status,
                             "secs": round(time.time() - t0, 2)}
        except Exception as exc:  # noqa: BLE001 -- the failure IS the result
            out[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:180],
                         "secs": round(time.time() - t0, 2)}
    for name, host, port in (("smtp_gmail_465", "smtp.gmail.com", 465),
                             ("smtp_gmail_587", "smtp.gmail.com", 587)):
        import socket
        t0 = time.time()
        try:
            socket.create_connection((host, port), timeout=15).close()
            out[name] = {"ok": True, "secs": round(time.time() - t0, 2)}
        except Exception as exc:  # noqa: BLE001
            out[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:180],
                         "secs": round(time.time() - t0, 2)}
    return out


def _pull_authorised(query):
    """Constant-time key check. False whenever no key is configured."""
    if not PULL_KEY:
        return False
    supplied = urllib.parse.parse_qs(query or "").get("key", [""])[0]
    return hmac.compare_digest(supplied, PULL_KEY)


def _pending_uploads():
    """Every collected upload still waiting to be fetched, oldest first."""
    out = []
    for d in sorted(JOBS.glob("*")):
        f = d / "upload.csv"
        if not f.is_dir() and f.exists() and not (d / "FETCHED").exists():
            try:
                m = json.loads((d / "meta.json").read_text())
            except Exception:  # noqa: BLE001 -- a job mid-write is not an error
                m = {}
            out.append({
                "job": d.name,
                "bytes": f.stat().st_size,
                "submitted": m.get("submit_time", ""),
                "email": m.get("email", ""),
                "client": m.get("client", ""),
                "notes_client": m.get("notes_client", ""),
                "notes_internal": m.get("notes_internal", ""),
                "branch": m.get("branch", ""),
            })
    out.sort(key=lambda r: r["submitted"])
    return out
TEASER_LINES = 45
STRIPE_SK   = os.environ.get("STRIPE_SECRET_KEY", "")
PRICE_ID    = os.environ.get("STRIPE_PRICE_ID", "")
WEBHOOK_SEC = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
SITE        = os.environ.get("SHOP_SITE",
    "https://lattice24-com.onrender.com")
# NOT https://lattice24.com. The apex is GitHub Pages; /report/<job> there is a
# hard 404, so every job link this machine has ever emailed was dead. Browser
# uploads only survived because upload.html strips the prefix client-side, which
# does nothing for a link that is copied, forwarded or read out of an email.
# This is the base URL of the deployment SERVING the job -- override SHOP_SITE
# per deployment; a job written by one server does not exist on another.
SHARE_PCT   = os.environ.get("SHARE_PCT", "25")

# --- Where this process listens ----------------------------------------
# It bound 0.0.0.0 unconditionally, so every run on Jim's laptop published an
# unauthenticated endpoint that runs the engine and sends mail to the whole
# LAN. Default to loopback; the container that genuinely needs a public bind
# says so explicitly (Dockerfile sets HOST=0.0.0.0, because Render routes to
# the container from outside it and a loopback bind there is unreachable).
HOST = os.environ.get("HOST", "127.0.0.1")

# Behind Render, self.client_address is the platform's proxy, identical for
# every visitor on earth. A per-IP throttle keyed on it is not a per-IP
# throttle -- it is one global bucket, and the first three strangers of the
# hour would spend the whole world's quota. Read the forwarded client only
# where a proxy is actually in front, because X-Forwarded-For is a request
# header and anyone talking to this socket directly can write it themselves.
TRUST_PROXY = os.environ.get("TRUST_PROXY", "").strip().lower() in ("1", "true", "yes")

# Verra disclaimer — fixed text, verified against disk
VERRA_DISCLAIMER = "Verra MINs submitted 28-31 July 2026, NOT approved, determination 30 Sep 2026"


def job_dir(job): return JOBS / job


#: Wall-clock the order-invariance annex is allowed, in seconds.
#:
#: The annex used to run INSIDE run_engine as a plain function call, after the
#: subprocess timeout had already been satisfied -- so it was the one part of a
#: job under no time limit at all. Measured 2026-08-26 on the laptop it costs
#: 6.3 s and does NOT vary with row count (200/400/600/1200 rows: 8.33/8.38/
#: 8.33/8.51 s before the surrogate work, 6.34 s after). It is 16 channels x 3
#: fixed windows, so it is pure fixed cost. On the live free instance, which
#: measures ~24x slower than this laptop, that is ~150 s on top of the engine --
#: and it was the larger half of an observed 235 s request.
#:
#: NOTHING IS CUT FROM THE ANNEX ITSELF. Its surrogate counts are controls and
#: they are untouched. What changes is that it now runs as a subprocess with a
#: budget: if the machine cannot afford it inside that budget it is not run, and
#: the report SAYS it was not run. A section silently missing would be worse
#: than a slow one.
# Default 0 -- OFF -- because the only deployment running this code cannot
# afford it. Measured on the live host with a 45 s budget: it timed out on every
# single job, so every customer waited an extra 45 seconds to be handed a
# sentence saying it had not run. Off, they get the same sentence immediately.
# A machine that can afford it sets SHOP_ANNEX_SECS to a real budget; Jim's own
# CLI runs of analyze.py never appended the annex in the first place, so nothing
# he does by hand changes.
ANNEX_SECS = int(os.environ.get("SHOP_ANNEX_SECS", "0"))


def _append_annex(csvp, outp, meta):
    """Run the annex under a wall-clock budget; say so if it did not fit."""
    if ANNEX_SECS <= 0:
        with open(outp, "a") as fh:
            fh.write("\n## Order-invariance annex (collapse engine)\n\n"
                     "Not run: this deployment has the annex switched off "
                     "(SHOP_ANNEX_SECS=0). Ask and it will be run on the full "
                     "record off-line.\n")
        meta["annex"] = "off"
        return
    started = time.time()
    try:
        r = subprocess.run(
            ["python3", "-c",
             "import sys;sys.path.insert(0,sys.argv[1]);import annex_arms;"
             "sys.stdout.write(annex_arms.annex(sys.argv[2]))",
             str(ROOT), str(csvp)],
            capture_output=True, text=True, timeout=ANNEX_SECS)
        text, took = r.stdout, time.time() - started
        if not text.strip():
            raise RuntimeError((r.stderr or "no output")[-300:])
        meta["annex"] = f"ran in {took:.1f}s"
    except subprocess.TimeoutExpired:
        text = ("\n## Order-invariance annex (collapse engine)\n\n"
                f"**Not run — it did not fit in the {ANNEX_SECS}s this deployment "
                "allows it.** This section is an extra screen on top of the report "
                "above; nothing in the report depends on it, and none of its "
                "controls were shortened to make it fit. It is fixed-cost work "
                "(16 channels x 3 windows) and this instance is small. Ask and it "
                "will be run on your full record off-line.\n")
        meta["annex"] = f"timeout after {ANNEX_SECS}s"
    except Exception as exc:  # noqa: BLE001
        text = ("\n## Order-invariance annex (collapse engine)\n\n"
                f"**Not run — it failed.** Verbatim: `{exc}`. Nothing in the "
                "report above depends on this section.\n")
        meta["annex"] = f"failed: {exc}"
    with open(outp, "a") as fh:
        fh.write(text)


def run_engine(job, meta):
    d = job_dir(job)
    csvp, outp = d / "input.csv", d / "report.md"
    csvp.write_text(meta["csv"], encoding="utf-8")
    cmd = ["python3", str(ROOT / "analyze.py"), str(csvp),
           "--out", str(outp)]
    if meta.get("label_col"): cmd += ["--label-col", meta["label_col"]]
    if meta.get("group_col"): cmd += ["--group-col", meta["group_col"]]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=ENGINE_SECS)
        ok = outp.exists() and outp.stat().st_size > 0
        meta["done"], meta["ok"] = True, bool(ok)
        meta["err"] = "" if ok else (r.stderr or r.stdout)[-800:]
        if ok:
            _append_annex(csvp, outp, meta)
    except subprocess.TimeoutExpired:
        meta.update(done=True, ok=False, err=f"engine timeout after {ENGINE_SECS}s")
    (d / "meta.json").write_text(json.dumps({**meta, "csv": None}))
    return meta


def load(job):
    d = job_dir(job)
    m = json.loads((d / "meta.json").read_text())
    m["report"] = ((d / "report.md").read_text(errors="replace")
                   if (d / "report.md").exists() else "")
    return m


def html_escape(s):
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# PDF render template — no title rewrites, fixed labels
SCRIPT = """<!doctype html>
<html lang="en"><head><meta charset=utf-8>
<title>Lattice24 report</title>
<style>
  @page {{ size: A4; margin: 1.6cm; }}
  body {{ font-family: Georgia, serif; font-size: 10.5pt; line-height: 1.55;
         color: #16191B; background: #fff; max-width: 100%; }}
  pre {{ white-space: pre-wrap; background: #fff; border: 1px solid #DBD9D3;
        border-left: 3px solid #A9601A; padding: 1rem; font-size: 9pt;
        overflow: hidden; }}
  h1 {{ font-size: 15pt; margin: 0 0 .4rem; }}
  .note {{ font-size: 8.5pt; color: #6B7E96; font-family: ui-monospace, monospace;
          margin: .4rem 0; padding: .3rem .5rem; background: #f5f4f1;
          border-left: 2px solid #A9601A; }}
  .disclaimer {{ font-size: 8pt; color: #A33C33; font-family: ui-monospace, monospace;
                margin-top: .6rem; padding: .4rem .5rem;
                background: #fff5f5; border: 1px solid #A33C33; }}
  .meta {{ font-size: 8pt; color: #4C5559; font-family: ui-monospace, monospace;
          margin-top: 1rem; border-top: 1px solid #DBD9D3; padding-top: .5rem; }}
  a {{ color: #A9601A; }}
</style></head><body>
<h1>{label}</h1>
<pre>{content}</pre>
<div class="disclaimer">{disclaimer}</div>
<div class="note">Notes (client-visible): {notes_client}</div>
<div class="note">Internal note: {notes_internal}</div>
<div class="meta">Job: {job} &middot; Generated: {generated} &middot;
Lattice24 &middot; James Jardine &middot; ORCID 0009-0004-9073-7192</div>
</body></html>"""


def render_html(job, meta):
    """Render report HTML for PDF generation."""
    return SCRIPT.format(
        label="Lattice24 Report",
        # Escaped like every other route. Chrome headless runs JS while
        # rendering to PDF, so an unescaped report body executes on this
        # machine -- and the body now legitimately carries customer-typed
        # text (column names, --price-source). Missed in the first pass.
        content=html_escape(meta.get("report", "")),
        disclaimer=VERRA_DISCLAIMER,
        notes_client=html_escape(meta.get("notes_client", "") or "None provided"),
        notes_internal=html_escape(meta.get("notes_internal", "") or "None provided"),
        job=job,
        generated=time.strftime("%Y-%m-%d %H:%M"))


def render_pdf(job, meta):
    """Render report to PDF via Chrome headless on a known path. No new libs."""
    d = job_dir(job)
    pdfp = d / "report.pdf"
    if not meta.get("ok") or not meta.get("report"):
        return str(pdfp)
    html = render_html(job, meta)
    (d / "render.html").write_text(html, encoding="utf-8")
    try:
        subprocess.run(
            ["google-chrome", "--headless", "--no-pdf-header-footer",
             "--print-to-pdf=" + str(pdfp),
             "--virtual-time-budget=2000",
             "file://" + str(d / "render.html")],
            capture_output=True, timeout=60)
    except Exception:
        pass
    return str(pdfp)


#: Shop submission log. NOT outreach/sent.json -- that file is send_drip.py's
#: do-not-re-mail state (STATE, send_drip.py:16). Until 2026-08-26 this function
#: read-modify-wrote it from an unauthenticated public POST endpoint, with no
#: lock on either side; a lost update in the shop-writes-last direction would
#: erase whatever send_drip had just recorded and those contacts would be
#: re-mailed on the next pass. Re-mailing is under a standing never-re-send
#: order, and it is the one defect here that escapes the machine and cannot be
#: undone. Keep these two files separate. Do not merge them back.
SUBMISSION_LOG = ROOT / "shop_submissions.json"

#: Throttle on the one endpoint that makes this machine send mail.
#:
#: /api/run is unauthenticated and, since notify.py was wired in on 2026-08-26,
#: an unauthenticated POST names who Jim's Gmail account writes to. That draft
#: used to carry "authorized", which cleared mail_gate's never-re-mail hold and
#: its daily cap; as of 2026-08-26 it carries "public_origin" instead and the
#: gate holds every one of them for an explicit human SEND, so no mail reaches
#: an outside address on a stranger's say-so. This throttle is no longer the
#: only thing in front of that -- but it is still what stops the queue filling
#: with drafts Jim has to read, and the account it protects is the same
#: credential mail_gate, watch_replies and send_drip all authenticate as.
#:
#: State is in memory on purpose: it must never persist. A daily-cap hold once
#: got written to mail_gate's state.json and froze 19 drafts for four days --
#: a throttle that outlives its window is a lockout, not a throttle.
RATE_PER_IP_HOUR = int(os.environ.get("SHOP_RATE_PER_IP_HOUR", "3"))
RATE_PER_ADDR_DAY = int(os.environ.get("SHOP_RATE_PER_ADDR_DAY", "2"))
_rate_lock = threading.Lock()
_rate_hits = {"ip": {}, "addr": {}}


#: Throttle on the endpoint ITSELF, not just on the mail it sends.
#:
#: rate_ok() below withholds the MAIL and lets the job run anyway -- correct
#: for a customer who uploads twice, useless as protection for the machine.
#: /api/run is unauthenticated and each accepted POST occupies a worker for up
#: to ENGINE_SECS (180 s) of CPU. ThreadingHTTPServer starts a thread per
#: request and never refuses one, so an unthrottled endpoint is a free
#: engine-run generator for anyone who finds the URL -- and at 100 outreach
#: emails a day, the URL becomes public by design.
#:
#: This cap is deliberately looser than the mail cap (a real customer retrying
#: a failed upload must not meet a 429 before he meets the mail throttle) and
#: is expressed in requests, not mails, so a caller who leaves the email field
#: blank is still counted.
# Back to 6, where it was. It was raised to 40 for roughly 35 minutes on
# 2026-08-26 (commit fcf97f3, 03:00-03:35 UTC) because six accepted runs cannot
# binary-search a ceiling whose probes take three minutes each, and a 429 is not
# a data point. That window is closed.
RUN_PER_IP_HOUR = int(os.environ.get("SHOP_RUN_PER_IP_HOUR", "6"))
_run_lock = threading.Lock()
_run_hits = {}


def run_rate_ok(client_ip):
    """False -> refuse the request outright. Returns (ok, seconds_to_retry).

    Checks without recording. The hit is recorded by run_rate_commit() only once
    the request has been accepted for an engine run -- see the note there.
    """
    if not client_ip:
        return True, 0
    now = time.time()
    with _run_lock:
        # Sweep every caller, not just this one: entries are only pruned when
        # the same address comes back, so a flood from rotating addresses would
        # otherwise grow this dict without bound.
        for ident in list(_run_hits):
            kept = [t for t in _run_hits[ident] if now - t < 3600]
            if kept:
                _run_hits[ident] = kept
            else:
                del _run_hits[ident]
        hits = _run_hits.get(client_ip, [])
        if len(hits) >= RUN_PER_IP_HOUR:
            return False, int(3600 - (now - hits[0])) + 1
        return True, 0


def run_rate_commit(client_ip):
    """Record one accepted run against the caller's hourly quota.

    Deliberately NOT called for a 400. The cap exists to stop the engine from
    being run for free, and a rejected request never reaches the engine -- it
    costs a JSON parse. Counting rejections instead turns the row ceiling added
    for Fault 4 into a lockout generator: a first-time customer who picks the
    wrong file six times is refused for an hour with Retry-After: 3600, having
    never once been told anything except how to fix his file.
    """
    if not client_ip:
        return
    now = time.time()
    with _run_lock:
        _run_hits.setdefault(client_ip, []).append(now)


def rate_ok(client_ip, addr):
    """True if this upload may generate mail. ThreadingHTTPServer -> locked."""
    now = time.time()
    with _rate_lock:
        for key, ident, window, cap in (
                ("ip", client_ip, 3600, RATE_PER_IP_HOUR),
                ("addr", (addr or "").lower(), 86400, RATE_PER_ADDR_DAY)):
            if not ident:
                continue
            hits = [t for t in _rate_hits[key].get(ident, []) if now - t < window]
            if len(hits) >= cap:
                _rate_hits[key][ident] = hits
                return False, f"{key} limit"
            hits.append(now)
            _rate_hits[key][ident] = hits
        return True, ""


def _log_sent(job, meta, client_ip):
    """Append a record to the shop submission log: hash, submit time, client IP."""
    h = hashlib.sha256((meta.get("csv") or "").encode()).hexdigest()[:16]
    rec = {
        "job": job,
        "file_hash": h,
        "submit_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "client_ip": client_ip,
        "email": meta.get("email", ""),
        "notes_client": meta.get("notes_client", ""),
        "notes_internal": meta.get("notes_internal", ""),
    }
    existing = {}
    if SUBMISSION_LOG.exists():
        try:
            existing = json.loads(SUBMISSION_LOG.read_text())
        except Exception:
            pass
    existing[job] = rec
    # Atomic: write a sibling temp file and rename over the target, so a crash
    # or a concurrent request can never leave a truncated log behind.
    tmp = SUBMISSION_LOG.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(existing, indent=1))
    os.replace(tmp, SUBMISSION_LOG)


def _page_html(title, body):
    return ("<!doctype html><meta charset=utf-8>"
            "<title>" + title + " &middot; Lattice24</title>"
            "<style>body{background:#F5F4F1;color:#16191B;font-family:Georgia,serif;"
            "max-width:72ch;margin:3rem auto;padding:0 1rem;line-height:1.6}"
            "pre{white-space:pre-wrap;background:#fff;border:1px solid #DBD9D3;"
            "border-left:3px solid #A9601A;padding:1.2rem;font-size:.86em;overflow-x:auto}"
            "a.btn{display:inline-block;background:#A9601A;color:#fff;padding:.7em 1.4em;"
            "text-decoration:none;border-radius:4px;font-family:sans-serif}</style>"
            f"<h1>{title}</h1>{body}")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _client_ip(self):
        """The caller's address, seen through whatever is in front of us."""
        if TRUST_PROXY:
            xff = self.headers.get("X-Forwarded-For", "")
            if xff:
                return xff.split(",")[0].strip()
        return self.client_address[0] if self.client_address else ""

    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        p = self.path.split("?")[0]
        q = self.path.split("?", 1)[1] if "?" in self.path else ""

        # --- pull endpoints -------------------------------------------------
        # Render's free tier blocks outbound SMTP (measured 2026-08-27:
        # [Errno 101] Network is unreachable, with valid credentials), so this
        # container cannot push a customer's file anywhere. These let Jim's
        # machine PULL instead, which needs nothing outbound from here.
        if p == "/api/netcheck":
            if not _pull_authorised(q):
                return self._send(404, _page_html("404", ""))
            return self._send(200, json.dumps(_netcheck(), indent=2),
                              "application/json")
        if p == "/api/pull":
            if not _pull_authorised(q):
                return self._send(404, _page_html("404", ""))
            return self._send(200, json.dumps(_pending_uploads(), indent=2),
                              "application/json")
        if p.startswith("/api/pull/"):
            if not _pull_authorised(q):
                return self._send(404, _page_html("404", ""))
            parts = p.strip("/").split("/")
            job = parts[2] if len(parts) > 2 else ""
            if not job.isalnum():          # job ids are uuid4().hex[:16]
                return self._send(400, '{"err":"bad job id"}', "application/json")
            f = job_dir(job) / "upload.csv"
            if not f.exists():
                return self._send(404, '{"err":"no such upload"}', "application/json")
            if len(parts) == 4 and parts[3] == "ack":
                # Marked only once the fetcher says it has the bytes, so a
                # failed download is retried rather than silently dropped.
                (job_dir(job) / "FETCHED").write_text(
                    time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
                return self._send(200, '{"ok":true}', "application/json")
            return self._send(200, f.read_bytes(), "text/csv")

        if p.startswith("/shop/") and p.endswith(".html"):
            name = Path(p).name
            f = STATIC / name
            if f.exists() and "/" not in name and ".." not in name:
                return self._send(200, f.read_text())
            return self._send(404, _page_html("404", ""))
        parts = p.strip("/").split("/")
        if p.startswith("/report/") and len(parts) == 2:
            try: m = load(parts[1])
            except Exception: return self._send(404, _page_html("Gone", "<p>No such job.</p>"))
            if not m.get("done"):
                return self._send(200, _page_html("Running&hellip;",
                    "<p>Your data is in the queue. Reload this page in a minute.</p>"))
            # A triaged job carries its own answer. The generic page below used
            # to serve "Check row ordering: the engine reads sequence, rows must
            # be time-ordered" to EVERY unsuccessful job -- including a PDF,
            # where row ordering is not the problem, is not the fix, and sends
            # the customer to re-sort a file that has no rows. A CANNOT-READ is
            # an answer plus a shopping list; a canned line that fits one case
            # out of six is neither.
            if m.get("branch") in ("REVIEW", "CANNOT") and m.get("triage_message"):
                title = ("A person is looking at this" if m["branch"] == "REVIEW"
                         else "We could not read this file")
                return self._send(200, _page_html(title,
                    f"<pre>{html_escape(m['triage_message'])}</pre>"
                    f"<p>Reply to <a href=\"mailto:{notify.REPLY_ADDRESS}\">"
                    f"{notify.REPLY_ADDRESS}</a> and it reaches Jim directly.</p>"))
            # Collect mode: the engine deliberately did not run, so ok=False
            # is not a failure and none of the engine-failure pages below are
            # the right answer. Serve the branch message the collect path
            # already wrote -- "we have your file", or, if the mail out failed,
            # a plain statement that it did not arrive and where to send it.
            if m.get("branch") in ("COLLECTED", "COLLECT_FAILED"):
                got = m["branch"] == "COLLECTED"
                body = m.get("triage_message", "")
                # Strip the leading markdown heading; the page has a title.
                lines = [ln for ln in body.splitlines() if not ln.startswith("# ")]
                para = "".join(f"<p>{html_escape(ln.strip())}</p>"
                               for ln in lines if ln.strip())
                return self._send(200, _page_html(
                    "We have your file" if got else "Your file did not reach us",
                    para + (f"<p>Reach Jim at <a href=\"mailto:{notify.REPLY_ADDRESS}\">"
                            f"{notify.REPLY_ADDRESS}</a>.</p>")))
            if not m["ok"]:
                err = m.get("err", "") or ""
                # An engine TIMEOUT is not a data problem, and the row-ordering
                # line below is the wrong answer to it. The 2026-08-26 triage
                # fix covered the REVIEW and CANNOT branches, but a timeout has
                # branch INSTANT -- the gate PASSED the file -- and ok=False, so
                # it fell straight through to a message telling the customer to
                # re-sort rows that are already in order. He would have sorted a
                # correct file and sent it back to time out again.
                if "timeout" in err.lower():
                    limit = m.get("triage", {}).get("n_rows") or "your file's"
                    return self._send(200, _page_html("This one is on us", (
                        f"<pre>{html_escape(err)}</pre>"
                        "<p><b>Your file is fine.</b> It passed every data check — the "
                        "ordering, the time column and the channel scales were all "
                        "read successfully. What failed is our side: the run did not "
                        "finish inside the time limit our hosting allows.</p>"
                        f"<p>There is nothing to fix in the file you sent. Send fewer "
                        f"rows — <b>consecutive</b> rows, not a random sample, because "
                        f"the engine reads sequence — and the same file will run. "
                        f"Yours had {limit} rows.</p>"
                        f"<p>Or reply to <a href=\"mailto:{notify.REPLY_ADDRESS}\">"
                        f"{notify.REPLY_ADDRESS}</a> and Jim will run the whole record "
                        "himself and send you the report.</p>")))
                return self._send(200, _page_html("Could not read it",
                    f"<pre>{html_escape(err)}</pre><p>Check row ordering: the engine "
                    "reads sequence, rows must be time-ordered.</p>"))
            if not notify.REPORT_TRUSTWORTHY:
                # notify.REPORT_TRUSTWORTHY gated the OUTGOING mail only, while
                # this page went on serving the whole report to anyone holding
                # the job id -- and the receipt advertises this exact URL. So
                # message 3 said "we are holding it" and the link in message 1
                # handed it over. One flag, one meaning: if the report is not
                # trustworthy, nobody reads it here either.
                return self._send(200, _page_html("Your file ran",
                    "<p>Your file ran clean. The written analysis is being held "
                    "back on purpose &mdash; an audit of our own report found "
                    "one of its headline statistics is reported from the wrong "
                    "tail, so it can say 'no sequence structure' about data that "
                    "is full of it.</p><p>Nothing about your file caused this. "
                    "Jim is reading it himself and will reply to you.</p>"
                    f"<p>Reach him at <a href=\"mailto:{notify.REPLY_ADDRESS}\">"
                    f"{notify.REPLY_ADDRESS}</a>.</p>"))
            lines = m["report"].splitlines()
            teaser = "\n".join(lines[:TEASER_LINES])
            if m.get("accepted"):
                cta = "<p><b>Agreement received.</b> We will be in touch with the savings-split paperwork.</p>"
            else:
                if m.get("market") == "compute":
                    pitch = ("<h2>The offer</h2><p>This report is free. If your signals feed a trading "
                             "or research pipeline, the redundant recompute it exposes is what we gate "
                             "before execution. First published result: <b>88% of financial-signal "
                             "compute removed</b> at unchanged decision quality (DOI "
                             "10.5281/zenodo.18763166). We deploy at our cost and keep <b>" + SHARE_PCT +
                             "% of first-year verified cloud savings</b>; you keep "
                             + str(100-int(SHARE_PCT)) + "% - and every gated MW is capacity you get back.</p>")
                else:
                    pitch = ("<h2>The offer</h2><p>This report is free. If it finds recoverable loss "
                             "(leaked product, wasted compute, avoidable emissions), we fix or monitor it "
                             "<b>at our cost</b>, and we keep <b>" + SHARE_PCT + "% of first-year verified savings</b>. "
                             "You keep " + str(100-int(SHARE_PCT)) + "% of money that currently leaks into the air.</p>")
                cta = (pitch +
                       "<form method=\"post\" action=\"/api/accept/{j}\">"
                       "<input type=hidden name=job value={j}>"
                       "<button class=btn type=submit>I want the savings split &mdash; contact me</button></form>").replace("{j}", parts[1])
            return self._send(200, _page_html("Your report",
                f"<pre>{html_escape(m['report'])}</pre>{cta}"))
        if p.startswith("/report/") and len(parts) == 3:
            job, tok = parts[1], parts[2]
            m = load(job)
            if hmac.compare_digest(tok, m.get("token", "")):
                if not notify.REPORT_TRUSTWORTHY:
                    return self._send(200, _page_html("Held back",
                        "<p>This report is being withheld while a defect in how "
                        "it states its own headline statistic is fixed. A valid "
                        "link does not change that &mdash; the report is wrong, "
                        "not private.</p>"
                        f"<p><a href=\"mailto:{notify.REPLY_ADDRESS}\">"
                        f"{notify.REPLY_ADDRESS}</a></p>"))
                return self._send(200, _page_html("Full report",
                    f"<pre>{html_escape(m['report'])}</pre>"))
            return self._send(403, _page_html("Locked", "<p>Bad link.</p>"))
        if p.startswith("/api/checkout/") and STRIPE_SK and PRICE_ID:
            job = parts[2]
            m = load(job)
            if m.get("paid"): return self._send(200, _page_html("Already yours", "<p>Paid.</p>"))
            tok = m.get("token") or uuid.uuid4().hex
            (job_dir(job) / "meta.json").write_text(json.dumps({**{k:v for k,v in m.items() if k!="report"}, "token": tok}))
            q = urllib.parse.urlencode({
                "mode": "payment", "success_url": f"{SITE}/report/{job}/{tok}",
                "cancel_url": f"{SITE}/report/{job}", "line_items[0][price]": PRICE_ID,
                "line_items[0][quantity]": "1", "client_reference_id": job})
            try:
                sess = json.loads(urllib.request.urlopen(urllib.request.Request(
                    "https://api.stripe.com/v1/checkout/sessions", q.encode(),
                    {"Authorization": f"Bearer {STRIPE_SK}",
                     "Content-Type": "application/x-www-form-urlencoded"}), timeout=30).read())
                self.send_response(302)
                self.send_header("Location", sess["url"]); self.end_headers(); return
            except Exception as e:
                return self._send(500, _page_html("Payment error",
                    f"<pre>{html_escape(str(e))}</pre>"))
        return self._send(404, _page_html("404", ""))

    def _accept(self, job):
        """Savings-split acceptance. Handled here because the report page posts
        a form to it -- it used to live in do_GET, so every real click returned
        404 and the acceptance reached nobody at all."""
        ln = int(self.headers.get("Content-Length", 0))
        if ln:
            self.rfile.read(ln)
        try:
            m = load(job)
        except Exception:
            return self._send(404, _page_html("Gone", "<p>No such job.</p>"))
        try:
            (job_dir(job) / "accepted.json").write_text(json.dumps(
                {"when": time.time(), "email": m.get("email")}))
        except Exception as exc:  # noqa: BLE001
            print("accept: could not record acceptance:", exc, flush=True)
        reached = False
        try:
            reached = notify.accept_alert(
                job, m, self.client_address[0] if self.client_address else "unknown")
        except Exception as exc:  # noqa: BLE001
            print("NOTIFY PATH FAILED on accept:", exc, flush=True)
        # Only claim the inbox promise when a message actually left.
        if reached and m.get("email"):
            body = "<p>Received. A confirmation is on its way to " \
                   f"{html_escape(m['email'])}, and Jim has your request.</p>"
        else:
            body = ("<p>Received, and recorded here. If you do not hear back "
                    f"within a day, email <a href=\"mailto:{notify.REPLY_ADDRESS}\">"
                    f"{notify.REPLY_ADDRESS}</a> directly &mdash; that address is "
                    "read by a person.</p>")
        return self._send(200, _page_html("Done", body))

    def do_POST(self):
        if self.path.startswith("/api/accept/"):
            parts = self.path.split("?")[0].strip("/").split("/")
            if len(parts) == 3:
                return self._accept(parts[2])
        if self.path != "/api/run":
            if self.path == "/api/stripe/webhook":
                ln = int(self.headers.get("Content-Length", 0))
                payload = self.rfile.read(ln)
                sig = self.headers.get("Stripe-Signature", "")
                ok = False
                if WEBHOOK_SEC:
                    t = dict(kv.split("=", 1) for kv in sig.split(","))
                    exp = hmac.new(WEBHOOK_SEC.encode(),
                                   f"{t['t']}.".encode() + payload, hashlib.sha256).hexdigest()
                    ok = hmac.compare_digest(exp, t.get("v1", ""))
                if ok:
                    ev = json.loads(payload)
                    if ev.get("type") == "checkout.session.completed":
                        ref = ev["data"]["object"].get("client_reference_id")
                        try:
                            m = load(ref)
                            (job_dir(ref) / "meta.json").write_text(json.dumps(
                                {**{k: v for k, v in m.items() if k != "report"},
                                 "paid": True, "token": m.get("token") or uuid.uuid4().hex}))
                        except Exception:
                            pass
                return self._send(200, "ok")
            return self._send(404, "")
        # Refuse before reading the body, before the engine, before notify --
        # a request that is over the cap must cost this machine nothing but the
        # 429 it writes back.
        client_ip = self._client_ip()
        ok, retry = run_rate_ok(client_ip)
        if not ok:
            b = json.dumps({"err": "rate limited", "retry_after": retry}).encode()
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.send_header("Retry-After", str(retry))
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
            print(f"429 /api/run from {client_ip} "
                  f"(> {RUN_PER_IP_HOUR}/hour), retry in {retry}s", flush=True)
            return
        # Check the declared size BEFORE reading. Reading min(len, MAX+65536)
        # truncated the body mid-JSON, so json.loads failed first and every
        # realistic oversized CSV was told it was malformed rather than large.
        # Re-authored 2026-08-26. The block that stood here was correct and
        # behaviourally verified, but no session on this machine claimed it and
        # no trace of the writer exists on disk, so it is replaced with an
        # equivalent this session wrote and tested rather than shipped as
        # unattributed code. It keeps both cases the original handled.
        #
        # Case 1, Content-Length present: reject on the DECLARED size, before
        # reading. The original bug read min(len, MAX+65536), which truncated
        # the body mid-JSON so json.loads failed first and every realistic
        # oversized CSV was told it was malformed rather than large.
        # Case 2, Content-Length absent (chunked, or a client that omits it):
        # there is no declared size to check, so read one byte past the limit
        # and reject if it is reached.
        declared = int(self.headers.get("Content-Length", 0))
        if declared > MAX_BYTES:
            return self._send(400, '{"err":"csv missing or too large"}')
        if declared:
            body_bytes = self.rfile.read(declared)
        else:
            body_bytes = self.rfile.read(MAX_BYTES + 1)
            if len(body_bytes) > MAX_BYTES:
                return self._send(400, '{"err":"csv missing or too large"}')
        try:
            req = json.loads(body_bytes)
        except Exception:
            return self._send(400, '{"err":"bad json"}')
        csv = req.get("csv", "")
        if not (10 < len(csv) <= MAX_BYTES):
            return self._send(400, '{"err":"csv missing or too large"}')
        # Count data rows, not lines: one header, and a trailing newline is not
        # a row. Refuse before the engine, and say the actual number -- "too
        # many rows" with no number is the same dead end as a timeout.
        n_rows = max(0, len([ln for ln in csv.splitlines() if ln.strip()]) - 1)
        # In collect mode the engine never runs here, so the host's 180 s limit
        # does not apply and there is no honest reason to refuse a big file.
        # The byte cap above is still the real guard.
        if n_rows > MAX_ROWS and not COLLECT_ONLY:
            # This is a CANNOT-READ, labelled as one. It is an answer plus a
            # specific ask, which is the whole contract of that branch -- it is
            # not a fourth branch where nothing happens. It is decided before
            # triage because it is a limit of the HOST, not a verdict on the
            # data: the same file on a bigger machine would read fine, and
            # saying "your data is bad" about a hosting ceiling would be a lie.
            return self._send(400, json.dumps({
                "branch": "CANNOT",
                "err": f"{n_rows:,} rows is past what this hosted instance can "
                       f"finish inside its {ENGINE_SECS}s limit. Send at most "
                       f"{MAX_ROWS:,} rows -- and send consecutive rows, not a "
                       f"random sample: the engine reads sequence.",
                "rows": n_rows, "max_rows": MAX_ROWS}))
        # One recipient, or none. smtplib fans a comma-separated To out to every
        # address in it, so a 120-character field was roughly eight strangers
        # per request -- an open relay signed "Jim Jardine / Lattice24".
        # Allowlist, not denylist. The previous check looked for comma,
        # semicolon and space; it passed a newline, a tab, an angle-bracket
        # form and a literal %0A straight through, and a filter that catches
        # one separator and not the encoded version is not a filter. Validate
        # against notify's single-address grammar instead, and do NOT strip
        # first -- surrounding whitespace is how an injected address arrives
        # looking innocent, so it is a reason to refuse, not something to tidy.
        email_in = (req.get("email") or "")[:254]
        if email_in and not notify.valid_single_address(email_in):
            return self._send(400, '{"err":"one ordinary email address, or leave it blank"}')
        job = uuid.uuid4().hex[:16]
        meta = {
            "csv": csv,
            "email": email_in,
            "notes_client": (req.get("notes_client") or "")[:500],
            "notes_internal": (req.get("notes_internal") or "")[:500],
            # Which of the sender's OWN clients this file is for. One contact
            # can have many: Verdis Group verifies a zoo, a hospital system, a
            # school district, a city, a university and an airport, and those
            # six do not share a column layout, a sampling rate or a unit.
            # Without this field six uploads are one pile.
            "client": (req.get("client") or "")[:80],
            "label_col": (req.get("label_col") or "")[:64],
            "group_col": (req.get("group_col") or "")[:64],
            "paid": False,
            "market": "compute" if req.get("market") == "compute" else "methane",
            # The review queue and the alert both print these. They were reading
            # keys that were never set, so every queue entry said submitted ""
            # with file hash "" -- Jim would have been starting cold on exactly
            # the fields meant to stop that.
            "submit_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "file_hash": hashlib.sha256(csv.encode("utf-8", "replace")).hexdigest(),
        }
        # Accepted: this one is going to the engine, so it counts.
        run_rate_commit(client_ip)
        # ---- pre-flight triage, BEFORE the engine -----------------------
        # Every upload lands in exactly one of INSTANT / REVIEW / CANNOT. There
        # is no fourth branch where nothing happens. triage_gate.py has existed
        # since 2026-08-26 16:43 and was imported by nothing, so until now every
        # upload took the INSTANT path by default -- including the 75% of
        # domains measured unreadable to this screen, who got a report built on
        # a verdict the gate would have refused.
        try:
            tri = triage_gate.triage(csv)
        except Exception as exc:  # noqa: BLE001 -- a gate that throws must not
            # become a silent INSTANT. Route it to a human instead.
            tri = {"branch": "REVIEW", "reasons": [("gate_error", "FAIL",
                   f"The pre-flight check could not run on this file ({exc}). "
                   "A person is looking at it.")],
                   "pre_flight": {}, "shape_problems": [], "saturation": [],
                   "summary": "The pre-flight check could not run; a person is looking.",
                   "n_channels": 0, "n_rows": 0}
        meta["branch"] = tri["branch"]
        meta["triage"] = {k: v for k, v in tri.items() if k != "reasons"}
        meta["triage_reasons"] = tri["reasons"]
        meta["triage_message"] = triage_gate.render_message(tri["branch"], tri, meta)

        (job_dir(job)).mkdir(parents=True, exist_ok=True)
        # Before the engine, not after: the run takes up to ENGINE_SECS and the
        # customer was told "watch your inbox". They hear back first, then wait.
        # Every notify call swallows nothing -- it logs and raises its own alarm
        # -- so the guard here is only for a failure inside the alarm itself.
        may_mail, why = rate_ok(client_ip or "unknown", meta.get("email"))
        meta["may_mail"] = may_mail
        try:
            if may_mail:
                notify.receipt(job, meta)
                notify.alert_jim(job, meta, client_ip)
            else:
                # The job still runs and the report is still theirs. Only the
                # mail is withheld, and it is withheld LOUDLY -- a silent drop
                # here would rebuild the exact defect this file was fixing.
                notify.log(event="mail_rate_limited", job=job, ip=client_ip,
                           to=meta.get("email", ""), limit=why)
                print(f"RATE LIMITED ({why}): job {job}, no mail sent", flush=True)
        except Exception as exc:  # noqa: BLE001
            print("NOTIFY PATH FAILED on intake:", exc, flush=True)
        if COLLECT_ONLY:
            # Mail it out FIRST, then answer. Render's free tier wipes the
            # filesystem on redeploy and on every spin-down, so the job on
            # disk is not storage -- the mail is. If the mail fails the
            # customer is told plainly that it failed, never that we have
            # their file when we do not.
            # Write the file to disk BEFORE trying to mail it.
            #
            # Until 2026-08-26 this block held the only copy of a customer's
            # CSV in a local variable: run_engine writes csvp, collect mode
            # bypasses run_engine, and meta.json persists "csv": None. So when
            # SMTP returned 550 the upload was not delayed, it was DESTROYED --
            # and the customer was told to re-send it to an address that was
            # itself throttled.
            #
            # Render's free tier wipes on spin-down after idle, not instantly,
            # so a job on disk has a real retry window. Disk-then-mail turns an
            # unrecoverable loss into a recoverable one.
            try:
                (job_dir(job) / "upload.csv").write_text(csv)
            except Exception as exc:  # noqa: BLE001 -- still try to mail it
                print(f"COLLECT: could not persist {job}: {exc}", flush=True)
            sent = False
            try:
                sent = notify.mail_upload_to_jim(job, csv, meta)
            except Exception as exc:  # noqa: BLE001 -- never lose the answer
                print(f"COLLECT: handler raised for {job}: {exc}", flush=True)
            meta["branch"] = "COLLECTED" if sent else "COLLECT_FAILED"
            meta["done"], meta["ok"] = True, False
            meta["err"] = ""
            if sent:
                msg = (
                    "# We have your file\n\n"
                    f"{max(0, len([l for l in csv.splitlines() if l.strip()]) - 1):,} "
                    "rows received.\n\n"
                    "A person reads it and comes back to you with what moved, "
                    "what it costs, and what it cannot support -- including "
                    "\"nothing here\" if that is the honest answer.\n\n"
                    "No account, no charge, no call unless you ask for one.\n"
                )
            else:
                msg = (
                    "# Your file did not reach us\n\n"
                    "Our mail is failing right now, so nobody has been "
                    "notified yet. We would rather say so than let you think "
                    "it arrived.\n\n"
                    "Your file IS saved and this page keeps working -- send "
                    "Jim this link and he can pick it up from here. Or email "
                    "the file to jjj101147@gmail.com.\n"
                )
            meta["triage_message"] = msg
            (job_dir(job) / "report.md").write_text(msg)
        elif meta["branch"] == "INSTANT":
            run_engine(job, meta)
        else:
            # REVIEW and CANNOT do not run the engine -- there is nothing
            # honest for it to report on data the gate has already refused.
            # The customer still gets an answer: the branch message is the
            # answer, and /report/<job> serves it.
            meta["done"], meta["ok"] = True, False
            meta["err"] = ""
            try:
                triage_gate.enqueue(job, tri, meta, csv)
                triage_gate.log_alert(job, tri, meta)
            except Exception as exc:  # noqa: BLE001 -- never lose the job
                print(f"TRIAGE QUEUE FAILED for {job}: {exc}", flush=True)
            (job_dir(job) / "report.md").write_text(meta["triage_message"])
        # Render PDF via Chrome headless on known path
        # render_pdf early-returns unless meta carries the report text, and
        # run_engine only ever wrote report.md to disk -- so no report.pdf has
        # ever been produced, while this JSON handed back its path as though one
        # had. Load the text render_pdf needs, then report only what exists.
        outp = job_dir(job) / "report.md"
        if meta.get("ok") and outp.exists():
            meta["report"] = outp.read_text(errors="replace")
        pdfp = render_pdf(job, meta)
        meta["pdf"] = pdfp if Path(pdfp).exists() else None
        # Log to sent.json: file hash + submit time + client IP
        _log_sent(job, meta, client_ip or "unknown")
        try:
            if meta.get("may_mail"):
                notify.outcome(job, meta)
        except Exception as exc:  # noqa: BLE001
            print("NOTIFY PATH FAILED on outcome:", exc, flush=True)
        (job_dir(job) / "meta.json").write_text(json.dumps({**meta, "csv": None}))
        url = f"{SITE}/report/{job}"
        return self._send(200, json.dumps({"job": job, "url": url, "pdf": pdfp}))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8790"))
    srv = ThreadingHTTPServer((HOST, port), Handler)
    # Print the port actually bound, not the literal 8790 the old banner
    # printed regardless of $PORT -- on Render, where PORT=10000, that banner
    # was simply false.
    print(f"shop server on {HOST}:{port}  trust_proxy={int(TRUST_PROXY)}", flush=True)
    srv.serve_forever()
