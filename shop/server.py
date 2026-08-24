#!/usr/bin/env python3
"""
Self-serve shop server: customers upload a CSV, the engine runs, report comes
back. Nobody talks to Jim. This is the missing piece between shop/analyze.py
(manual CLI) and a product that runs while he sleeps.

Endpoints:
  GET  /shop/upload.html        upload page (static)
  POST /api/run                 body: JSON {csv, email?, label_col?, group_col?}
                                 -> {job}
  GET  /report/{job}            teaser (first TEASER_LINES lines of report)
  GET  /report/{job}/{token}    full report once unlocked
  GET  /api/checkout/{job}      -> redirect to Stripe Checkout (needs STRIPE_SECRET_KEY, PRICE_ID)
  POST /api/stripe/webhook      verifies signature, unlocks job

Honest limits (v1): one worker, jobs under ~/research_platform/shop/jobs/,
5 MB cap, 180 s engine timeout. The engine refuses bad data on its own --
analyze.py's gates are the quality control, this server only moves bytes.
"""
import hashlib, hmac, json, os, subprocess, time, uuid
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
TEASER_LINES = 45
STRIPE_SK   = os.environ.get("STRIPE_SECRET_KEY", "")
PRICE_ID    = os.environ.get("STRIPE_PRICE_ID", "")
WEBHOOK_SEC = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
SITE        = os.environ.get("SHOP_SITE", "https://lattice24.com")
SHARE_PCT   = os.environ.get("SHARE_PCT", "25")


def job_dir(job): return JOBS / job

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
            import annex_arms
            with open(outp, "a") as fh:
                fh.write(annex_arms.annex(str(csvp)))
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

def md_page(title, body):
    return ("<!doctype html><meta charset=utf-8>"
            "<title>Lattice24 report</title>"
            "<style>body{background:#F5F4F1;color:#16191B;font-family:Georgia,serif;"
            "max-width:72ch;margin:3rem auto;padding:0 1rem;line-height:1.6}"
            "pre{white-space:pre-wrap;background:#fff;border:1px solid #DBD9D3;"
            "border-left:3px solid #A9601A;padding:1.2rem;font-size:.86em;overflow-x:auto}"
            "a.btn{display:inline-block;background:#A9601A;color:#fff;padding:.7em 1.4em;"
            "text-decoration:none;border-radius:4px;font-family:sans-serif}</style>"
            f"<h1>{title}</h1>{body}")

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/shop/upload.html", "/shop/compute.html"):
            return self._send(200, (STATIC / Path(p).name).read_text())
        parts = p.strip("/").split("/")
        if p.startswith("/report/") and len(parts) == 2:
            try: m = load(parts[1])
            except Exception: return self._send(404, md_page("Gone", "<p>No such job.</p>"))
            if not m.get("done"):
                return self._send(200, md_page("Running&hellip;",
                    "<p>Your data is in the queue. Reload this page in a minute.</p>"))
            if not m["ok"]:
                return self._send(200, md_page("Could not read it",
                    f"<pre>{m.get('err','')}</pre><p>Check row ordering: the engine "
                    "reads sequence, rows must be time-ordered.</p>"))
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
                             "% of first-year verified cloud savings</b>; you keep " +
                             str(100-int(SHARE_PCT)) + "% - and every gated MW is capacity you get back.</p>")
                else:
                    pitch = ("<h2>The offer</h2><p>This report is free. If it finds recoverable loss "
                             "(leaked product, wasted compute, avoidable emissions), we fix or monitor it "
                             "<b>at our cost</b>, and we keep <b>" + SHARE_PCT + "% of first-year verified savings</b>. "
                             "You keep " + str(100-int(SHARE_PCT)) + "% of money that currently leaks into the air.</p>")
                cta = (pitch +
                       "<form method=\"post\" action=\"/api/accept/{j}\">"
                       "<input type=hidden name=job value={j}>"
                       "<button class=btn type=submit>I want the savings split &mdash; contact me</button></form>").replace("{j}", parts[1])
            return self._send(200, md_page("Your report",
                f"<pre>{m['report']}</pre>{cta}"))
        if p.startswith("/api/accept/") and len(parts) == 3:
            ln = int(self.headers.get("Content-Length", 0))
            self.rfile.read(ln)
            try:
                m = load(parts[2])
                import time as _t
                (job_dir(parts[2]) / "accepted.json").write_text(json.dumps(
                    {"when": _t.time(), "email": m.get("email")}))
            except Exception:
                pass
            return self._send(200, md_page("Done", "<p>Received. Watch your inbox.</p>"))
        if p.startswith("/report/") and len(parts) == 3:
            job, tok = parts[1], parts[2]
            m = load(job)
            if hmac.compare_digest(tok, m.get("token", "")):
                return self._send(200, md_page("Full report", f"<pre>{m['report']}</pre>"))
            return self._send(403, md_page("Locked", "<p>Bad link.</p>"))
        if p.startswith("/api/checkout/") and STRIPE_SK and PRICE_ID:
            job = parts[2]
            m = load(job)
            if m.get("paid"): return self._send(200, md_page("Already yours", "<p>Paid.</p>"))
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
                return self._send(500, md_page("Payment error", f"<pre>{e}</pre>"))
        return self._send(404, md_page("404", ""))

    def do_POST(self):
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
        ln = min(int(self.headers.get("Content-Length", 0)), MAX_BYTES + 65536)
        try:
            req = json.loads(self.rfile.read(ln))
        except Exception:
            return self._send(400, '{"err":"bad json"}')
        csv = req.get("csv", "")
        if not (10 < len(csv) <= MAX_BYTES):
            return self._send(400, '{"err":"csv missing or too large"}')
        job = uuid.uuid4().hex[:16]
        meta = {"csv": csv, "email": (req.get("email") or "")[:120],
                "label_col": (req.get("label_col") or "")[:64],
                "group_col": (req.get("group_col") or "")[:64], "paid": False,
                "market": "compute" if req.get("market") == "compute" else "methane"}
        (job_dir(job)).mkdir(parents=True, exist_ok=True)
        run_engine(job, meta)
        url = f"{SITE}/report/{job}"
        return self._send(200, json.dumps({"job": job, "url": url}))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8790"))
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print("shop server on :8790")
    srv.serve_forever()
