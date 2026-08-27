"""The dollar figure -- or, far more often, the honest refusal to print one.

THE COMMERCIAL MODEL THIS IMPLEMENTS
------------------------------------
**75/25 on the market value per tonne of CO2e.** The customer keeps 75%, Lattice24
takes 25%. That is the deal already in the outreach mail ("we remediate at our cost
and take 25% of first-year verified savings -- you keep 75%"), so the report must
show both halves and nobody should discover the split later.

The chain, and every link comes from the customer:

    their metered column -> rate per hour -> hours -> quantity
    quantity x their emission factor       -> tonnes CO2e
    tonnes x market price at purchase date -> gross value
    gross split 75 / 25                    -> their share / ours

NOT CREDITS. The Verra MINs are SUBMITTED, NOT APPROVED -- determination
30 Sep 2026. Nothing here is a VCU and nothing here is being sold as one. What
this prices is documented avoided tCO2e for the customer's own reporting, and
the report says so on its face.

FOUR RULES, NONE OPTIONAL
-------------------------
  1. TRACEABLE. Every number in the arithmetic came from the customer -- a column
     of their file or a value they typed. **No constant in this code contributes
     to the figure**, with one named exception disclosed in the assumptions table
     (WIN_FOR_SPREAD, which sets the width of the range and nothing else -- see
     rule 3). There is no default price, no default emission factor, no default
     period.
  2. THE PRICE IS NEVER HARDCODED. It is a market rate, it moves, and it must be
     priced at the time of purchase and carried with its SOURCE, its DATE and its
     MARKET. Avoidance and removal are not the same number; compliance and
     voluntary are not the same number. A price without its market is not
     quotable and this file will not print one.
  3. A RANGE, FROM THE SPREAD IN THE DATA -- and honestly labelled. Low and high
     come from the 10th and 90th percentile of the per-window means of the
     customer's own column. That is a SAMPLING INTERVAL ON THE MEAN, not an
     uncertainty on the value: its width scales as 1/sqrt(WIN_FOR_SPREAD), so on
     near-iid data it narrows toward zero and would read as precision it does not
     have. Measured 2026-08-26 (DooDoo): the same file and the same customer
     inputs gave a band of 0.94% of central at WIN_FOR_SPREAD=8 and 0.06% at
     1200, a 15x swing from a constant nobody chose deliberately. So the window
     is printed in the assumptions table beside the range, and the range is
     labelled for what it is.
  4. NO FIGURE WHEN THE DATA CANNOT SUPPORT ONE. If the tonnes cannot be derived,
     there is no dollar figure -- the report says what would be needed instead.
     That is a correct answer and a sellable one.

WHAT WAS THERE BEFORE
---------------------
Nothing in the report, and on the public page at lattice24.com/send-data.html a
JavaScript function named `estimate(file)` whose only argument was the FILE SIZE.
Its own comment read "fake $ from rows*cols heuristic". Measured: a 403,212-byte
file of pure noise returned **$17**, and Jim's real 3,023,053-byte refinery record
returned **$17** -- the same number, because both were past an 8,000-row clamp and
the function never looked inside either one. Removed. Do not reintroduce its shape.
"""
import re

import numpy as np


def _cell(v):
    """Free text safe to drop into a markdown table cell.

    Every one of these fields is typed by the customer and interpolated into a
    pipe-delimited table. A single `|` splits the row into the wrong number of
    columns and silently shifts every value after it under the wrong heading --
    on a table whose whole job is to say where each number came from. Found
    2026-08-26 while adversarially testing the priced path.
    """
    if v is None:
        return "—"
    t = re.sub(r"[\x00-\x1f\x7f]", " ", str(v))
    return t.replace("|", "&#124;").strip() or "—"


#: Spellings of "per hour" this will accept. Anything else is refused rather
#: than guessed -- see is_hourly_rate.
#:
#: Three families, and the boundaries are deliberate:
#:   1. an explicit separator: "per hour", "per hr", "per h", "/h", "/hour"
#:   2. an exponent form: "hr-1", "h^-1", "h^{-1}", "h-1" with a unicode minus
#:      or superscript. This is what a unit field looks like when it has been
#:      copied out of an engineering document.
#:   3. a COMPOUND HISTORIAN TAG where the trailing H *is* the per-hour and
#:      there is no separator at all: SCFH, MCFH, MSCFH, MMSCFH, E3M3H.
#:
#: Family 3 is a narrow whitelist on purpose. A general "ends in H" rule would
#: be the obvious implementation and it is not safe -- it would accept anything
#: whose unit happens to end in that letter. This requires the H to follow a
#: recognised VOLUME token (CF / SCF / M3), so MMSCFD and SCFM still refuse on
#: their last letter, which is the whole point.
#:
#: Boundary measured 2026-08-26 by voodoo-cf against 48 unit strings it chose
#: itself, sourced from PI / Wonderware / Cygnet / AVEVA historian tags and
#: AER/GHGRP filings rather than from my list: 28 of 28 refusals correct, ZERO
#: false accepts, six false refusals -- SCFH, scfh, MCFH, Mscf hr-1, Mscf h^-1,
#: Mscf-h^-1. Families 2 and 3 exist to take those six. My own eight-string test
#: had found none of them, and its list and mine did not overlap on a single
#: failure: both were one person's imagination.
_PER_HOUR = re.compile(
    r"""\s*(
          per\s*(hour|hr|h)                     # per hour / per hr / per h
        | /\s*(hour|hr|h)                       # /h  /hour
        | [\s.\u00b7*]*(hour|hr|h)\s*[\^]?\s*[-\u2212\u207b]\s*[1\u00b9]   # h^-1, hr-1, h-1
        | [\s.\u00b7*]*(hour|hr|h)\s*[\^]?\s*\{\s*-\s*1\s*\}                # h^{-1}
        )\s*$""",
    re.I | re.X)

#: Family 3: the trailing H is the per-hour, glued to a volume token.
_TAG_HOURLY = re.compile(r"^\s*(?:[a-z0-9]{0,3})?(?:s?cf|m3)\s*h\s*$", re.I)

#: A CONFLICTING time base anywhere in the string. This is the guard that
#: matters most in this file, and it exists because the refusal message above it
#: was manufacturing its own worst input.
#:
#: A customer types `scfm`, is correctly refused, reads "this must be an
#: explicit per-hour rate -- we will not convert it for you", and does the
#: obvious thing: appends "per hour". `scfm per hour` then contains an hour
#: marker, passed the presence test, and `quantity_unit` stripped only the
#: "per hour" -- so the emission-factor row printed "tCO2e per **scfm**", a
#: PER-MINUTE unit in the slot that must hold a volume, while the arithmetic had
#: already treated the rate as hourly. 60x wrong, with a dimensionally
#: impossible label, under the table whose whole job is to say where each number
#: came from. Ten members of the family false-accepted. Found 2026-08-26 by
#: voodoo-cf, testing against the CURRENT rule rather than re-running the list
#: that produced it.
#:
#: The presence of an hour marker was never the question. TWO time bases in one
#: unit is not an ambiguity anyone may resolve on the customer's behalf -- it is
#: a typo or a misunderstanding, and either way only they know which they meant.
#:
#: Short tokens are anchored, not searched: a naive `s`/`d`/`m` test eats Mscf,
#: MSCF, m3 and Nm3. Bare letters count only as whole atoms between separators,
#: and glued forms are matched the same way the trailing H is -- as a time letter
#: following a recognised VOLUME token, which is what makes SCFM and MSCFD
#: detectable at all.
_NONHOURLY_WORD = re.compile(
    r"^(min|mins|minute|minutes|sec|secs|second|seconds|s|m|"
    r"day|days|d|week|weeks|wk|w|month|months|mo|"
    r"year|years|yr|yrs|y|a)"
    r"(?:\s*[\^]?\s*[-\u2212\u207b]\s*[1\u00b9])?$", re.I)
#: SCFM, MSCFD, MMSCFD, e3m3d -- volume token with a NON-hourly time letter glued on.
_TAG_NONHOURLY = re.compile(r"^(?:[a-z0-9]{0,3})?(?:s?cf|m3)\s*(?:m|d|s|y|w)$", re.I)
#: Trailing qualifiers a historian tag carries that say nothing about the unit.
_QUALIFIER = re.compile(
    r"(\s*\([^)]*\)|\s*@[^/]*|\s+(avg|average|mean|calc|calculated|raw|inst|"
    r"instantaneous|total|net|gross))+\s*$", re.I)


def _atoms(unit):
    """Split a unit string into the tokens a time base could hide in."""
    return [a for a in re.split(r"[\s/,\u00b7*]+", str(unit or "").strip()) if a]


def nonhourly_token(value_unit):
    """The non-hourly time token in this unit, or None.

    Checked on the ORIGINAL string, before any qualifier is stripped, so nothing
    can be normalised away before it is seen.
    """
    for a in _atoms(value_unit):
        if _TAG_NONHOURLY.match(a):
            return a
        core = re.sub(r"^per$", "", a, flags=re.I)
        if core and _NONHOURLY_WORD.match(core):
            return a
    return None


def is_hourly_rate(value_unit):
    """True only when the unit is explicitly a PER-HOUR rate.

    The whole calculation is `quantity = rate x hours`, which is correct only if
    the rate's time base is the hour. Nothing else in the chain notices if it is
    not, and the error goes straight into the tonnes and the money:

        'scfm'          standard cubic feet per MINUTE -- wrong by 60x
        'Mscf per day'  wrong by 24x
        'Mscf'          not a rate at all; multiplying it by hours is meaningless

    Found 2026-08-26 when voodoo-cf asked for a second pair of eyes on the unit
    label, which is exactly the review that should have found it. This refuses
    instead of converting: a silent unit conversion is the same class of defect
    as a silent unit assumption, and the customer is the one who knows what
    their historian logs.
    """
    t = str(value_unit or "").strip()
    # Two time bases in one unit -- refuse before anything else looks at it.
    if nonhourly_token(t):
        return False
    t = _QUALIFIER.sub("", t).strip()
    if not (_PER_HOUR.search(t) or _TAG_HOURLY.match(t)):
        return False
    # "tonnes/h/h" is per-hour-per-hour -- an acceleration, not a flow. One
    # per-hour marker, or this is not the quantity the chain assumes.
    if len(re.findall(r"(?:per\s*|/)\s*(?:hour|hr|h)\b", t, re.I)) > 1:
        return False
    # A bare "per hour" names no quantity, so there is nothing for the emission
    # factor to be per. The factor row would read "tCO2e per " with a hole in it.
    return bool(quantity_unit(t))


def quantity_unit(value_unit):
    """'Mscf per hour' -> 'Mscf'. The unit the emission factor is really per.

    The arithmetic is quantity = rate x hours, then tonnes = quantity x factor.
    So the factor is tCO2e per MSCF, not per Mscf-per-hour -- and the report
    used to label it with the rate unit. A customer reading "tCO2e per Mscf per
    hour" would go and find a rate-based factor, which is the wrong number, and
    the error lands straight in the money. Mislabelled units are how unit-error
    losses happen; this is not cosmetic.
    """
    t = _QUALIFIER.sub("", str(value_unit or "").strip()).strip()
    if _TAG_HOURLY.match(t):
        # SCFH -> SCF, MCFH -> MCF: the trailing letter IS the per-hour.
        return t[:-1].strip()
    m = _PER_HOUR.search(t)
    # Strip the separator characters an exponent form leaves behind too, so
    # "Mscf.h-1" and "Mscf\u00b7h\u207b\u00b9" both reduce to "Mscf".
    return t[:m.start()].strip(" .\u00b7*\u2212-") if m else t

#: Window used to form the spread. NOT a free parameter hidden in the code: it is
#: printed in the assumptions table of every report that quotes a range, because
#: it sets that range's width and nothing else. See rule 3.
WIN_FOR_SPREAD = 24
#: The split. Printed in full, both halves, every time.
CUSTOMER_PCT = 75
LATTICE_PCT = 25

VERRA_STATUS = (
    "**No credits are being sold.** The Verra methodology submissions are "
    "SUBMITTED and NOT APPROVED — determination is due 30 September 2026. "
    "Nothing above is a VCU or any other issued credit. What is priced here is "
    "documented avoided tCO2e from your own data, for your own reporting."
)

REQUIRED = [
    ("--value-col", "value_col",
     "the column that meters the quantity you lose", "`flare_flow_mscfh`"),
    ("--value-unit", "value_unit",
     "what one unit of that column is, so a mean becomes a quantity per hour",
     "`Mscf per hour`"),
    ("--hours", "hours",
     "hours of process time the record covers, so a rate becomes a total",
     "`8760`"),
    ("--co2e-per-unit", "co2e_per_unit",
     "tonnes CO2e per one unit of that column — your emission factor, not ours",
     "`0.0545`"),
    ("--co2e-source", "co2e_source",
     "where that emission factor came from", "`AER Directive 060, Table 3`"),
    ("--price-per-tonne", "price_per_tonne",
     "the market price per tonne CO2e **at the time of purchase**", "`38.50`"),
    ("--price-source", "price_source",
     "where that price came from", "`Xpansiv CBL N-GEO settlement`"),
    ("--price-date", "price_date",
     "the date that price was taken — it is a market rate and it moves",
     "`2026-08-26`"),
    ("--market", "market",
     "which market the rate is from — avoidance and removal are different "
     "numbers, and so are compliance and voluntary",
     "`voluntary avoidance`"),
]


def _windowed_means(x, w=WIN_FOR_SPREAD):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = (x.size // w) * w
    if n < w:
        return np.array([])
    return x[:n].reshape(-1, w).mean(axis=1)


def _refuse(reason, extra=None):
    L = ["\n## What this is worth\n", f"**No figure is printed.** {reason}"]
    if extra:
        L += extra
    L.append("\n" + VERRA_STATUS)
    return L


def value_block(X, names, args, order_result=None, label_result=None):
    """Report lines. A figure only when every input exists and the tonnes derive.

    `label_result` carries the LABELLED path's primary verdict into the money
    block. Until 2026-08-26 nothing did, and the consequence was measured by
    voodoo-cf: two files with identical pricing flags, one whose labels were coin
    flips (primary p = 0.1997) and one with a real shape injection (p = 0.0002),
    produced money blocks that were **byte-identical once the numbers were
    masked** -- neither containing the word "separates", "null", "primary", or
    any p value at all. The same "**You keep** 75% / Lattice24 25%" table sat
    under both. Everywhere else this report refuses to let a null pass -- "a
    secondary channel does not rescue a null primary" is in it verbatim -- and
    that discipline stopped at the one section a customer reads for money.

    The standing qualifier ("a price on a quantity, not a promise of a saving")
    is true and does real work, but it is STATIC: identical in both files, so it
    cannot distinguish finding something from finding nothing. The 75/25 split is
    an offer structure, and it only means anything if there is a recoverable
    fraction to split.
    """
    missing = [r for r in REQUIRED if getattr(args, r[1], None) in (None, "")]

    if missing:
        L = ["\n## What this is worth\n",
             "**No figure is printed, because this file does not contain what is "
             "needed to price a loss.** A CSV of sensor readings says how a "
             "quantity moved. It does not say what one unit of that quantity is "
             "worth in tonnes of CO2e, what a tonne was trading at on the day, or "
             "which column is the one that costs you money. Guessing any of those "
             "would produce a number you could quote and we could not defend.\n",
             "Here is exactly what would let us price it. **Every one of these "
             "comes from you, not from us**, and the report prints each of them "
             "beside the result:\n",
             "| what is missing | why it is needed | example |",
             "|---|---|---|"]
        for flag, _attr, why, example in missing:
            L.append(f"| `{flag}` | {why} | {example} |")
        L.append(f"\nWith those, the report prints the whole chain — your rate → "
                 f"your hours → tonnes CO2e → the market price on the date you "
                 f"name → **your {CUSTOMER_PCT}% and our {LATTICE_PCT}%** — with "
                 f"every assumption named beside it. Until then it prints "
                 f"nothing, which is the only honest thing it can do.")
        L.append("\n" + VERRA_STATUS)
        return L

    col = args.value_col
    if col not in names:
        return _refuse(
            f"The column named for pricing, `{col}`, is not among the channels "
            f"that were read: {', '.join(names)}. Nothing is guessed in its place.")

    # A CLASH is a non-hourly token sitting beside an hourly one -- "scfm per
    # hour". A bare "scfm" is not a clash, it is simply the wrong time base, and
    # telling a customer their single-time-base unit "names two time bases" is
    # its own small lie. Caught immediately after writing the guard.
    _u = _QUALIFIER.sub("", str(args.value_unit or "").strip()).strip()
    clash = nonhourly_token(args.value_unit) if (
        _PER_HOUR.search(_u) or _TAG_HOURLY.match(_u)) else None
    if clash:
        return _refuse(
            f"`--value-unit` is `{_cell(args.value_unit)}`, which names **two "
            f"different time bases** — `{_cell(clash)}` and an hourly one. That "
            f"is not an ambiguity anyone can resolve on your behalf: it is "
            f"either a typo or a misunderstanding, and only you know which "
            f"number the column actually holds.",
            [f"\n**If the column really is `{_cell(clash)}`-based, relabelling it "
             f"does not change it.** Converting the *values* is the step — "
             f"multiply a per-minute rate by 60, a per-day rate by 1/24 — and "
             f"then the unit is genuinely per hour. Relabelling alone would make "
             f"this report print a number that is wrong by exactly that factor, "
             f"with your original unit still sitting in the emission-factor row.",
             "\nIf the hourly label is the correct one, remove the other time "
             "token and send it again."])

    if not is_hourly_rate(args.value_unit):
        return _refuse(
            f"`--value-unit` is `{_cell(args.value_unit)}`, and this calculation "
            f"only works on a rate expressed **per hour**. It multiplies your "
            f"rate by the hours you gave to get a quantity, so any other time "
            f"base is wrong by that base's ratio and nothing downstream would "
            f"notice — `scfm` would be out by 60×, `per day` by 24×, and a bare "
            f"quantity is not a rate at all.",
            ["\n**Convert the VALUES in the column, then label it per hour.** "
             "This matters and it is the easy thing to get wrong: appending "
             "`per hour` to a per-minute unit — sending `scfm per hour` — does "
             "not convert anything. It relabels a number that is still per "
             "minute, and the result would be wrong by 60× with your original "
             "unit still printed beside it. **This report refuses that string "
             "too**, for that reason.",
             "\nWhat is wanted: a column whose values are already per hour, "
             "labelled `Mscf per hour`, `SCFH`, `m3/h` or similar. **We will not "
             "convert it for you** — a silent unit conversion is the same defect "
             "as a silent unit assumption, and you are the one who knows what "
             "your historian logs."])

    x_raw = X[:, names.index(col)]
    x = x_raw[np.isfinite(x_raw)]
    # Rows dropped from a PRICED column are disclosed, never dropped quietly.
    # np.isfinite silently removes blanks, NaNs and infinities, and the mean
    # that gets multiplied into the money is then a mean over a set the customer
    # did not know had shrunk. Same class as one bad cell deleting a sensor.
    dropped = int(x_raw.size - x.size)
    if x.size == 0 or float(np.mean(x)) <= 0:
        m = float(np.mean(x)) if x.size else float("nan")
        return _refuse(
            f"`{col}` has a mean of {m:.4g}, which cannot be a positive rate of "
            f"loss. A negative or zero rate priced positively is how a report "
            f"invents money.")

    # POSITIVE AND FINITE, not "not non-positive". `nan <= 0` is False and
    # `inf <= 0` is False, so the old guard -- `if factor <= 0 or price <= 0 or
    # hours <= 0` -- passed nan, inf, -inf and 1e400 (which float() silently
    # overflows to inf, with no rejection the customer ever sees). Measured
    # 2026-08-26 by voodoo-cf on the real priced path: --co2e-per-unit nan
    # rendered the whole savings-split table with `nan USD` in every money cell,
    # under the correct unit note and the Verra block; --price-per-tonne inf
    # printed a PLAUSIBLE tonnage line with infinite money beside it, which is
    # the worse of the two because only half of it looks wrong.
    #
    # Non-positivity is not the same predicate as positivity, and NaN is
    # neither. One predicate closes nan, inf, -inf, 1e400 and 0 together.
    checks = [("emission factor", "--co2e-per-unit", args.co2e_per_unit),
              ("price per tonne", "--price-per-tonne", args.price_per_tonne),
              ("hours covered", "--hours", args.hours)]
    for label, flag, raw in checks:
        v = float(raw)
        if not (np.isfinite(v) and v > 0):
            shown = ("not a number" if v != v else
                     "infinite (a value this large overflows to infinity when "
                     "it is parsed, which is why you saw no error)"
                     if np.isinf(v) else f"{v:g}")
            return _refuse(
                f"The {label} you gave (`{flag}`) is **{shown}**. Every tonne "
                f"and every dollar below is a straight multiplication through "
                f"that number, so there is nothing to print — not a large "
                f"figure, not a small one, and not a table of `nan`. It must be "
                f"a finite number greater than zero.")

    factor = float(args.co2e_per_unit)
    price = float(args.price_per_tonne)
    hours = float(args.hours)

    mean_rate = float(np.mean(x))
    wm = _windowed_means(x)
    if wm.size >= 5:
        lo_rate = float(np.percentile(wm, 10))
        hi_rate = float(np.percentile(wm, 90))
        spread_note = (f"10th and 90th percentile of the {wm.size} "
                       f"non-overlapping {WIN_FOR_SPREAD}-row window means of "
                       f"`{col}` in your own file")
    else:
        lo_rate = hi_rate = mean_rate
        spread_note = ("your file has too few rows to form a spread, so the "
                       "range collapses to the mean — a point with no error bar, "
                       "which is a weakness of the file, not a precise answer")

    cur = args.currency
    tonnes = lambda r: r * hours * factor          # noqa: E731
    gross = lambda r: tonnes(r) * price            # noqa: E731
    t_mean, t_lo, t_hi = tonnes(mean_rate), tonnes(lo_rate), tonnes(hi_rate)
    g_mean, g_lo, g_hi = gross(mean_rate), gross(lo_rate), gross(hi_rate)
    cus = lambda g: g * CUSTOMER_PCT / 100.0       # noqa: E731
    lat = lambda g: g * LATTICE_PCT / 100.0        # noqa: E731

    _imp = hours / x.size if x.size else float("nan")
    if not np.isfinite(_imp) or _imp <= 0:
        implied_note = ""
    elif _imp < 1.0 / 3600 or _imp > 24:
        implied_note = (f" — **check that.** {_imp:,.4g} h between consecutive "
                        f"rows is outside anything a process historian normally "
                        f"logs, so either the hours or the row count is not what "
                        f"you think. Every tonne below scales linearly with this "
                        f"number")
    else:
        implied_note = (" — if that is not your logging interval, the hours are "
                        "wrong and every tonne below is wrong in the same "
                        "proportion")

    L = ["\n## What this is worth\n"]
    L.append(f"**{t_lo:,.1f} to {t_hi:,.1f} tonnes CO2e** over the period you "
             f"gave, central estimate **{t_mean:,.1f} tCO2e**.\n")
    L.append(f"At the price you named, that is **{g_lo:,.0f} to {g_hi:,.0f} "
             f"{cur}** of market value, central **{g_mean:,.0f} {cur}**, split:\n")
    L.append(f"| | share | low | central | high |")
    L.append(f"|---|---|---|---|---|")
    L.append(f"| **You keep** | {CUSTOMER_PCT}% | {cus(g_lo):,.0f} {cur} | "
             f"**{cus(g_mean):,.0f} {cur}** | {cus(g_hi):,.0f} {cur} |")
    L.append(f"| Lattice24 | {LATTICE_PCT}% | {lat(g_lo):,.0f} {cur} | "
             f"{lat(g_mean):,.0f} {cur} | {lat(g_hi):,.0f} {cur} |")
    L.append(f"\nWe remediate at our cost and take {LATTICE_PCT}% of first-year "
             f"verified savings. You keep {CUSTOMER_PCT}%.\n")
    L.append("### The arithmetic, in full\n")
    L.append("| input | value | where it came from |")
    L.append("|---|---|---|")
    qunit = quantity_unit(args.value_unit)
    implied = hours / x.size if x.size else float("nan")
    L.append(f"| metered column | `{_cell(col)}` | you named it |")
    L.append(f"| unit of that column | {_cell(args.value_unit)} | you stated it |")
    L.append(f"| mean rate over the record | {mean_rate:,.6g} "
             f"{_cell(args.value_unit)} | arithmetic mean of {x.size:,} rows of "
             f"your file"
             + (f". **{dropped:,} of {x_raw.size:,} rows in `{_cell(col)}` were "
                f"blank, NaN or infinite and are not in this mean** — the "
                f"figures below price the {x.size:,} rows that were readable, "
                f"not your whole record"
                if dropped else "") + " |")
    L.append(f"| low / high rate | {lo_rate:,.6g} / {hi_rate:,.6g} "
             f"{args.value_unit} | {spread_note} |")
    L.append(f"| spread window | {WIN_FOR_SPREAD} rows | **the one constant in "
             f"this calculation that is ours, not yours.** It sets the WIDTH of "
             f"the range and nothing else: the interval above is a sampling "
             f"interval on the mean, so its width shrinks as 1/√(window) and a "
             f"narrow band here is not evidence of precision |")
    L.append(f"| hours covered | {hours:,.6g} | **you stated it, and nothing in "
             f"your file confirms it.** Across {x.size:,} rows that implies "
             f"**{implied:,.4g} hours between rows**{implied_note} |")
    L.append(f"| emission factor | {factor:,.6g} tCO2e per **{_cell(qunit)}** | "
             f"you stated it — source: {_cell(args.co2e_source)}. **Note the "
             f"unit: the factor is per {_cell(qunit)}, not per "
             f"{_cell(args.value_unit)}** — your rate is turned into a quantity "
             f"first, so a rate-based factor here would be the wrong number |")
    L.append(f"| market price | {price:,.6g} {_cell(cur)} per tonne CO2e | you "
             f"stated it — source: {_cell(args.price_source)}, priced "
             f"{_cell(args.price_date)}, market: **{_cell(args.market)}** |")
    L.append(f"\n    quantity = {mean_rate:,.6g} × {hours:,.6g} h "
             f"= {mean_rate * hours:,.6g} {qunit}")
    L.append(f"    tonnes   = {mean_rate * hours:,.6g} × {factor:,.6g} "
             f"= {t_mean:,.1f} tCO2e")
    L.append(f"    gross    = {t_mean:,.1f} × {price:,.6g} = {g_mean:,.0f} {cur}")
    L.append(f"    you {CUSTOMER_PCT}%  = {cus(g_mean):,.0f} {cur}"
             f"        Lattice24 {LATTICE_PCT}% = {lat(g_mean):,.0f} {cur}\n")
    L.append(f"That is the market value of the metered quantity over the period "
             f"you gave, **at the {args.market} rate you named on "
             f"{args.price_date}**. It is a price on a quantity, not a promise of "
             f"a saving: what fraction of that flow is recoverable is an "
             f"engineering question this file cannot answer.")

    L.append("\n### Two things this arithmetic cannot check\n")
    L.append(f"- **That `{_cell(col)}` is a rate at all.** It is multiplied by "
             f"hours as though one row were one {_cell(qunit)} per hour. If you "
             f"named a level rather than a flow — a pressure, a temperature, a "
             f"tank height — the arithmetic still runs and still prints a "
             f"number, and that number is meaningless. Nothing in a CSV says "
             f"which it is.")
    L.append(f"- **That the hours match the record.** {hours:,.6g} h over "
             f"{x.size:,} rows is {implied:,.4g} h per row. That is arithmetic "
             f"on what you typed, not a check against your data.")

    if label_result is not None:
        pv = label_result.get("p")
        ch = label_result.get("channel", "the primary channel")
        g0, g1 = label_result.get("g0", "group 0"), label_result.get("g1", "group 1")
        if label_result.get("separates"):
            L.append(f"\n**Your labels do separate.** The primary channel named "
                     f"before reading, `{_cell(ch)}`, distinguishes "
                     f"**{_cell(g0)}** from **{_cell(g1)}** at p = {pv:.4f}. "
                     f"That is what puts something under the split table above: "
                     f"there is a measured difference between the two "
                     f"conditions, so there is a fraction of the flow it is "
                     f"reasonable to argue about. It is still not a measurement "
                     f"of how much of it you can recover.")
        else:
            L.append(f"\n**Your labels do not separate, and that changes what "
                     f"the figure above is.** The primary channel named before "
                     f"reading, `{_cell(ch)}`, does not distinguish "
                     f"**{_cell(g0)}** from **{_cell(g1)}** — p = {pv:.4f}. "
                     f"Nothing in this file identifies a recoverable fraction, "
                     f"so the number above is the value of your **whole metered "
                     f"flow**, not of a saving anyone has evidence for. The "
                     f"{CUSTOMER_PCT}/{LATTICE_PCT} split is an offer structure; "
                     f"on this file there is nothing yet measured for it to be a "
                     f"split of.")

    if order_result is not None:
        if order_result.get("reads"):
            L.append(f"\nThe order screen above found ordinal structure in "
                     f"`{col}` ({order_result['fired']} of "
                     f"{order_result['ran']} windows). That is a reason to look "
                     f"at *when* the loss happens; it is still not a measurement "
                     f"of what is recoverable.")
        else:
            L.append(f"\n**This file gives no evidence that any part of that "
                     f"value is recoverable.** The order screen found no "
                     f"structure in `{col}` — {order_result['fired']} of "
                     f"{order_result['ran']} windows fired. The figure above is "
                     f"arithmetic on a mean you supplied a factor and a price "
                     f"for; nothing in the reading says a controllable pattern "
                     f"sits under it.")
    L.append("\n" + VERRA_STATUS)
    return L
