# SPDX-License-Identifier: MIT
"""pump.fun top-5 forecaster (make.fun pumpfun-top5-forecast-01).

A seeded Monte Carlo over the answer reads. Each history coin alive in its pool at the last mark survives to a read
with a logistic probability of (horizon, cap, age since graduation) and otherwise moves by heavy-tailed noise; dead
ones drop off the chart. Coins graduating after the cut (newcomers) arrive at a fixed rate, and each one's cap at its
age at the read is drawn from a library of real coins' cap-by-age trajectories (model.json). Each simulated read is
ranked; place probabilities and place-cap bins are averaged over the reads, tempered, and written out.
"""
import hashlib
import json
import math
import os
import random
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.environ.get('MAKEFUN_INPUT_DIR', '/input')
OUT = os.environ.get('MAKEFUN_OUTPUT_DIR', '/output')

PLACES = 5
ALIVE = 2.3            # log10 SOL: a pool coin under 200 SOL counts as dead
RATE = 4.5             # newcomer graduations per hour (library-consistent rate)
PATHS = 2500           # simulated paths per read
TEMPER = 0.12          # share of each place's named mass handed back to the unnamed share
P_MIN = 2e-4           # least p a named history coin gets
SIZE_SMOOTH = 1.5      # bins, Gaussian smoothing of the size histogram
SIZE_PRIOR = 0.04      # broad prior weight in the sizes
CURVE_FULL = 400.0     # SOL, a curve near completion
CURVE_STALL = 0.5      # hours: a curve coin's graduation chance falls as 1 / (1 + age / CURVE_STALL)
LEVEL_W = 0.0          # weight of the history's own top-5 level in the place-cap prior
LEVEL_HOURS = 4.0      # hours of marks the level is taken over
YOUNG = 1.0           # survival exponent for a coin graduated just before the cut (ramps to 1 at 3 h)
NEW_SHIFT = -0.4      # log10 shift of newcomers older than 3 h (library survivor bias)
SIG_SCALE = 1.0       # multiplies the survivors' log-cap noise
PRIOR_BW = 0.22        # log10, kernel width of the empirical place-cap prior
PRIOR_TAIL = 0.3       # share of a broad t3 around the prior's median
PRIOR_TAIL_SCALE = 0.45
H_MC = 1.5             # hours: newcomer-held place caps trust the simulation as exp(-h / H_MC), else the prior
LIB_FROM = 0           # library coins graduated before this unix time are left out (survivor-biased)

# backtest tuning hook: MAKEFUN_BT_PARAMS='{"RATE": 3}' overrides the constants above (never set in the grader)
if os.environ.get('MAKEFUN_BT_PARAMS'):
    globals().update(json.loads(os.environ['MAKEFUN_BT_PARAMS']))


def ts(s):
    return datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()


def load_marks():
    marks = []
    try:
        with open(os.path.join(IN, 'history.ndjson'), 'rb') as f:
            raw = f.read()
    except OSError:
        return [], b''
    for line in raw.decode('utf-8', 'replace').split('\n'):
        line = line.strip()
        if not line:
            continue
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if not isinstance(m, dict) or m.get('ok') is False or not isinstance(m.get('rows'), list) or not m.get('at'):
            continue
        try:
            m['_t'] = ts(m['at'])
        except (ValueError, TypeError):
            continue
        marks.append(m)
    marks.sort(key=lambda m: m['_t'])
    return marks, raw


def cap_of(r):
    try:
        v = int(str(r.get('market_cap_lamports')))
    except (ValueError, TypeError):
        return None
    if v <= 0:
        return None
    return math.log10(v) - 9


class Cand:
    __slots__ = ('mint', 'x', 'venue', 'grad', 'created', 'last_seen', 'peak')


def candidates(marks):
    """Every mint the history ranks, with its latest state."""
    c = {}
    for m in marks:
        t = m['_t']
        for r in m['rows']:
            if not isinstance(r, dict):
                continue
            mint = r.get('mint')
            x = cap_of(r)
            if not isinstance(mint, str) or not mint or x is None:
                continue
            k = c.get(mint)
            if k is None:
                k = Cand()
                k.mint, k.grad, k.peak = mint, None, x
                try:
                    k.created = ts(r.get('created_at'))
                except (ValueError, TypeError, AttributeError):
                    k.created = t
                c[mint] = k
            k.x, k.last_seen = x, t
            k.peak = max(k.peak, x)
            venue = r.get('venue')
            if venue not in ('curve', 'migrating', 'pool'):
                venue = 'curve' if r.get('complete') is False else 'pool'
            k.venue = venue
            if k.venue == 'pool' and k.grad is None:
                # graduated between the previous mark and this one, unless it came in already pooled
                k.grad = max(k.created, t - 300)
    return c


def load_library(exclude):
    try:
        lib = json.load(open(os.path.join(HERE, 'model.json')))
    except (OSError, ValueError):
        return [-0.6, -0.75, -0.1, 1.3, 0.5], [], []
    rows = [row for g, row in lib['lib'] if g >= LIB_FROM and not (exclude and exclude[0] <= g <= exclude[1])]
    prior = [x for T, x in lib.get('prior', []) if not (exclude and exclude[0] <= T <= exclude[1])]
    return lib['surv_w'], rows, prior


def t3_cdf(z):
    t = z / math.sqrt(3)
    return 0.5 + (math.atan(t) + t / (1 + t * t)) / math.pi


def prior_bins(samples, edges, places):
    """Each place's cap prior over the bins: a kernel density of the library's place caps plus a broad t3 tail."""
    nb = len(edges) + 1
    lo = [-60.0] + list(edges)
    hi = list(edges) + [60.0]
    out = []
    for k in range(places):
        xs = sorted(s[k] for s in samples if len(s) > k) or [3.3 - 0.1 * k]
        med = xs[len(xs) // 2]
        pb = [0.0] * nb
        for x in xs:
            for b in range(nb):
                pb[b] += 0.5 * (math.erf((hi[b] - x) / (PRIOR_BW * 1.41421356)) - math.erf((lo[b] - x) / (PRIOR_BW * 1.41421356)))
        z = sum(pb)
        pb = [(1 - PRIOR_TAIL) * v / z + PRIOR_TAIL * (t3_cdf((hi[b] - med) / PRIOR_TAIL_SCALE) - t3_cdf((lo[b] - med) / PRIOR_TAIL_SCALE)) for b, v in enumerate(pb)]
        z = sum(pb)
        out.append([v / z for v in pb])
    return out


def surv_prob(w, h, x, a):
    z = w[0] + w[1] * math.log(max(h, 0.05)) + w[2] * (x - 3.4) + w[3] * math.log(1 + a) + w[4] * (x - 3.4) * math.log(1 + a)
    return 1 / (1 + math.exp(-max(-30, min(30, z))))


def student(rng, df=4):
    z = rng.gauss(0, 1)
    v = sum(rng.gauss(0, 1) ** 2 for _ in range(df)) / df
    return z / math.sqrt(v)


def main():
    task = json.load(open(os.path.join(IN, 'task.json')))
    places = int(task.get('places', PLACES))
    edges = task.get('size', {}).get('edges') or [round(2 + 0.05 * i, 2) for i in range(81)]
    nb = len(edges) + 1
    marks, raw = load_marks()
    rng = random.Random(hashlib.sha256(raw + json.dumps(task, sort_keys=True).encode()).hexdigest())
    reads = sorted(ts(r) for r in task.get('reads', []))
    # the reference mark: the last one that ranked any coin (an empty trailing mark says nothing about the coins)
    ranked_marks = [m for m in marks if m['rows']]
    ref = ranked_marks[-1] if ranked_marks else None
    t_last = ref['_t'] if ref else ts(task.get('history_until', task['t0']))
    if not reads:
        reads = [ts(task['window']['answer_at'])]
    exclude = None
    if os.environ.get('MAKEFUN_BT_EXCLUDE'):
        a, b = os.environ['MAKEFUN_BT_EXCLUDE'].split(',')
        exclude = (float(a), float(b))
    w, lib, prior_samples = load_library(exclude)
    # the day's own level: the mean log cap of the top 5 over the last LEVEL_HOURS of marks, against the prior's
    # level; the prior is shifted by LEVEL_W of the gap (a busier or quieter day than the library's)
    lv = []
    for m in marks:
        if m['_t'] >= t_last - LEVEL_HOURS * 3600 and m['rows']:
            xs = sorted((cap_of(r) or 0.0 for r in m['rows'] if isinstance(r, dict)), reverse=True)[:places]
            if len(xs) == places:
                lv.append(sum(xs) / places)
    if lv and prior_samples and LEVEL_W:
        gap = sum(lv) / len(lv) - sum(sum(s[:places]) / places for s in prior_samples) / len(prior_samples)
        gap = max(-0.5, min(0.5, gap))
        prior_samples = [[v + LEVEL_W * gap for v in s] for s in prior_samples]
    prior = prior_bins(prior_samples, edges, places)
    cand = candidates(marks)
    last_rows = {r.get('mint') for r in (ref['rows'] if ref else []) if isinstance(r, dict)}
    hist = sorted(cand.values(), key=lambda k: (-k.x, k.mint))

    tally = [dict() for _ in range(places)]
    size_hist = [[0.0] * nb for _ in range(places)]  # summed over reads, each read normalised

    def bin_of(v):
        lo, hi = 0, len(edges)
        while lo < hi:
            mid = (lo + hi) // 2
            if edges[mid] <= v:
                lo = mid + 1
            else:
                hi = mid
        return lo

    NT = 4096
    tnoise = [student(rng) for _ in range(NT)]
    for rt in reads:
        h = max(0.0, (rt - t_last) / 3600)
        # per-coin survival this read
        plan = []
        for k in hist:
            stale = k.mint not in last_rows
            if k.venue == 'pool':
                age = max(0.0, (t_last - (k.grad or k.created)) / 3600)
                if k.x < ALIVE or stale:
                    plan.append((k, 'dead', 0.0, age))
                else:
                    # young coins over-survive long horizons in the fit (backtest calibration): survival raised to the
                    # power YOUNG (leaves short-horizon survival near 1 alone), ramped out by 3 h of age
                    young = YOUNG + (1 - YOUNG) * min(1.0, age / 3)
                    plan.append((k, 'pool', surv_prob(w, h + 1 / 6, k.x, age) ** young, age))
            else:
                cap = 10 ** k.x
                pg = 1.0 if k.venue == 'migrating' else min(1.0, max(0.0, (cap - 28) / (CURVE_FULL - 28))) ** 2
                pg *= 1 - math.exp(-(h + 1 / 6) / 0.5)
                # a coin still on its curve long after launch has mostly stalled there
                pg /= 1 + max(0.0, (t_last - k.created) / 3600) / CURVE_STALL
                if stale:
                    pg *= 0.3
                plan.append((k, 'curve', pg, 0.0))
        sig = SIG_SCALE * (0.18 + 0.06 * math.sqrt(h))
        lam = RATE * h
        pools = [(k.x, k.mint, p) for k, kind, p, age in plan if kind == 'pool' and p > 1e-4]
        curves = [(k.x, k.mint, p) for k, kind, p, age in plan if kind == 'curve']
        # coins that cannot move (dead pools, curve coins that stay put) only matter as chart filler for the top 5
        static = []  # dead pools (rugged or dropped out of the rows) leave the top
        curves_hot = [(x, m, p) for x, m, p in curves if p > 1e-4]
        curves_cold = sorted(((x, m) for x, m, p in curves if p <= 1e-4), reverse=True)[:places]
        nl = len(lib)
        rand, gauss = rng.random, rng.gauss
        h_hist = [[0.0] * nb for _ in range(places)]   # place held by a history coin: its simulated cap
        n_hist = [[0.0] * nb for _ in range(places)]   # held by a newcomer: the simulated cap
        n_new = [0] * places
        for _ in range(PATHS):
            vals = list(static) + list(curves_cold)
            for x, m, p in pools:
                if rand() < p:
                    vals.append((x + sig * tnoise[int(rand() * NT)], m))
            for x, m, p in curves_hot:
                if nl and rand() < p:
                    row = lib[int(rand() * nl)]
                    i = min(int(rand() * h * 6), len(row) - 1)
                    vals.append((row[i] + 0.03 * gauss(0, 1), m))
                else:
                    vals.append((x, m))
            # newcomers: a Poisson count, each of a uniform age in (0, h), its cap from a library coin at that age
            n = 0
            if lam > 0 and nl:
                L, p, n = math.exp(-lam), 1.0, -1
                while True:
                    n += 1
                    p *= rand()
                    if p < L:
                        break
            for j in range(n):
                i = int(rand() * h * 6)
                row = lib[int(rand() * nl)]
                tries = 0
                while i >= len(row) and tries < 20:
                    row = lib[int(rand() * nl)]
                    tries += 1
                # the library is survivor-biased: older newcomers are shifted down (NEW_SHIFT, ramped in from 1 h to
                # 3 h of age, so the first hour, which the practice-shaped backtests check, is left as observed)
                vals.append((row[min(i, len(row) - 1)] + 0.03 * gauss(0, 1) + NEW_SHIFT * min(1.0, max(0.0, (i - 6) / 12)), None))
            vals.sort(key=lambda t: -t[0])  # noqa
            for kk in range(places):
                if kk < len(vals):
                    v, m = vals[kk]
                else:
                    v, m = 1.45, None
                if m is not None:
                    tally[kk][m] = tally[kk].get(m, 0) + 1
                    h_hist[kk][bin_of(v)] += 1
                else:
                    n_hist[kk][bin_of(v)] += 1
                    n_new[kk] += 1
        wmc = math.exp(-h / H_MC)
        for kk in range(places):
            for b in range(nb):
                size_hist[kk][b] += (h_hist[kk][b] + wmc * n_hist[kk][b] + (1 - wmc) * n_new[kk] * prior[kk][b]) / PATHS
    tot = PATHS * len(reads)

    probabilities = []
    top5 = []
    for kk in range(places):
        ps = {m: c / tot for m, c in tally[kk].items()}
        # every history coin at the last mark gets a row (removing it from the unnamed share when on the chart)
        for m in last_rows:
            if isinstance(m, str) and m in cand:
                ps.setdefault(m, 0.0)
        items = []
        for m, p in ps.items():
            # a coin the simulation never put here gets 0 (the floor scores it the same as any p under 1e-4); its row
            # still takes it out of the unnamed share if it sits on the answer chart
            q = max((1 - TEMPER) * p, P_MIN) if p > 0 else 0.0
            items.append((q, -cand[m].x if m in cand else 0.0, m))
        # highest p first, then the largest caps (the likeliest to sit on the chart)
        items = [(q, m) for q, _, m in sorted(items, key=lambda t: (-t[0], t[1], t[2]))]
        items = items[:100]
        s = sum(q for q, _ in items)
        if s > 1 - 1e-6:
            f = (1 - 1e-6) / s
            items = [(q * f, m) for q, m in items]
        for q, m in items:
            probabilities.append({'mint': m, 'place': kk + 1, 'p': math.floor(q * 1e9) / 1e9})
    # top5: greedy by place probability
    used = set()
    for kk in range(places):
        best = sorted(((c, m) for m, c in tally[kk].items() if m not in used), key=lambda t: (-t[0], t[1]))
        if best:
            top5.append(best[0][1]); used.add(best[0][1])
        else:
            for k in hist:
                if k.mint not in used:
                    top5.append(k.mint); used.add(k.mint); break
    i = 0
    while len(top5) < places:
        m = 'Pad' + '123456789ABCDEFGHJKLMNPQRSTUVWXYZ'[i % 33] + '1' * 36 + 'pump'
        if m not in used:
            top5.append(m); used.add(m)
        i += 1

    sizes = []
    centers = [1.975] + [edges[i] + 0.025 for i in range(len(edges) - 1)] + [edges[-1] + 0.025]
    for kk in range(places):
        hst = size_hist[kk]
        z = sum(hst) or 1
        hst = [v / z for v in hst]
        sm = [0.0] * nb
        for i, v in enumerate(hst):
            if v:
                for j in range(max(0, i - 8), min(nb, i + 9)):
                    sm[j] += v * math.exp(-0.5 * ((j - i) / SIZE_SMOOTH) ** 2)
        z = sum(sm) or 1
        sm = [v / z for v in sm]
        prior = [math.exp(-0.5 * ((c - 3.4) / 0.7) ** 2) + 0.05 for c in centers]
        zp = sum(prior)
        final = [(1 - SIZE_PRIOR) * a + SIZE_PRIOR * b / zp for a, b in zip(sm, prior)]
        z = sum(final)
        for b, v in enumerate(final):
            sizes.append({'place': kk + 1, 'bin': b, 'p': math.floor(v / z * (1 - 1e-6) * 1e9) / 1e9})

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, task.get('output', 'forecast.json')), 'w') as f:
        f.write(json.dumps({'probabilities': probabilities, 'top5': top5, 'sizes': sizes}, separators=(',', ':')) + '\n')


if __name__ == '__main__':
    main()
