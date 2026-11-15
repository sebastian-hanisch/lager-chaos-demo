"""Kern: Lager mit linearer Bestellregel und Klemmung (Bestellung >= 0), Lyapunov-Exponenten und Normalform der Randkollision.

Modell (Abweichungen vom Soll, mittlerer Bedarf D, Lieferzeit L Perioden, Anpassung alpha, kein Leitungsgewicht):
    Zustand  x_t = (i_t, u_{t-1}, ..., u_{t-L+1})      i = Bestand minus Soll, u = Bestellung minus D
    Regel    u_t = max(-alpha * i_t, -D)                 die Bestellung D + u_t darf nicht unter 0 fallen (Klemmung)
    Lager    i_{t+1} = i_t + u_{t+1-L} - e_t             e_t = Bedarfsabweichung (0 ohne Rauschen)
Alle Exponenten gelten je Periode. Zwei Verfahren: Benettin (Tangentialvektor, die Ableitung der Klemmung ist 0) und Zweibahn
(Abstand zweier Bahnen, auf d0 zurückgesetzt). Die Rechnung läuft über ein Gitter von alpha vektorisiert (numpy).

Die Messlogik ist neu geschrieben; Vorbild waren die Wegwerf-Messskripte der Planung (nicht importiert).
Nie alpha, Schrittweiten oder Startwerte als Zweierpotenz voreinstellen: in Gleitkomma rechnet der Rechner dann in Dualbrüchen exakt
(siehe `brent_cycle`, `exact_cycle` und den Abschnitt Zweierpotenz-Falle der App)."""

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from lgc_rng import normal_array

D_STANDARD = 10.0
BURN = 6000            # Einschwingschritte (Gitter)
REC = 6000             # Messschritte (Gitter)
LAMBDA_SCHWELLE = 0.003   # chaotisch heißt: größter Exponent > 0,003 je Periode bei 6000 Messschritten
START_I0 = 0.5         # Startbestand der Gitter (alpha nie Zweierpotenz, deshalb unkritisch)
D0_ZWEIBAHN = 1e-9
OBERHALB = 1.0005      # "oberhalb alpha_krit": alpha > 1,0005 alpha_krit


def alpha_crit(L: int) -> float:
    """Stabilitätsgrenze der linearen Regel: alpha_krit = 2 sin(pi / (4L - 2)); kritische Periode 4L - 2."""
    return 2.0 * math.sin(math.pi / (4 * L - 2))


def alpha_grid(lo: float, hi: float, n: int) -> np.ndarray:
    """Gitter über alpha mit nicht-dyadischem Schritt und kleinem irrationalem Versatz (nie eine Zweierpotenz)."""
    return lo + (np.arange(n) + 0.37) * (hi - lo) / n + 1.234567e-6 * math.sqrt(2)


def classify(lam: float, schwelle: float = LAMBDA_SCHWELLE) -> str:
    """'chaotisch' (lam > Schwelle), 'periodisch' (lam < -Schwelle, Fenster), sonst 'unentschieden' (|lam| <= Schwelle)."""
    if lam > schwelle:
        return "chaotisch"
    if lam < -schwelle:
        return "periodisch"
    return "unentschieden"


def jacobian(L: int, alpha: float, clipped: bool) -> np.ndarray:
    """Jacobi-Matrix des Schritts (Zustand -> Zustand); bei Klemmung ist die Ableitung der Bestellung nach dem Bestand 0."""
    c = 0.0 if clipped else -alpha
    if L == 1:
        return np.array([[1.0 + c]])
    J = np.zeros((L, L))
    J[0, 0] = 1.0
    J[0, L - 1] += 1.0
    J[1, 0] = c
    for k in range(2, L):
        J[k, k - 1] = 1.0
    return J


# --- eine Bahn -------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Orbit:
    i: np.ndarray        # Bestand i_0 ... i_n (n + 1 Werte)
    u: np.ndarray        # Bestellung minus Bedarf u_0 ... u_{n-1}
    clipped: np.ndarray  # True, wo die Klemmung greift (Bestellung genau 0)


def simulate(L: int, alpha: float, D: float, i0: float, n: int, noise=None, w0=None) -> Orbit:
    """Eine Bahn mit n Schritten ab x_0 = (i0, w0 ...) (Standard: Leitung leer, w0 = 0). `noise`: Feld e_0 ... e_{n-1}."""
    x = [float(i0)] + (list(w0) if w0 is not None else [0.0] * (L - 1))
    iv = np.empty(n + 1)
    uv = np.empty(n)
    cl = np.zeros(n, bool)
    iv[0] = x[0]
    for t in range(n):
        cx = -alpha * x[0]
        if cx < -D:
            u = -D
            cl[t] = True
        else:
            u = cx
        e = 0.0 if noise is None else float(noise[t])
        if L == 1:
            x = [x[0] + u - e]
        else:
            x = [x[0] + x[L - 1] - e, u] + x[1:L - 1]
        iv[t + 1] = x[0]
        uv[t] = u
    return Orbit(iv, uv, cl)


def orbit_summary(L: int, alpha: float, D: float, i0: float, burn: int = BURN, rec: int = REC) -> dict:
    """Kennzahlen einer einzelnen Bahn: Exponent (Benettin), Spannweite/D und Zahl der Klemmschritte im Messfenster, Urteil."""
    res = lyapunov_grid(L, [alpha], D=D, burn=burn, rec=rec, i0=i0)
    return {
        "lam": float(res.lam[0]), "span": float(res.span[0]), "clip_frac": float(res.clip_frac[0]),
        "clip_count": int(round(res.clip_frac[0] * rec)), "urteil": classify(float(res.lam[0])),
    }


# --- Gitter über alpha: Benettin und Zweibahn ------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class GridResult:
    alphas: np.ndarray
    lam: np.ndarray         # größter Exponent je Periode
    span: np.ndarray        # (max i - min i) / D im Messfenster
    clip_frac: np.ndarray   # Anteil der Perioden mit Klemmung im Messfenster
    peaks: tuple | None = None  # (Index des alpha, Spitze) der letzten Spitzen des Bestands, nur wenn verlangt


def _advance(X, a, D, L, e=None):
    """Ein Schritt für alle Gitterpunkte (X hat die Form (L, n)); gibt die Maske der Klemmung zurück."""
    cx = -a * X[0]
    clipped = cx < -D
    u = np.where(clipped, -D, cx)
    if L == 1:
        X[0] = X[0] + u
    else:
        nx0 = X[0] + X[L - 1]
        if L > 2:
            X[2:] = X[1:L - 1]
        X[1] = u
        X[0] = nx0
    if e is not None:
        X[0] -= e
    return clipped


def _advance_tangent(V, a, clipped, L):
    cv = np.where(clipped, 0.0, -a * V[0])
    if L == 1:
        V[0] = V[0] + cv
    else:
        nv0 = V[0] + V[L - 1]
        if L > 2:
            V[2:] = V[1:L - 1]
        V[1] = cv
        V[0] = nv0


def peak_points(hist: np.ndarray, per: int = 30):
    """Lokale Maxima je Spalte (Gitterpunkt) von `hist` (Zeit, Gitter): (Spaltenindex, Wert) der letzten `per` Spitzen."""
    mask = (hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:])
    cols, vals = [], []
    for j in range(hist.shape[1]):
        idx = np.nonzero(mask[:, j])[0][-per:]
        cols.extend([j] * len(idx))
        vals.extend(hist[1:-1][idx, j])
    return np.array(cols, int), np.array(vals, float)


def lyapunov_grid(L, alphas, D=D_STANDARD, burn=BURN, rec=REC, i0=START_I0, noise=None, peaks=0, peaks_per=30) -> GridResult:
    """Benettin-Verfahren über ein Gitter von alpha: Tangentialvektor, nach jedem Schritt auf Länge 1 gesetzt, Mittel von ln(Länge).

    `noise`: Feld e_0 ... e_{burn+rec-1}, dieselbe Folge für alle Gitterpunkte (Exponent entlang der Rauschfolge);
    ein Feld der Form (burn + rec, n) gibt jedem Gitterpunkt seine eigene Folge.
    `peaks` > 0: speichert die letzten `peaks` Bestände des Messfensters und liefert die Spitzen."""
    a = np.asarray(alphas, float)
    n = a.size
    X = np.zeros((L, n))
    X[0] = i0
    V = np.full((L, n), 1.0 / math.sqrt(L))
    acc = np.zeros(n)
    clips = np.zeros(n)
    lo = np.full(n, np.inf)
    hi = np.full(n, -np.inf)
    hist = np.empty((peaks, n)) if peaks else None
    total = burn + rec
    for t in range(total):
        clipped = _advance(X, a, D, L, None if noise is None else noise[t])
        _advance_tangent(V, a, clipped, L)
        nv = np.sqrt((V * V).sum(axis=0))
        nv = np.maximum(nv, 1e-300)
        V /= nv
        if t >= burn:
            acc += np.log(nv)
            clips += clipped
            np.minimum(lo, X[0], out=lo)
            np.maximum(hi, X[0], out=hi)
        if peaks and t >= total - peaks:
            hist[t - (total - peaks)] = X[0]
    pk = peak_points(hist, peaks_per) if peaks else None
    return GridResult(a, acc / rec, (hi - lo) / D, clips / rec, pk)


def two_orbit_grid(L, alphas, D=D_STANDARD, burn=BURN, rec=REC, i0=START_I0, d0=D0_ZWEIBAHN, noise=None) -> np.ndarray:
    """Zweibahn-Verfahren (zweite Methode): nach dem Einschwingen eine Nachbarbahn im Abstand d0, nach jedem Schritt auf d0 zurückgesetzt.

    Gibt den Exponenten je Periode zurück (Mittel von ln(Abstand / d0)); Schritte mit Abstand 0 werden nicht gezählt."""
    a = np.asarray(alphas, float)
    n = a.size
    X = np.zeros((L, n))
    X[0] = i0
    for t in range(burn):
        _advance(X, a, D, L, None if noise is None else noise[t])
    Y = X.copy()
    Y[0] += d0
    acc = np.zeros(n)
    cnt = np.zeros(n)
    for t in range(burn, burn + rec):
        e = None if noise is None else noise[t]
        _advance(X, a, D, L, e)
        _advance(Y, a, D, L, e)
        diff = Y - X
        dist = np.sqrt((diff * diff).sum(axis=0))
        ok = dist > 0
        safe = np.where(ok, dist, 1.0)
        acc += np.where(ok, np.log(safe / d0), 0.0)
        cnt += ok
        Y = X + diff * (d0 / safe)
        Y[0] = np.where(ok, Y[0], X[0] + d0)
    return acc / np.maximum(cnt, 1)


def chaos_counts(res_or_lam, alphas, L: int, schwelle: float = LAMBDA_SCHWELLE) -> dict:
    """Zählung oberhalb alpha_krit: Punkte, chaotische Punkte, Punkte mit lam < -Schwelle (Fenster), Läufe der Nicht-Chaos-Punkte."""
    lam = np.asarray(res_or_lam.lam if hasattr(res_or_lam, "lam") else res_or_lam)
    a = np.asarray(alphas)
    above = a > alpha_crit(L) * OBERHALB
    ch = lam > schwelle
    win = lam < -schwelle
    non = (~ch) & above
    return {
        "n_above": int(above.sum()), "n_chaotic": int((ch & above).sum()), "n_window": int((win & above).sum()),
        "share_chaotic": float((ch & above).sum() / max(above.sum(), 1)), "runs_non_chaotic": count_runs(non[above]),
        "lam_min": float(lam[above].min()) if above.any() else float("nan"),
    }


def count_runs(mask) -> int:
    """Zahl zusammenhängender Läufe von True."""
    m = np.asarray(mask, int)
    if m.size == 0:
        return 0
    return int(np.sum((m[1:] == 1) & (m[:-1] == 0)) + (m[0] == 1))


def chaotic_ranges(alphas, lam, L: int, schwelle: float = LAMBDA_SCHWELLE):
    """Zusammenhängende Bereiche (alpha von, bis) oberhalb alpha_krit, in denen jeder Gitterpunkt chaotisch ist."""
    runs, cur = [], None
    for a, l in zip(alphas, lam):
        if a <= alpha_crit(L) * OBERHALB:
            continue
        if l > schwelle:
            cur = [a, a] if cur is None else [cur[0], a]
        elif cur is not None:
            runs.append(tuple(cur))
            cur = None
    if cur is not None:
        runs.append(tuple(cur))
    return runs


def noise_sequence(seed: int, n: int, s: float, D: float = D_STANDARD) -> np.ndarray:
    """Bedarfsabweichung e_t ~ N(0, (s D)^2) aus SplitMix64 (gleiche Folge für alle Gitterpunkte und beide Bahnen)."""
    return s * D * normal_array(seed, n)


# --- Vorhersagehorizont ---------------------------------------------------------------------------------------------------------------

def two_orbit_distance(L, alpha, D=D_STANDARD, n=2000, burn=3000, d0=D0_ZWEIBAHN, i0=START_I0) -> np.ndarray:
    """Abstand (euklidisch im Zustand) zweier Lager mit Anfangsunterschied d0 im Bestand, n + 1 Werte ab dem Unterschied.

    Beide starten nach `burn` Schritten Einschwingen auf derselben Bahn, die zweite mit i + d0. d0 = 1e-9 (keine Zweierpotenz)."""
    X = np.zeros((L, 1))
    X[0] = i0
    a = np.array([alpha], float)
    for _ in range(burn):
        _advance(X, a, D, L)
    Y = X.copy()
    Y[0] += d0
    dist = np.empty(n + 1)
    dist[0] = float(np.sqrt(((Y - X) ** 2).sum()))
    for t in range(n):
        _advance(X, a, D, L)
        _advance(Y, a, D, L)
        dist[t + 1] = float(np.sqrt(((Y - X) ** 2).sum()))
    return dist


def first_visible(dist: np.ndarray, schwelle: float):
    """Erste Periode, in der der Abstand die Schwelle überschreitet, sonst None."""
    idx = np.nonzero(dist > schwelle)[0]
    return int(idx[0]) if idx.size else None


# --- Normalform der Randkollision -----------------------------------------------------------------------------------------------------

def lager_normal_form(alpha: float, D: float) -> dict:
    """Das Lager mit L = 2 als Normalform G(x, y) = (tau x + y + mu, -delta x) mit x = i - D/alpha, y = w + D (w = u_{t-1}).

    Links (x <= 0, Klemmung inaktiv): tau = 1, delta = alpha; rechts (x > 0, geklemmt): tau = 1, delta = 0; mu = -D."""
    return {"tau_L": 1.0, "delta_L": alpha, "tau_R": 1.0, "delta_R": 0.0, "mu": -D}


def normal_form_orbit(alpha: float, D: float, i0: float, w0: float, n: int) -> np.ndarray:
    """Bahn der Normalform des Lagers mit L = 2 in den Koordinaten (x, y); Form (n + 1, 2)."""
    x, y = i0 - D / alpha, w0 + D
    out = np.empty((n + 1, 2))
    out[0] = (x, y)
    for t in range(n):
        x, y = (x + y - D, -alpha * x if x <= 0 else 0.0)
        out[t + 1] = (x, y)
    return out


def normal_form_lyapunov(tl, dl, trs, dr, mu=1.0, n=60000, burn=3000, x0=(0.3, 0.1), limit=1e8) -> np.ndarray:
    """Exponent der Normalform G(x, y) = (tau x + y + mu, -delta x) (links x <= 0, rechts x > 0) über ein Feld von tau_R.

    Benettin wie beim Lager. NaN, wenn die Bahn unbeschränkt ist (|x| > limit)."""
    trs = np.asarray(trs, float)
    x = np.full(trs.shape, x0[0])
    y = np.full(trs.shape, x0[1])
    v1 = np.ones(trs.shape)
    v2 = np.zeros(trs.shape)
    acc = np.zeros(trs.shape)
    cnt = 0
    dead = np.zeros(trs.shape, bool)
    for t in range(burn + n):
        left = x <= 0
        tau = np.where(left, tl, trs)
        det = np.where(left, dl, dr)
        x, y = tau * x + y + mu, -det * x
        v1, v2 = tau * v1 + v2, -det * v1
        nv = np.maximum(np.hypot(v1, v2), 1e-300)
        v1 = v1 / nv
        v2 = v2 / nv
        if t >= burn:
            acc += np.log(nv)
            cnt += 1
        newly = np.abs(x) > limit
        if newly.any():
            dead |= newly
            x = np.where(dead, 0.0, x)
            y = np.where(dead, 0.0, y)
    lam = acc / max(cnt, 1)
    lam[dead] = np.nan
    return lam


def condition_17(tl, dl, tr, dr) -> bool:
    """Bedingung (17) des Aufsatzes: tau_L > 1 + delta_L und tau_R < 1 + delta_R (Randkollisionspaar, Fixpunkte nur für mu > 0)."""
    return bool(tl > 1 + dl and tr < 1 + dr)


def condition_18(tl, dl, tr, dr) -> bool:
    """Bedingung (18): tau_L > 1 + delta_L und tau_R < -(1 + delta_R) (Gebiet, in dem der Aufsatz den Attraktor untersucht)."""
    return bool(tl > 1 + dl and tr < -(1 + dr))


def condition_19_value(tl, dl, tr, dr):
    """Linke Seite von (19) (Stabilität des chaotischen Attraktors, > 0 verlangt); None, wenn die Eigenwerte der linken Matrix komplex sind.

    delta_L tau_R l1 - delta_R l1 l2 + delta_R l2 - delta_L tau_R + tau_L delta_L - delta_L^2 - l2 delta_L mit den Eigenwerten l1 >= l2 der linken Matrix."""
    disc = tl * tl - 4 * dl
    if disc < 0:
        return None
    l1 = (tl + math.sqrt(disc)) / 2
    l2 = (tl - math.sqrt(disc)) / 2
    return dl * tr * l1 - dr * l1 * l2 + dr * l2 - dl * tr + tl * dl - dl * dl - l2 * dl


def condition_19(tl, dl, tr, dr) -> bool:
    v = condition_19_value(tl, dl, tr, dr)
    return v is not None and v > 0


def robust_chaos_region(tl, dl, tr, dr) -> bool:
    """Gebiet des Aufsatzes für robustes Chaos: 1 > delta_L > 0, 1 > delta_R > 0, tau_R < -(1 + delta_R), tau_L > 2 sqrt(delta_L) und (19)."""
    return bool(0 < dl < 1 and 0 < dr < 1 and condition_18(tl, dl, tr, dr) and tl > 2 * math.sqrt(dl) and condition_19(tl, dl, tr, dr))


# --- Zyklen: Zweierpotenz-Falle ----------------------------------------------------------------------------------------------------

def make_step(L: int, alpha, D):
    """Schrittfunktion auf Tupeln (i, u_{t-1}, ...); rechnet in dem Zahlentyp von alpha und D (float oder Fraction)."""
    if L == 1:
        def step1(x):
            cx = -alpha * x[0]
            return (x[0] + (-D if cx < -D else cx),)
        return step1
    if L == 2:
        def step2(x):
            cx = -alpha * x[0]
            return (x[0] + x[1], -D if cx < -D else cx)
        return step2
    if L == 3:
        def step3(x):
            cx = -alpha * x[0]
            return (x[0] + x[2], -D if cx < -D else cx, x[1])
        return step3

    def stepn(x):
        cx = -alpha * x[0]
        return (x[0] + x[L - 1], -D if cx < -D else cx) + x[1:L - 1]
    return stepn


def brent_cycle(step, x0, cap: int):
    """Zyklus der Folge x, step(x), ... nach Brent: (Periode, Vorperiode) oder None, wenn innerhalb von `cap` Schritten keiner gefunden ist.

    Gleichheit heißt bitweise gleicher Zustand (Gleitkomma) bzw. gleicher Bruch."""
    power = lam = 1
    tort = x0
    hare = step(x0)
    count = 1
    while tort != hare:
        if power == lam:
            tort = hare
            power *= 2
            lam = 0
        hare = step(hare)
        lam += 1
        count += 1
        if count > cap:
            return None
    tort = hare = x0
    for _ in range(lam):
        hare = step(hare)
    mu = 0
    while tort != hare:
        tort = step(tort)
        hare = step(hare)
        mu += 1
    return lam, mu


def start_state(L: int, i0):
    return (i0,) + (type(i0)(0),) * (L - 1)


def exact_cycle(L: int, alpha: Fraction, D: Fraction, i0: Fraction, cap: int = 200000):
    """Exakte Bahn mit Brüchen: (Vorperiode, Periode, Zustände des Zyklus) oder None."""
    step = make_step(L, alpha, D)
    x = start_state(L, i0)
    seen = {}
    t = 0
    while x not in seen:
        if t > cap:
            return None
        seen[x] = t
        x = step(x)
        t += 1
    mu = seen[x]
    cyc = [s for s, k in seen.items() if k >= mu]
    return mu, t - mu, cyc


def cycle_multipliers(L: int, alpha: float, D: float, cycle_states) -> np.ndarray:
    """Beträge der Multiplikatoren eines Zyklus: Eigenwerte des Produkts der Jacobi-Matrizen längs des Zyklus (Zyklusreihenfolge)."""
    M = np.eye(L)
    for s in cycle_states:
        M = jacobian(L, float(alpha), bool(-alpha * s[0] < -D)) @ M
    return np.sort(np.abs(np.linalg.eigvals(M)))[::-1]


def detect_period(orbit: Orbit, L: int, max_p: int = 4000, tol: float = 1e-6):
    """Kleinste Periode p <= max_p, mit der der letzte Zustand der Bahn wiederkehrt (Abstand höchstens tol mal Größe), sonst None.

    Zustand zur Zeit t: (i_t, u_{t-1}, ..., u_{t-L+1}). Eine Wiederkehr auf 1e-6 ist ein Hinweis, kein Beweis (Float-Bahn)."""
    i, u = orbit.i, orbit.u
    T = len(u)
    if T < max_p + L + 1:
        max_p = T - L - 1
    if max_p < 1:
        return None
    ref = np.array([i[T]] + [u[T - k] for k in range(1, L)])
    scale = max(1.0, float(np.max(np.abs(ref))))
    lo = T - max_p
    diff = np.abs(i[lo:T] - ref[0])
    for j in range(1, L):
        diff = np.maximum(diff, np.abs(u[lo - j:T - j] - ref[j]))
    ok = np.nonzero(diff < tol * scale)[0]
    if ok.size == 0:
        return None
    return int(T - (lo + ok[-1]))
