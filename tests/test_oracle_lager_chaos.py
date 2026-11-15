"""Orakel-Tests des Lager-Kerns: jede Kernroutine gegen einen anderen Rechenweg (ORACLE-PLAYBOOK §2, §5 und §6).

Orakel:
 1. Benettin (Tangentialvektor) gegen das Produkt der Jacobi-Matrizen mit QR-Zerlegung (numpy.linalg.qr), eigene Schleife in diesem Test, an 20 Punkten.
 2. Benettin gegen Zweibahn (zweites Verfahren der App) an 20 Punkten: gleiches Urteil, Zahl auf 0,002.
 3. Ungeklemmt (D sehr groß) gleicht der Exponent dem Betrag des Spektralradius der Zustandsmatrix (numpy.linalg.eigvals): ln rho.
 4. Das Lager mit L = 2 gegen seine Normalform (x = i - D/alpha, y = w + D) auf 1e-12 über 2000 Schritte.
 5. alpha_krit gegen numpy.linalg.eigvals der Zustandsmatrix für L = 1 bis 12.
 6. Normalform der Randkollision: Exponent gegen QR-Produkt; (19) im Sonderfall delta_L = delta_R gegen die im Aufsatz angegebene verkürzte Form.
 7. Brent-Zyklussuche gegen eine Zählung mit Wörterbuch (kleine Abbildungen und das Lager).
 8. SplitMix64: Referenzwerte, Feld gegen Klasse.
Gegenprobe für Absolutaussagen (ORACLE-PLAYBOOK §6): die Aussage „im Bereich [1,677; 2,397] zeigt L = 3 kein Fenster“ gegen 6000 Läufe mit zufälligen alpha und Anfangszuständen und
doppelter Messdauer (data/lgc_frozen.json, erzeugt von tools/make_frozen.py; hier ein Teil neu gerechnet)."""

import math

import numpy as np
import pytest

import lgc_dynamics as K
import lgc_rng as R


def _state_matrix(L, a):
    """Zustandsmatrix der linearen Regel (ungeklemmt), unabhängig vom Kern aufgeschrieben: x_{t+1} = A x_t."""
    A = np.zeros((L, L))
    if L == 1:
        A[0, 0] = 1.0 - a
        return A
    A[0, 0] = 1.0
    A[0, L - 1] = 1.0
    A[1, 0] = -a
    for k in range(2, L):
        A[k, k - 1] = 1.0
    return A


def _qr_lyapunov(L, alpha, D, i0, burn, rec):
    """Größter Exponent über QR des Matrixprodukts; Bahn und Jacobi-Matrizen in diesem Test selbst gebaut."""
    x = np.zeros(L)
    x[0] = i0
    Q = np.eye(L)
    acc = 0.0
    for t in range(burn + rec):
        clipped = -alpha * x[0] < -D
        J = _state_matrix(L, alpha)
        u = -D if clipped else -alpha * x[0]
        if clipped:
            J[1, 0] = 0.0
        new = np.empty(L)
        new[0] = x[0] + (x[L - 1] if L > 1 else u)
        if L > 1:
            new[1] = u
            new[2:] = x[1:L - 1]
        x = new
        Q, Rm = np.linalg.qr(J @ Q)
        if t >= burn:
            acc += math.log(abs(Rm[0, 0]))
    return acc / rec


POINTS = [(3, a) for a in (0.45, 0.7, 0.9, 1.05, 1.15, 1.4, 1.7, 1.95, 2.15, 2.3)] + [(6, a) for a in (0.31, 0.52, 0.77, 1.11, 1.4, 1.65, 1.89, 2.05, 2.2, 2.35)]


def test_benettin_agrees_with_the_qr_product_at_20_points():
    burn, rec = 2000, 4000
    for L in (3, 6):
        pts = [a for (l, a) in POINTS if l == L]
        lam = K.lyapunov_grid(L, pts, burn=burn, rec=rec).lam
        for a, lb in zip(pts, lam):
            lq = _qr_lyapunov(L, a, 10.0, 0.5, burn, rec)
            assert lb == pytest.approx(lq, abs=0.002), (L, a)
            assert (lb > K.LAMBDA_SCHWELLE) == (lq > K.LAMBDA_SCHWELLE) or abs(lb - K.LAMBDA_SCHWELLE) < 0.001, (L, a)


def test_two_orbit_method_agrees_with_benettin_at_20_points():
    for L in (3, 6):
        pts = [a for (l, a) in POINTS if l == L]
        lam = K.lyapunov_grid(L, pts).lam
        wolf = K.two_orbit_grid(L, pts)
        assert np.max(np.abs(lam - wolf)) < 0.002, L
        assert list(lam > K.LAMBDA_SCHWELLE) == list(wolf > K.LAMBDA_SCHWELLE), L


def test_unclamped_exponent_equals_the_log_spectral_radius():
    """Ohne Klemmung (D = unendlich) ist der Exponent ln des Spektralradius; L = 3, alpha = 0,9: Wachstum je Periode 1,1159 (Eigenwert; die Vormessung simulierte 1,1163)."""
    for L, a in ((3, 0.9), (3, 0.3), (6, 0.5), (2, 1.3), (4, 0.2)):
        rho = float(np.max(np.abs(np.linalg.eigvals(_state_matrix(L, a)))))
        lam = float(K.lyapunov_grid(L, [a], D=math.inf, burn=200, rec=800).lam[0])
        assert lam == pytest.approx(math.log(rho), abs=1e-3), (L, a)
    rho = float(np.max(np.abs(np.linalg.eigvals(_state_matrix(3, 0.9)))))
    assert rho == pytest.approx(1.1159, abs=5e-5)


def test_two_period_store_equals_its_normal_form_over_2000_steps():
    al, D = 1.2, 10.0
    orb = K.simulate(2, al, D, 5.0, 2000)
    nf = K.normal_form_orbit(al, D, 5.0, 0.0, 2000)
    assert np.max(np.abs((orb.i - D / al) - nf[:, 0])) < 1e-12
    assert np.max(np.abs((np.concatenate([[0.0], orb.u]) + D) - nf[:, 1])) < 1e-12


def test_alpha_crit_equals_the_eigenvalue_boundary_for_L_1_to_12():
    for L in range(1, 13):
        ak = K.alpha_crit(L)
        rho = lambda a: float(np.max(np.abs(np.linalg.eigvals(_state_matrix(L, a)))))  # noqa: E731
        assert rho(ak) == pytest.approx(1.0, abs=1e-6), L
        assert rho(ak * 0.98) < 1.0 < rho(ak * 1.02), L
        # kritische Periode 4L - 2 aus dem Winkel des größten Eigenwerts (L >= 2)
        if L >= 2:
            ev = np.linalg.eigvals(_state_matrix(L, ak))
            big = ev[np.argmax(np.abs(ev))]
            assert 2 * math.pi / abs(np.angle(big)) == pytest.approx(4 * L - 2, abs=1e-6), L


def _qr_nf_lyapunov(tl, dl, tr, dr, mu, n, burn):
    x, y = 0.3, 0.1
    Q = np.eye(2)
    acc = 0.0
    for t in range(burn + n):
        if x <= 0:
            A = np.array([[tl, 1.0], [-dl, 0.0]])
        else:
            A = np.array([[tr, 1.0], [-dr, 0.0]])
        x, y = A[0, 0] * x + y + mu, A[1, 0] * x
        Q, Rm = np.linalg.qr(A @ Q)
        if t >= burn:
            acc += math.log(abs(Rm[0, 0]))
    return acc / n


def test_normal_form_exponent_agrees_with_the_qr_product():
    for tl, dl, tr, dr in ((1.7, 0.3, -1.8, 0.05), (1.7, 0.3, -2.2, 0.3), (1.9, 0.5, -1.52, 0.5), (1.7, 0.3, 0.4, 0.05)):
        lam = float(K.normal_form_lyapunov(tl, dl, [tr], dr, n=8000, burn=1000)[0])
        assert lam == pytest.approx(_qr_nf_lyapunov(tl, dl, tr, dr, 1.0, 8000, 1000), abs=2e-3), (tl, dl, tr, dr)


def test_condition_19_reduces_to_the_short_form_of_the_paper_for_equal_determinants():
    """Im Aufsatz: für delta_L = delta_R = delta gilt (19) in der Form tau_R l1 - l1 l2 + tau_L - tau_R - delta > 0 (Eigenwerte der linken Matrix, l1 >= l2).
    Ausklammern von delta aus der vollen Fassung ergibt: volle linke Seite = delta mal verkürzte linke Seite (hier an 200 Zufallspunkten geprüft)."""
    rng = np.random.default_rng(3)
    n_checked = 0
    for _ in range(200):
        tl, d, tr = rng.uniform(1.0, 2.5), rng.uniform(0.05, 0.9), rng.uniform(-3, -1)
        ev = np.roots([1.0, -tl, d])                    # Eigenwerte der linken Matrix aus dem charakteristischen Polynom
        if np.iscomplex(ev).any():
            continue
        l1, l2 = max(ev.real), min(ev.real)
        short = tr * l1 - l1 * l2 + tl - tr - d
        full = K.condition_19_value(tl, d, tr, d)
        assert full == pytest.approx(d * short, abs=1e-9)
        n_checked += 1
    assert n_checked > 100


def test_brent_agrees_with_a_dictionary_count_on_random_maps_and_on_the_store():
    rng = np.random.default_rng(5)
    for _ in range(40):
        n = int(rng.integers(2, 60))
        table = rng.integers(0, n, size=n)
        step = lambda x, t=table: (int(t[x[0]]),)  # noqa: E731
        x0 = (int(rng.integers(0, n)),)
        seen, x, k = {}, x0, 0
        while x not in seen:
            seen[x] = k
            x = step(x)
            k += 1
        assert K.brent_cycle(step, x0, 10 * n) == (k - seen[x], seen[x])
    for L, al, start in ((3, 2.0, 0.5), (3, 2.0, 7.5), (2, 2.0, 0.5)):
        step = K.make_step(L, al, 10.0)
        x0 = K.start_state(L, start)
        seen, x, k = {}, x0, 0
        while x not in seen:
            seen[x] = k
            x = step(x)
            k += 1
        assert K.brent_cycle(step, x0, 100000) == (k - seen[x], seen[x]), (L, al, start)


def test_splitmix64_reference_values_and_array_equals_class():
    g = R.SplitMix64(0)
    assert [g.next_u64() for _ in range(3)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]
    seed = 123456789
    cls = R.SplitMix64(seed)
    arr = R.splitmix_array(seed, 50)
    assert [int(v) for v in arr] == [cls.next_u64() for _ in range(50)]
    cls = R.SplitMix64(seed)
    nrm = R.normal_array(seed, 30)
    assert list(nrm) == pytest.approx([cls.normal() for _ in range(30)], abs=1e-12)


def test_normal_noise_has_the_requested_moments():
    e = K.noise_sequence(7, 60000, 0.25, 10.0)
    assert abs(float(e.mean())) < 0.05 and float(e.std()) == pytest.approx(2.5, rel=0.02)


def test_counter_check_no_window_in_the_L3_chaos_range_with_random_alphas_and_starts():
    """ORACLE-PLAYBOOK §6: 400 zufällige alpha in [1,677; 2,397], je 3 zufällige Anfangszustände in [-50; 50], doppelte Messdauer (kleine Fassung; die
    große mit 1000 alpha und 6 Anfangszuständen steht in data/lgc_frozen.json): in n von n Läufen ist lambda > 0,003."""
    a = 1.677 + (2.397 - 1.677) * R.uniform_array(99, 400)
    i0 = -50.0 + 100.0 * R.uniform_array(100, 1200)
    lam = K.lyapunov_grid(3, np.repeat(a, 3), burn=8000, rec=8000, i0=i0).lam
    assert lam.size == 1200 and int((lam > K.LAMBDA_SCHWELLE).sum()) == 1200
