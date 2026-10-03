from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CausalEstimate:
    estimate: float
    standard_error: float


def _add_intercept(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    return np.column_stack([np.ones(len(x)), x])


def randomized_ate(y: np.ndarray, treatment: np.ndarray) -> CausalEstimate:
    y = np.asarray(y, dtype=float)
    t = np.asarray(treatment, dtype=int)
    yt, yc = y[t == 1], y[t == 0]
    if len(yt) < 2 or len(yc) < 2:
        raise ValueError("Both treatment arms need at least two observations")
    est = float(yt.mean() - yc.mean())
    se = float(np.sqrt(yt.var(ddof=1) / len(yt) + yc.var(ddof=1) / len(yc)))
    return CausalEstimate(est, se)


def difference_in_differences(y: np.ndarray, treated_group: np.ndarray, post: np.ndarray) -> CausalEstimate:
    y = np.asarray(y, dtype=float)
    g = np.asarray(treated_group, dtype=int)
    p = np.asarray(post, dtype=int)
    cells = {}
    for gv in (0, 1):
        for pv in (0, 1):
            vals = y[(g == gv) & (p == pv)]
            if len(vals) < 2:
                raise ValueError("Each group x period cell needs at least two observations")
            cells[(gv, pv)] = vals
    est = (cells[(1, 1)].mean() - cells[(1, 0)].mean()) - (cells[(0, 1)].mean() - cells[(0, 0)].mean())
    var = sum(v.var(ddof=1) / len(v) for v in cells.values())
    return CausalEstimate(float(est), float(np.sqrt(var)))


def _logistic_fit(x: np.ndarray, t: np.ndarray, ridge: float = 1e-6, max_iter: int = 100) -> np.ndarray:
    x = _add_intercept(x)
    t = np.asarray(t, dtype=float)
    beta = np.zeros(x.shape[1])
    eye = np.eye(x.shape[1]); eye[0, 0] = 0.0
    for _ in range(max_iter):
        z = np.clip(x @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        w = np.clip(p * (1 - p), 1e-8, None)
        grad = x.T @ (t - p) - ridge * (eye @ beta)
        hess = x.T @ (w[:, None] * x) + ridge * eye
        step = np.linalg.solve(hess, grad)
        beta_new = beta + step
        if np.max(np.abs(step)) < 1e-9:
            beta = beta_new
            break
        beta = beta_new
    return beta


def _predict_logistic(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    z = np.clip(_add_intercept(x) @ beta, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def _linear_fit(x: np.ndarray, y: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    x = _add_intercept(x)
    eye = np.eye(x.shape[1]); eye[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + ridge * eye, x.T @ np.asarray(y, dtype=float))


def aipw_ate(x: np.ndarray, treatment: np.ndarray, y: np.ndarray, clip: float = 0.02) -> CausalEstimate:
    """Augmented inverse-propensity-weighted ATE with logistic propensity and linear outcome models."""
    x = np.asarray(x, dtype=float)
    t = np.asarray(treatment, dtype=float)
    y = np.asarray(y, dtype=float)
    if not (len(x) == len(t) == len(y)):
        raise ValueError("x, treatment and y must have the same length")
    pb = _logistic_fit(x, t)
    e = np.clip(_predict_logistic(x, pb), clip, 1 - clip)
    b1 = _linear_fit(x[t == 1], y[t == 1])
    b0 = _linear_fit(x[t == 0], y[t == 0])
    xa = _add_intercept(x)
    m1, m0 = xa @ b1, xa @ b0
    psi = (m1 - m0) + t * (y - m1) / e - (1 - t) * (y - m0) / (1 - e)
    est = float(np.mean(psi))
    se = float(np.std(psi, ddof=1) / np.sqrt(len(psi)))
    return CausalEstimate(est, se)


def budgeted_policy(uplift: np.ndarray, cost: np.ndarray, budget: float) -> np.ndarray:
    """Greedy policy for equal-value treatment units ranked by uplift per unit cost."""
    uplift = np.asarray(uplift, dtype=float)
    cost = np.asarray(cost, dtype=float)
    if np.any(cost <= 0) or budget < 0:
        raise ValueError("Costs must be positive and budget non-negative")
    order = np.argsort(-(uplift / cost))
    chosen = np.zeros(len(uplift), dtype=int)
    spent = 0.0
    for i in order:
        if uplift[i] <= 0:
            continue
        if spent + cost[i] <= budget + 1e-12:
            chosen[i] = 1
            spent += cost[i]
    return chosen


def simulate_observational_data(n: int = 5000, seed: int = 2026, true_ate: float = 2.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    logits = -0.2 + 0.9 * x[:, 0] - 0.7 * x[:, 1]
    propensity = 1 / (1 + np.exp(-logits))
    t = rng.binomial(1, propensity)
    baseline = 5 + 1.5 * x[:, 0] - 1.0 * x[:, 1]
    y = baseline + true_ate * t + rng.normal(scale=1.0, size=n)
    return x, t, y
