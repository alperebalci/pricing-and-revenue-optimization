from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class MNLModel:
    intercepts: np.ndarray
    price_coefficient: float

    def shares(self, prices: np.ndarray) -> np.ndarray:
        p = np.asarray(prices, dtype=float)
        u = self.intercepts + self.price_coefficient * p
        m = max(0.0, float(np.max(u)))
        expu = np.exp(u - m)
        outside = np.exp(-m)
        return expu / (outside + expu.sum())

    def elasticities(self, prices: np.ndarray) -> np.ndarray:
        prices = np.asarray(prices, dtype=float)
        s = self.shares(prices)
        e = np.empty((len(s), len(s)))
        for i in range(len(s)):
            for j in range(len(s)):
                e[i, j] = self.price_coefficient * prices[j] * ((1.0 if i == j else 0.0) - s[j])
        return e


def simulate_mnl(n: int = 15000, intercepts=(1.5, 1.1, 0.8), price_coefficient: float = -0.7, seed: int = 2026):
    rng = np.random.default_rng(seed)
    alpha = np.asarray(intercepts, dtype=float)
    j = len(alpha)
    prices = rng.uniform(1.0, 4.0, size=(n, j))
    u = alpha[None, :] + price_coefficient * prices
    utilities = np.column_stack([np.zeros(n), u])
    utilities += rng.gumbel(size=utilities.shape)
    choices = np.argmax(utilities, axis=1)
    return prices, choices


def fit_mnl(prices: np.ndarray, choices: np.ndarray) -> MNLModel:
    prices = np.asarray(prices, dtype=float)
    choices = np.asarray(choices, dtype=int)
    n, j = prices.shape
    if choices.shape != (n,) or np.any((choices < 0) | (choices > j)):
        raise ValueError("choices must be 0..J with 0 representing the outside option")

    def objective(theta):
        alpha = theta[:j]
        beta = -np.exp(theta[j])
        u = alpha[None, :] + beta * prices
        m = np.maximum(0.0, u.max(axis=1))
        denom = np.exp(-m) + np.exp(u - m[:, None]).sum(axis=1)
        log_denom = np.log(denom) + m
        chosen_u = np.zeros(n)
        mask = choices > 0
        idx = choices[mask] - 1
        chosen_u[mask] = u[np.where(mask)[0], idx]
        return float(np.sum(log_denom - chosen_u))

    x0 = np.r_[np.zeros(j), np.log(0.5)]
    res = minimize(objective, x0, method="BFGS", options={"gtol": 1e-7, "maxiter": 1000})
    if not res.success and res.fun > objective(x0) - 1e-6:
        raise RuntimeError(f"MNL estimation failed: {res.message}")
    return MNLModel(res.x[:j], -float(np.exp(res.x[j])))


def expected_profit(model: MNLModel, prices: np.ndarray, unit_costs: np.ndarray, market_size: float = 1.0) -> float:
    p = np.asarray(prices, dtype=float)
    c = np.asarray(unit_costs, dtype=float)
    if p.shape != c.shape or np.any(p < c):
        raise ValueError("prices and costs must align and prices must not be below costs")
    return float(market_size * np.dot(model.shares(p), p - c))


def optimize_prices(model: MNLModel, unit_costs: np.ndarray, lower_markup: float = 0.01, upper_price: float = 10.0):
    costs = np.asarray(unit_costs, dtype=float)
    bounds = [(float(c + lower_markup), float(upper_price)) for c in costs]
    x0 = np.minimum(costs + 1.0, upper_price)
    res = minimize(lambda p: -expected_profit(model, p, costs), x0, method="L-BFGS-B", bounds=bounds)
    if not res.success:
        raise RuntimeError(f"price optimization failed: {res.message}")
    return res.x, -float(res.fun)
