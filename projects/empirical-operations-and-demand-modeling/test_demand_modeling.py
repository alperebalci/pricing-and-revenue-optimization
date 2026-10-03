import numpy as np
from demand_modeling import simulate_mnl, fit_mnl, expected_profit, optimize_prices


def test_mnl_recovers_price_sensitivity():
    prices, choices = simulate_mnl(n=25000, price_coefficient=-0.8, seed=11)
    model = fit_mnl(prices, choices)
    assert abs(model.price_coefficient + 0.8) < 0.08


def test_elasticity_signs_are_economically_consistent():
    prices, choices = simulate_mnl(n=12000, seed=12)
    model = fit_mnl(prices, choices)
    e = model.elasticities(np.array([2.0, 2.5, 3.0]))
    assert np.all(np.diag(e) < 0)
    off = e[~np.eye(3, dtype=bool)]
    assert np.all(off > 0)


def test_optimized_prices_improve_profit_over_simple_markup():
    prices, choices = simulate_mnl(n=18000, seed=13)
    model = fit_mnl(prices, choices)
    costs = np.array([1.0, 1.2, 0.8])
    baseline = costs + 1.0
    base_profit = expected_profit(model, baseline, costs)
    pstar, profit = optimize_prices(model, costs, upper_price=8.0)
    assert np.all(pstar >= costs)
    assert profit >= base_profit - 1e-7
