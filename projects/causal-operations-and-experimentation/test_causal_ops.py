import numpy as np
from causal_ops import randomized_ate, difference_in_differences, aipw_ate, budgeted_policy, simulate_observational_data


def test_randomized_ate_recovers_effect():
    rng = np.random.default_rng(1)
    t = rng.binomial(1, 0.5, 5000)
    y = 3.0 * t + rng.normal(size=5000)
    assert abs(randomized_ate(y, t).estimate - 3.0) < 0.1


def test_did_recovers_effect():
    rng = np.random.default_rng(2)
    n = 4000
    g = np.repeat([0, 1], n // 2)
    p = np.tile(np.repeat([0, 1], n // 4), 2)
    y = 10 + 1.5*g + 2.0*p + 4.0*(g*p) + rng.normal(scale=.5, size=n)
    assert abs(difference_in_differences(y, g, p).estimate - 4.0) < 0.1


def test_aipw_handles_observational_assignment():
    x, t, y = simulate_observational_data(n=8000, seed=3, true_ate=2.0)
    assert abs(aipw_ate(x, t, y).estimate - 2.0) < 0.12


def test_budgeted_policy_obeys_budget_and_skips_negative_uplift():
    uplift = np.array([5., 4., -1., 3.])
    cost = np.array([5., 2., 1., 2.])
    chosen = budgeted_policy(uplift, cost, budget=4.)
    assert np.dot(chosen, cost) <= 4.0
    assert chosen[2] == 0
    assert chosen.tolist() == [0, 1, 0, 1]
