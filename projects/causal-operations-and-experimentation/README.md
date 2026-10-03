# Causal Operations and Experimentation

A compact Management Science benchmark for moving from predictive associations to intervention effects and decision policies.

Implemented methods:

- randomized A/B-test ATE estimation with standard errors;
- 2x2 difference-in-differences;
- observational causal estimation with augmented inverse-propensity weighting (AIPW);
- logistic propensity estimation and separate outcome regressions;
- a budget-constrained treatment policy using estimated heterogeneous uplift scores.

The synthetic observational generator deliberately makes treatment assignment depend on covariates, so a naive treated-vs-control difference is confounded. The AIPW estimator combines a propensity model with outcome models and is evaluated against a known data-generating effect.

Run tests:

```bash
python -m pip install numpy pytest
pytest -q
```

This is an educational reference implementation, not a substitute for design-based identification, overlap diagnostics, sensitivity analysis, cluster-robust inference, or domain-specific experimental governance.
