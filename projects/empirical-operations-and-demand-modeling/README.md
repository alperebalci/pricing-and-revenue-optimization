# Empirical Operations and Demand Modeling

A structural demand-estimation layer for pricing and assortment decisions.

The project implements a multinomial-logit demand model with an outside option using synthetic choice occasions with experimentally varying prices. It includes:

- maximum-likelihood estimation of product intercepts and a common price coefficient;
- an explicit negative price-sensitivity parameterization;
- market shares with an outside option;
- own- and cross-price elasticities;
- expected-margin evaluation;
- constrained price optimization using the fitted demand system.

The key separation is empirical: demand is estimated from choice data before it is passed to a prescriptive pricing decision. This avoids treating arbitrary predictive scores as economically interpretable demand curves.

Run:

```bash
python -m pip install -r requirements.txt
pytest -q
```

Limitations: the benchmark assumes homogeneous MNL preferences, exogenous price variation, no inventory constraints, and no endogeneity correction. Mixed logit, IV/control-function estimation, panel heterogeneity, substitution under stockouts, and causal price experiments are natural extensions.
