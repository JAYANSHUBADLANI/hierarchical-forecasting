# Which level to forecast, and what to do about it

A short guide to the results, for someone who has to produce one plan.

## The short answer

Forecast at the bottom with a method built for intermittent demand, forecast the
aggregates with a pooled model, and reconcile the pooled model against the tree
using variance weights. On this data that combination was the most accurate at
every level from the total down to store by category, and within a few percent of
the best at the two lowest levels.

If you only want one rule: **do not read a hierarchy of forecasts that has not
been reconciled.** The pooled model's own top level number was 16 percent below
the sum of its own bottom level numbers.

## What each choice costs

**Bottom up** is free, needs no estimation, and is a reasonable default. It was
within a few percent of the best method at four of the six levels. Its weakness
is that it inherits everything wrong with the bottom level forecasts, and at the
bottom the data is 59 percent zeros.

**Top down** was the worst option at almost every level here. It throws away every
forecast except the root and then splits it by historical shares, which assumes
the shares are stable. Over a 28 day horizon on 30,490 items they are not.

**Variance weighted projection** was the winner at the top four levels and needs
one thing you may not have: an honest estimate of forecast error variance per
node. I get it from a held out window rather than from in sample residuals. If
you cannot hold out a window, use the structural weighting instead, which needs
no residuals at all and was close behind.

**Ordinary least squares** should not be used. It weights an error on the grand
total the same as an error on one item in one store, and it more than doubled the
error at department level. If a reconciliation library offers it as the default,
change the default.

## When reconciliation will not help you

If your forecasting method is linear and applied identically to every series, its
forecasts are already coherent and reconciliation will return them unchanged. A
seasonal naive baseline is the clearest case. This is worth checking before
building any of this machinery: forecast, aggregate, compare, and if the gap is
zero you have nothing to fix.

The gain from reconciliation tracked how bad the base forecast was at that level,
not how intermittent the series were. So the question to ask is not "are my series
sparse" but "which level am I currently worst at", and the answer to that is where
reconciliation will pay.

## What this rests on

One retailer, 28 days, one cut of the hierarchy, point forecasts only, and a
diagonal covariance rather than a full one. The direction of these results should
travel; the exact percentages should not be quoted anywhere else.
