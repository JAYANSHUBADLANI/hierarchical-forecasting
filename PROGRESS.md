# Progress log

A running note to myself on where this stands, what I decided and why, and what
is still open.

## Where it stands

All four phases are built and run end to end from `make demo` in about ninety seconds.
35 tests pass. Two consecutive runs give byte identical result tables.

**Phase 1, hierarchy and data.** The mirror I use ships long format parquet with
only `unique_id`, `ds` and `y`, so the hierarchy had to be parsed back out of the
id string. `FOODS_1_001_CA_1` gives category, department, item, state and store.
Six levels, 30,604 nodes over 30,490 bottom series, and the counts come out
exactly as the config asserts.

The coherence check computes every aggregate twice, through the sparse summing
matrix and through a groupby, and requires them to agree exactly. They do, at
66,927,173 units on every level. There is a test that corrupts the summing matrix
to confirm the check fails when it should, otherwise the check proves nothing.

**The data property that took the longest to notice.** Series start on 1,652
different dates and only 23 percent span the full 1,941 days. The tell is that
not one series has a zero in its first row, so those leading rows were stripped
rather than the item having sold nothing. Padding them with zero is necessary for
the tree to add up and wrong for anything measured per series: it moves 1,286
series into a more intermittent class than they belong in and inflates median ADI
from 2.72 to 3.75. So the bottom matrix now comes with a live mask and every per
series statistic uses it.

**Phase 2, base forecasts.** Seasonal naive, simple exponential smoothing with a
per series alpha, Croston with the SBA correction, and a pooled gradient boosting
model. All vectorised across nodes, with the recursive ones looping over time and
updating every series at once, so the slowest of them takes 8 seconds on all
30,604 nodes rather than the hours a per series loop would have cost.

Croston needed its interval level initialised from the first observed interval
rather than from 1. Starting at 1 biases the rate upward for as long as the
smoother takes to climb, which at alpha 0.1 is most of a short series. A test
pins the rate on a series with a known constant interval.

**Phase 3, reconciliation.** Bottom up, top down on historical proportions,
middle out at store level, and OLS, structural and variance weighted projections.

The projections cannot be done by forming the normal matrix, which is 30,490 by
30,490 and dense because every pair of bottom series shares the root. They are
solved by conjugate gradient against an operator that only applies the sparse
summing matrix. OLS converged immediately. The variance weighted version did not
converge in 2,000 iterations, because variance weights span many orders of
magnitude across levels. A Jacobi preconditioner took it to 5 iterations, and it
is free here because every entry of S is one so the normal matrix diagonal is
just `S' w_inv`.

All 24 reconciled combinations come out exactly coherent. The three that do not
are the unreconciled base rows, which is correct.

**Phase 4, results.** The hypothesis I started with was half wrong and the wrong
half is the better finding: reconciliation helped most where the base forecast was
worst, which happened to be the intermittent series, not least. What predicts the
gain is base forecast quality at that level, not intermittency.

## Decisions worth remembering

- **Hierarchy cut.** M5 supports several. I use a strict tree down location then
  product so every bottom series has exactly one parent per level. A grouped
  structure crossing product with location is defensible too and would change
  every number.
- **Top down disaggregation** is historical proportions of the grand total. That
  choice does real work in the result and is named in the README rather than
  buried.
- **Residual variances** for the variance weighting come from a validation origin
  that holds out the last 28 days of training, not from in sample residuals. The
  pooled model's in sample residuals are optimistic in a way that would quietly
  mis-weight everything.
- **The GBM trains on a recent window and samples the bottom level**, because the
  bottom outnumbers every aggregate by two orders of magnitude and would otherwise
  be all the model learns.

## Open

- MinT with a full shrunk covariance. The covariance is 30,604 squared and the
  sample estimate from 1,941 days has rank at most 1,941, so a Woodbury identity
  on a diagonal plus low rank form would make it tractable. That is the single
  biggest gap against the method as usually written down.
- Prediction intervals. Everything here is a point forecast.
- Cross temporal reconciliation across daily, weekly and monthly views.
- The price file is downloaded but not used as a feature yet.
