"""Run the whole study and write every table it produces."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hierforecast import hierarchy as H          # noqa: E402
from hierforecast import intermittency as IM     # noqa: E402
from hierforecast import pipeline as P           # noqa: E402
from hierforecast import reconcile as R          # noqa: E402
from hierforecast.config import load_config, repo_path  # noqa: E402
from hierforecast.data import load_metadata_from_ids, load_panel  # noqa: E402

TABLES = repo_path("outputs/tables")


def main() -> None:
    started = time.perf_counter()
    cfg = load_config()
    hzn, season = cfg.horizon, cfg.seasonal_period

    train = load_panel(cfg, "train")
    test = load_panel(cfg, "test")
    meta = load_metadata_from_ids(train["unique_id"])
    h = H.build(meta, cfg.levels)

    B, dates, live = H.bottom_matrix(train, h.bottom_ids)
    B_test, dates_test, _ = H.bottom_matrix(test, h.bottom_ids)
    Y, Y_test = H.aggregate(h, B), H.aggregate(h, B_test)

    H.check_coherence(h, B, meta, train).to_csv(TABLES / "coherence_check.csv", index=False)

    prof = IM.profile(B, live, h.bottom_ids,
                      cfg["intermittency"]["adi_threshold"],
                      cfg["intermittency"]["cv2_threshold"])
    prof.to_parquet(TABLES / "intermittency_profile.parquet")

    validation = P.Origin("validation", Y[:, :-hzn], dates[:-hzn], Y[:, -hzn:])
    test_origin = P.Origin("test", Y, dates, Y_test)
    for origin in (validation, test_origin):
        P.make_base_forecasts(origin, h, cfg)

    pd.DataFrame([{"origin": o.name, "method": m, "seconds": s}
                  for o in (validation, test_origin) for m, s in o.timings.items()]
                 ).to_csv(TABLES / "timings.csv", index=False)

    incoh, results, coherence_rows = [], [], []
    for method in P.BASE_METHODS:
        yhat = test_origin.forecasts[method]
        implied = h.S @ yhat[h.rows_for("bottom")]
        viol = np.abs(implied - yhat)
        root = h.rows_for("total")[0]
        incoh.append({"method": method,
                      "max_abs_violation": float(viol.max()),
                      "mean_abs_violation": float(viol.mean()),
                      "root_forecast": float(yhat[root].sum()),
                      "sum_of_bottom": float(implied[root].sum()),
                      "gap_pct_of_root": float(100 * (yhat[root].sum() - implied[root].sum())
                                               / yhat[root].sum()),
                      "already_coherent": bool(viol.max() < 1e-6)})

        rv = P.residual_variance(validation, method)
        for recon, pred in P.reconcile_all(h, meta, yhat, validation.Y_fit, rv).items():
            ok, err = R.is_coherent(h, pred)
            coherence_rows.append({"base": method, "reconciliation": recon,
                                   "coherent": ok, "max_violation": err})
            scored = P.score(h, Y_test, pred, Y, season)
            scored["base"], scored["reconciliation"] = method, recon
            results.append(scored)

    pd.DataFrame(incoh).to_csv(TABLES / "incoherence.csv", index=False)
    pd.DataFrame(coherence_rows).to_csv(TABLES / "reconciled_coherence.csv", index=False)

    scored = pd.concat(results, ignore_index=True)
    scored.to_parquet(TABLES / "scores_by_node.parquet")

    summary = (scored.groupby(["base", "reconciliation", "level"], observed=True)
               .agg(nodes=("rmsse", "size"), rmsse=("rmsse", "mean"),
                    mase=("mase", "mean"), rmsse_sd=("rmsse", "std")).reset_index())
    summary["rmsse_se"] = summary["rmsse_sd"] / np.sqrt(summary["nodes"])
    summary.drop(columns="rmsse_sd").to_csv(TABLES / "accuracy_by_level.csv", index=False)

    bottom = scored[scored.level == "bottom"].merge(
        prof[["unique_id", "pattern"]], left_on="node_id", right_on="unique_id")
    slice_ = (bottom.groupby(["base", "reconciliation", "pattern"], observed=True)
              .agg(series=("rmsse", "size"), rmsse=("rmsse", "mean"),
                   rmsse_sd=("rmsse", "std")).reset_index())
    slice_["rmsse_se"] = slice_["rmsse_sd"] / np.sqrt(slice_["series"])
    slice_.drop(columns="rmsse_sd").to_csv(TABLES / "accuracy_by_pattern.csv", index=False)

    elapsed = time.perf_counter() - started
    pd.DataFrame([{"seed": cfg.seed, "runtime_seconds": round(elapsed, 1),
                   "nodes": int(h.S.shape[0]), "bottom_series": h.n_bottom,
                   "horizon": hzn, "levels": ",".join(cfg.levels)}]
                 ).to_csv(TABLES / "run_metadata.csv", index=False)
    print(f"done in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
