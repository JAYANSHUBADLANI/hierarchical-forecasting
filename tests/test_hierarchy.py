import numpy as np
import pandas as pd

from hierforecast import hierarchy as H
from hierforecast.data import load_metadata_from_ids

from conftest import LEVELS


def _meta(ids):
    return load_metadata_from_ids(pd.Series(ids))


def test_ids_parse_into_the_right_parents(toy_ids):
    m = _meta(toy_ids).set_index("unique_id")
    assert m.loc["FOODS_1_001_CA_1", "cat_id"] == "FOODS"
    assert m.loc["FOODS_1_001_CA_1", "dept_id"] == "FOODS_1"
    assert m.loc["FOODS_1_001_CA_1", "item_id"] == "FOODS_1_001"
    assert m.loc["FOODS_1_001_CA_1", "state_id"] == "CA"
    assert m.loc["FOODS_1_001_CA_1", "store_id"] == "CA_1"
    assert m.loc["FOODS_2_002_CA_1", "dept_id"] == "FOODS_2"


def test_summing_matrix_has_one_entry_per_series_per_level(toy_ids):
    h = H.build(_meta(toy_ids), LEVELS)
    assert h.S.shape[1] == 4
    assert h.S.nnz == 4 * len(LEVELS)
    assert set(np.unique(h.S.data)) == {1.0}
    # the total row selects every bottom series exactly once
    total_row = h.rows_for("total")[0]
    assert h.S[total_row].toarray().sum() == 4


def test_aggregates_match_hand_computed_values(toy_ids, toy_panel):
    h = H.build(_meta(toy_ids), LEVELS)
    B, dates, live = H.bottom_matrix(toy_panel, h.bottom_ids)
    Y = H.aggregate(h, B)

    # by hand, day by day
    assert list(Y[h.rows_for("total")[0]]) == [7.0, 6.0, 9.0, 6.0]

    states = dict(zip(h.node_names("state"), Y[h.rows_for("state")]))
    assert list(states["CA"]) == [1.0, 0.0, 7.0, 4.0]
    assert list(states["TX"]) == [6.0, 6.0, 2.0, 2.0]

    stores = dict(zip(h.node_names("store"), Y[h.rows_for("store")]))
    assert list(stores["CA_1"]) == [1.0, 0.0, 7.0, 4.0]
    assert list(stores["TX_2"]) == [2.0, 2.0, 2.0, 2.0]

    depts = dict(zip(h.node_names("store_dept"), Y[h.rows_for("store_dept")]))
    assert list(depts["CA_1|FOODS_1"]) == [1.0, 0.0, 2.0, 3.0]
    assert list(depts["CA_1|FOODS_2"]) == [0.0, 0.0, 5.0, 1.0]


def test_every_level_sums_to_the_same_total(toy_ids, toy_panel):
    h = H.build(_meta(toy_ids), LEVELS)
    B, _, _ = H.bottom_matrix(toy_panel, h.bottom_ids)
    chk = H.check_coherence(h, B, _meta(toy_ids), toy_panel)
    assert chk["exact"].all()
    assert chk["all_levels_same_total"].all()
    assert chk["grand_total"].nunique() == 1


def test_a_broken_aggregate_is_caught(toy_ids, toy_panel):
    """If S is corrupted the check must fail, otherwise it proves nothing."""
    h = H.build(_meta(toy_ids), LEVELS)
    B, _, _ = H.bottom_matrix(toy_panel, h.bottom_ids)
    S = h.S.tolil()
    S[h.rows_for("total")[0], 0] = 0.0        # drop one series from the total
    broken = H.Hierarchy(h.levels, h.nodes, S.tocsr(), h.bottom_ids)
    chk = H.check_coherence(broken, B, _meta(toy_ids), toy_panel)
    assert not chk["exact"].all()


def test_missing_days_are_padded_and_masked(toy_ids, toy_panel):
    short = toy_panel[~((toy_panel.unique_id == "FOODS_1_001_CA_1")
                        & (toy_panel.ds == toy_panel.ds.min()))]
    h = H.build(_meta(toy_ids), LEVELS)
    B, dates, live = H.bottom_matrix(short, h.bottom_ids)
    i = h.bottom_ids.index("FOODS_1_001_CA_1")
    assert B[i, 0] == 0.0          # padded so the tree still adds up
    assert not live[i, 0]          # but not counted as an observed zero
    assert live[i, 1:].all()
    assert H.first_live_index(live)[i] == 1
