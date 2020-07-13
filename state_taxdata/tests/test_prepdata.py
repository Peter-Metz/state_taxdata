import pytest
import os
import math
import pandas as pd
import numpy as np
from prepdata import PrepData


CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))

ADJUSTMENT = {
    "AGI_STUB": [{"value": 6}],
    "MARS1_targ": [{"value": True}],
    "A00200_targ": [{"value": True}],
    "N04470_targ": [{"value": True}],
}


def test_choose_targets(data_with_targs):
    AGI_STUB, targ_list, var_list = data_with_targs.choose_targets(ADJUSTMENT)

    assert isinstance(AGI_STUB, np.int64)
    assert isinstance(targ_list, list)
    assert isinstance(var_list, list)

    if "AGI_STUB" in ADJUSTMENT.keys():
        assert len(ADJUSTMENT.keys()) == len(targ_list) + 1
    else:
        assert len(ADJUSTMENT.keys()) == len(targ_list)

    if "N1_targ" in ADJUSTMENT.keys():
        assert len(var_list) == len(targ_list)
    else:
        assert len(var_list) == len(targ_list) + 1


def test_prepare_puf(data_with_targs, puf_prep):

    AGI_STUB, targ_list, var_list = data_with_targs.choose_targets(ADJUSTMENT)

    for var in ["N1", "AGI_STUB", "s006", "pid"]:
        assert var in puf_prep.columns
    for var in targ_list:
        assert var in puf_prep.columns
    assert len(puf_prep["pid"]) == len(puf_prep["pid"].unique())
    assert (puf_prep["AGI_STUB"] == AGI_STUB).all()
    assert (puf_prep["N1"] == 1).all()


def test_puf_summary(data_with_targs, puf_prep, puf_sum):
    AGI_STUB, targ_list, var_list = data_with_targs.choose_targets(ADJUSTMENT)

    assert len(puf_sum.index) == len(var_list)
    n1_weighted = (puf_prep["N1"] * puf_prep["s006"]).sum()
    assert round(puf_sum.loc["N1"], 2) == round(n1_weighted, 2)


def test_calc_ratios(ratios):

    n1_ratio_sums = ratios.groupby(["AGI_STUB"]).sum()["N1"]
    assert (round(n1_ratio_sums, 4) == 1).all()

    ht2_path = os.path.join(CURRENT_PATH, "../data/17in54cmcsv.csv")
    ht2 = pd.read_csv(ht2_path)
    state_list = list(ht2["STATE"].unique())
    state_list.remove("US")

    ratio_state = list(ratios["STATE"].unique())
    assert state_list == ratio_state


def test_state_targets_wide(targets_wide, puf_sum):

    targ_vars = ["MARS1_targ", "A00200_targ", "N04470_targ", "N1_targ"]
    sum_vars = ["MARS1", "A00200", "N04470", "N1"]

    for targ, var in zip(targ_vars, sum_vars):
        targ_total = targets_wide[targ].sum()
        puf_total = puf_sum.loc[var]
        # TODO: why don't these numbers add up exactly?
        assert math.isclose(targ_total, puf_total, rel_tol=0.0001)


def test_state_targets_long(
    data_with_targs, ratios, targets_long, puf_prep, targets_wide
):

    state_list = list(ratios["STATE"].unique())
    AGI_STUB, targ_list, var_list = data_with_targs.choose_targets(ADJUSTMENT)

    # Note that even if N1 is not a target, we need to include it in
    # this count because N1 is used for initial weight calculations
    exp_num_agg_rows = len(state_list) * len(var_list)
    agg_filter = targets_long[targets_long["targtype"] == "aggregate"]
    act_num_agg_rows = len(agg_filter)
    agg_cname_unique = len(agg_filter["cname"].unique())
    assert exp_num_agg_rows == act_num_agg_rows == agg_cname_unique

    exp_num_addup_rows = len(puf_prep.index)
    addup_filter = targets_long[targets_long["targtype"] == "addup"]
    act_num_addup_rows = len(addup_filter)
    addup_cname_unique = len(addup_filter["cname"].unique())
    assert exp_num_addup_rows == act_num_addup_rows == addup_cname_unique

    targ_vars = ["MARS1_targ", "A00200_targ", "N04470_targ", "N1_targ"]

    for targ in targ_vars:
        long_filter = targets_long[targets_long["variable"] == targ]
        long_var_sum = long_filter["value"].sum()
        wide_var_sum = targets_wide[targ].sum()
        assert long_var_sum == wide_var_sum

    weights_sum = addup_filter["value"].sum()
    puf_weights_sum = puf_prep["s006"].sum()
    assert weights_sum == puf_weights_sum


def test_initial_weights(iweights, puf_prep):

    pid_unique = len(iweights['pid'].unique())
    states_unique = len(iweights['STATE'].unique())
    # There should be a row for each person-state combination
    assert len(iweights.index) == (pid_unique * states_unique)
    state_share_sum = iweights['st_share'].sum()
    assert math.isclose(pid_unique, state_share_sum, abs_tol=0.0001)
    assert pid_unique == len(puf_prep.index)
    state_weights_sum = iweights['iweight_state'].sum()
    s006_sum = puf_prep['s006'].sum()
    assert math.isclose(state_weights_sum, s006_sum, abs_tol=0.0001)




