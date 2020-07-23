import pytest
import os
import math
import pandas as pd
import numpy as np
from state_taxdata.prepdata import PrepData


CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))

ADJUSTMENT = {
    "AGI_STUB": [{"value": 6}],
    "MARS1_targ": [{"value": True}],
    "A00200_targ": [{"value": True}],
    "N04470_targ": [{"value": True}],
}


def test_choose_targets(data_with_targs):
    AGI_STUB, targ_list, var_list = data_with_targs.choose_targets()

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

    AGI_STUB, targ_list, var_list = data_with_targs.choose_targets()

    for var in ["N1", "AGI_STUB", "s006", "pid"]:
        assert var in puf_prep.columns
    for var in targ_list:
        assert var in puf_prep.columns
    assert len(puf_prep["pid"]) == len(puf_prep["pid"].unique())
    assert (puf_prep["AGI_STUB"] == AGI_STUB).all()
    assert (puf_prep["N1"] == 1).all()


def test_puf_summary(data_with_targs, puf_prep, puf_sum):
    AGI_STUB, targ_list, var_list = data_with_targs.choose_targets()

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


def test_state_targets_wide(data_with_targs, targets_wide, puf_sum):

    targ_vars = [var + "_targ" for var in data_with_targs.var_list]
    sum_vars = data_with_targs.var_list

    for targ, var in zip(targ_vars, sum_vars):
        targ_total = targets_wide[targ].sum()
        puf_total = puf_sum.loc[var]
        # TODO: why don't these numbers add up exactly?
        assert math.isclose(targ_total, puf_total, rel_tol=0.0001)


def test_state_targets_long(
    data_with_targs, ratios, targets_long, puf_prep, targets_wide
):

    state_list = list(ratios["STATE"].unique())
    AGI_STUB, targ_list, var_list = data_with_targs.choose_targets()

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

    targ_vars = [var + "_targ" for var in data_with_targs.var_list]

    for targ in targ_vars:
        long_filter = targets_long[targets_long["variable"] == targ]
        long_var_sum = long_filter["value"].sum()
        wide_var_sum = targets_wide[targ].sum()
        assert long_var_sum == wide_var_sum

    weights_sum = addup_filter["value"].sum()
    puf_weights_sum = puf_prep["s006"].sum()
    assert weights_sum == puf_weights_sum


def test_initial_weights(iweights, puf_prep):

    pid_unique = len(iweights["pid"].unique())
    states_unique = len(iweights["STATE"].unique())

    # There should be a row for each person-state combination
    assert len(iweights.index) == (pid_unique * states_unique)
    state_share_sum = iweights["st_share"].sum()
    assert math.isclose(pid_unique, state_share_sum, abs_tol=0.0001)
    assert pid_unique == len(puf_prep.index)

    state_weights_sum = iweights["iweight_state"].sum()
    s006_sum = puf_prep["s006"].sum()
    assert math.isclose(state_weights_sum, s006_sum, abs_tol=0.0001)


def test_cc_dense(dense, ratios, puf_prep, puf_sum, data_with_targs):
    # Check states
    cc_states = dense["STATE"].unique()
    ratios_states = ratios["STATE"].unique()
    assert len(cc_states) == len(ratios_states)

    # Number of people in cc_dense should be same as the filtered puf
    cc_pid = dense["pid"].unique()
    puf_pid = puf_prep["pid"].unique()
    assert len(cc_pid) == len(puf_pid)

    # One row for each person-state combination
    assert len(dense.index) == (len(cc_states) * len(cc_pid))

    # Check sums for each target var
    for targ in data_with_targs.targ_list:
        targ_var = targ + "_targ"
        cc_sum = dense[targ_var].sum()
        puf_sum_val = puf_sum.loc[targ]
        assert math.isclose(cc_sum, puf_sum_val, abs_tol=0.001)

    cc_weight_sum = dense["iweight_state"].sum()
    puf_sum_n1 = puf_sum.loc["N1"]
    assert math.isclose(cc_weight_sum, puf_sum_n1, abs_tol=0.001)


def test_cc_sparse(sparse, dense, data_with_targs, iweights):
    # Separate aggregate constraints from addup
    sparse_agg = sparse[sparse["targtype"] == "aggregate"]
    sparse_addup = sparse[sparse["targtype"] == "addup"]

    targs = [var + "_targ" for var in data_with_targs.targ_list]

    # Length of cc_sparse should be the number of nonzero values
    # in cc_dense
    dense_vars = dense[targs].copy()
    dense_nonzero = np.count_nonzero(dense_vars)
    assert len(sparse_agg.index) == dense_nonzero

    # The sums of cc_sparse and cc_dense should be the same
    dense_total_sum = dense_vars.sum().sum()
    sparse_sum = sparse_agg["nzcc"].sum()
    assert math.isclose(dense_total_sum, sparse_sum, abs_tol=0.001)

    # Addup nzcc should be initial state weights
    iweights_total = iweights["iweight_state"].sum()
    addup_total = sparse_addup["nzcc"].sum()
    assert math.isclose(iweights_total, addup_total, abs_tol=0.001)


def test_compare_national_puf_ht2(
    compare_national, puf_sum, data_with_targs, targets_wide
):
    for var in data_with_targs.var_list:
        compare_puf_val = compare_national.loc[var, "PUF_val"]
        sum_puf_val = puf_sum.loc[var] / 1e6
        assert math.isclose(compare_puf_val, sum_puf_val, abs_tol=0.001)

        # The way to test the HT_2 values would be the same code as in prepdata.py
