from taxcalc import *
import pandas as pd
import numpy as np
import os
import paramtools


CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))


class TAXDATA_PARAMS(paramtools.Parameters):
    defaults = os.path.join(CURRENT_PATH, "defaults.json")


class PrepData:
    """
    Contructor for PrepData class.

    The purpose of this class is to construct the input data needed
    by IPOPT to generate state weights. In particular, we need
    to generate target values, intial state weights, and a sparse
    matrix of constraint coefficients.

    In addition, the compare_national_puf_ht2() and
    compare_state_puf_ht2() methods allow for comparison of aggregate
    PUF values and aggregate HT2 values for the chosen target
    variables and AGI group.

    Parameters
    ---------
    adjustment: ParamTools-style dictionary to adjust targets
        and AGI group. The default parameter values for all
        possible targets are False. The data prep routines 
        will only treat each variable as a target when its
        parameter value is True. For example:

        adjustment = {
            "AGI_STUB": [{"value": 6}],
            "MARS1_targ": [{"value": True}],
            "A00200_targ": [{"value": True}],
            "N04470_targ": [{"value": True}],
        }

    puf_df: The purpose of this parameter is to allow the CS
    web app to use the PUF from the OSPC S3 bucket. When calling
    the Python API, it is receommended to keep this parameter
    set to None.

    Returns
    -------
    class instance: PrepData

    """

    POSSIBLE_TARGET_VARS = [
        "e00200",
        "c04470",
        "c17000",
        "c04800",
        "c05800",
        "c09600",
        "e00700",
        "c01000",
        "c00100",
        "mars1",
        "mars2",
    ]

    # AGI groups to target separately
    HT2_AGI_STUBS = [-9e99, 1.0, 10e3, 25e3, 50e3, 75e3, 100e3, 200e3, 500e3, 1e6, 9e99]

    # Maps PUF variable names to HT2 variable names
    VAR_CROSSWALK = {
        "n1": "N1",  # Total population
        "mars1_n": "MARS1",  # Single returns number
        "mars2_n": "MARS2",  # Joint returns number
        "c00100": "A00100",  # AGI amount
        "e00200": "A00200",  # Salary and wage amount
        "e00200_n": "N00200",  # Salary and wage number
        "c01000": "A01000",  # Capital gains amount
        "c01000_n": "N01000",  # Capital gains number
        "c04470": "A04470",  # Itemized deduction amount (0 if standard deduction)
        "c04470_n": "N04470",  # Itemized deduction number (0 if standard deduction)
        "c17000": "A17000",  # Medical expenses deducted amount
        "c17000_n": "N17000",  # Medical expenses deducted number
        "c04800": "A04800",  # Taxable income amount
        "c04800_n": "N04800",  # Taxable income number
        "c05800": "A05800",  # Regular tax before credits amount
        "c05800_n": "N05800",  # Regular tax before credits amount
        "c09600": "A09600",  # AMT amount
        "c09600_n": "N09600",  # AMT number
        "e00700": "A00700",  # SALT amount
        "e00700_n": "N00700",  # SALT number
    }

    def __init__(self, adjustment={}, puf_df=None):

        CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
        HT2_PATH = os.path.join(CURRENT_PATH, "data/17in54cmcsv.csv")

        self.puf_df = puf_df
        self.ht2_path = HT2_PATH
        self.params = TAXDATA_PARAMS()
        self.params.adjust(adjustment)
        self.AGI_STUB, self.targ_list, self.var_list = self.choose_targets()
        self.puf_2017_filter = self.prepare_puf()
        self.puf_sum = self.puf_summary()
        self.ratio_df = self.calc_ratios()
        self.targets_wide = self.state_targets_wide()
        self.targets_long = self.state_targets_long()
        self.iweights = self.initial_weights()
        self.dense = self.cc_dense()

        
    def choose_targets(self):
        """
        Use ParamTools to generate list of user-specified targets and AGI group
        """
        # params = TAXDATA_PARAMS()
        # params.adjust(adjustment)
        AGI_STUB = self.params.to_array("AGI_STUB")

        targ_list = []
        for k, v in self.params.dump().items():
            if v.get("section_1") == "Targets" and v.get("value")[0]["value"] is True:
                targ_list.append(k.split("_")[0])
        var_list = list(targ_list)
        # we need population variable to construct initial state weights
        if "N1" not in var_list:
            var_list.append("N1")
        return AGI_STUB, targ_list, var_list

    def prepare_puf(self):
        """
        Create version of PUF that is extrapolated to 2017, contains necessary
        indicator variables, and is filtered for AGI group and target variables.
        """

        def puf_to_2017():
            """
            Extrapolate PUF to 2017
            """
            pol = Policy()
            if self.puf_df == None:
                recs = Records()
            else:
                assert isinstance(self.puf_df, pd.DataFrame)
                recs = Records(self.puf_df)
            calc = Calculator(pol, recs)
            calc.advance_to_year(2017)
            calc.calc_all()

            puf_cps_2017 = calc.dataframe(variable_list=[], all_vars=True)
            # filter out filers imputed from CPS
            puf_2017 = puf_cps_2017.copy()
            puf_2017 = puf_2017.loc[puf_2017["data_source"] == 1]
            # create unique variable for personal ID
            puf_2017["pid"] = np.arange(len(puf_2017)) + 1
            return puf_2017

        def puf_indicators():
            """
            Create indicator variables (i.e. 1 when a condition is met, 0
            otherwise). These variables are needed for targeting "count"
            aggregates (i.e. number of returns that meet a condition).
            """
            puf_2017 = puf_to_2017()
            for var in self.POSSIBLE_TARGET_VARS:
                count_var = var + "_n"
                # Positive AGI. Note that unlike other variables, we count positive
                # AGI as opposed to non-negative to line up with HT2 variables
                if var == "c00100":
                    puf_2017[count_var] = np.where(puf_2017[var] > 1, 1, 0)
                # Marital status
                elif var == "mars1":
                    puf_2017["mars1_n"] = np.where(puf_2017.MARS == 1, 1, 0)
                elif var == "mars2":
                    puf_2017["mars2_n"] = np.where(puf_2017.MARS == 2, 1, 0)
                # All other possible target variables
                else:
                    puf_2017[count_var] = np.where(puf_2017[var] != 0, 1, 0)

            # Sort PUF filers into AGI bins (same bins as HT2)
            puf_2017["AGI_STUB"] = pd.cut(
                puf_2017["c00100"],
                self.HT2_AGI_STUBS,
                labels=list(range(1, 11)),
                right=False,
            )
            puf_2017["n1"] = 1

            # Rename variables to align with HT2 names
            puf_2017.rename(columns=self.VAR_CROSSWALK, inplace=True)

            return puf_2017

        def filter_puf(self):
            """
            Filter PUF for target variables and AGI group
            """
            puf_2017 = puf_indicators()
            keep_list = list(self.var_list)
            additional_vars = ["AGI_STUB", "s006", "pid"]
            keep_list.extend(additional_vars)

            puf_2017_filter = puf_2017[keep_list]
            puf_2017_filter = puf_2017_filter[
                puf_2017_filter["AGI_STUB"] == self.AGI_STUB
            ]
            return puf_2017_filter

        return filter_puf(self)

    def puf_summary(self):
        """
        Calculate weighted totals for each target variable within the
        given AGI group.
        """
        puf_summary_temp = self.puf_2017_filter.copy()
        puf_summary = pd.DataFrame()

        for var in self.var_list:
            puf_summary[var] = puf_summary_temp[var] * puf_summary_temp["s006"]

        puf_summary = puf_summary.sum()

        return puf_summary

    def calc_ratios(self):
        """
        For each target and AGI group, calculate the ratio of the state
        totals to the national totals using HT2 
        """
        # Read in HT2
        # NOTE: before reading csv, I cleared number formatting in
        # excel to get rid of thousands comma separator
        ht2 = pd.read_csv(self.ht2_path)

        # Filter for US
        ht2_us = ht2[ht2["STATE"] == "US"]

        keep_list = list(self.var_list)
        additional_vars = ["STATE", "AGI_STUB"]
        keep_list.extend(additional_vars)

        states = list(ht2.STATE.unique())
        # remove US from list of states
        states.pop(0)

        ht2_us_vals = ht2_us.drop(["STATE", "AGI_STUB"], axis=1)

        # Loop through each state to construct table of ratios
        ratio_df = pd.DataFrame()
        for state in states:
            state_df = ht2[ht2["STATE"] == state].reset_index()
            state_id = state_df[["STATE", "AGI_STUB"]]
            state_vals = state_df.drop(["index", "STATE", "AGI_STUB"], axis=1)

            # divide each state's total/stub by the U.S. total/stub
            ratios = state_vals / ht2_us_vals
            # tack back on states and stubs
            ratios_state = pd.concat([state_id, ratios], axis=1)
            # add each state ratio df to overall ratio df
            ratio_df = pd.concat([ratio_df, ratios_state])

        ratio_df = ratio_df[keep_list]
        return ratio_df

    def state_targets_wide(self):
        """
        Create a dataframe where each column is a target variable and the values
        are the totals for that target/state/AGI group combination.

        We calculate these state values using the state ratios we calculated from
        HT2 and aggregate values from the PUF.
        """

        # Note that even if N1 is not a target, we need to include it in
        # this dataframe for initial weights to be calculted.
        keep_list_ht2 = list(self.var_list)
        keep_list_puf = list(self.var_list)

        additional_vars_ht2 = ["STATE", "AGI_STUB"]
        additional_vars_puf = ["AGI_STUB", "s006", "pid"]

        keep_list_ht2.extend(additional_vars_ht2)
        keep_list_puf.extend(additional_vars_puf)

        state_targs = pd.DataFrame()

        # filter ratio df by AGI stub
        stub_ratio = self.ratio_df[self.ratio_df["AGI_STUB"] == self.AGI_STUB]
        stub_vals = stub_ratio.drop(["STATE", "AGI_STUB"], axis=1)
        # multiply the ratios for each stub by the PUF totals
        ratio_vals_bin = self.puf_sum * stub_vals
        ratio_vals_bin = pd.concat([stub_ratio.STATE, ratio_vals_bin], axis=1)
        state_targs = pd.concat([state_targs, ratio_vals_bin])

        state_targs = state_targs.reset_index()
        state_targs = state_targs.rename(columns={"index": "AGI_STUB"})

        add_suffix = {}
        for column in state_targs:
            no_change = ["AGI_STUB", "STATE"]
            if column in no_change:
                add_suffix[column] = column
            else:
                add_suffix[column] = column + "_targ"

        targets_wide = state_targs.rename(columns=add_suffix)
        return targets_wide

    def state_targets_long(self):
        """
        Pivot targets to create "long" dataframe and assign constraint
        names ("cname").
        """

        def state_targets_long_agg():
            """
            Create a "long" dataframe of the aggregate targets.
            """
            targets_long_agg = pd.melt(self.targets_wide, id_vars=["AGI_STUB", "STATE"])
            targets_long_agg["cname"] = (
                targets_long_agg["variable"] + "_" + targets_long_agg["STATE"]
            )
            # Create a new variable that distinguishes aggregate targets from
            # add-up targets
            targets_long_agg["targtype"] = "aggregate"
            return targets_long_agg

        def state_targets_long_addup():
            """
            Create a "long" dataframe of the add-up targets (to ensure that
            state weights add up to national weight).
            """
            # The constraint name is a unique personal ID (e.g. "p00000001")
            cname_pid = self.puf_2017_filter["pid"].astype(str)
            cname_pid = "p" + cname_pid.str.zfill(8)
            targets_add_up = pd.DataFrame()
            targets_add_up["pid"] = self.puf_2017_filter["pid"]
            targets_add_up["value"] = self.puf_2017_filter["s006"]
            # Create a new variable that distinguishes aggregate targets from
            # add-up targets
            targets_add_up["targtype"] = "addup"
            targets_add_up["cname"] = cname_pid
            return targets_add_up

        targets_long_agg = state_targets_long_agg()
        targets_add_up = state_targets_long_addup()

        # Combine the aggregate targets and the add-up targets
        targets_long = pd.concat([targets_long_agg, targets_add_up])
        # i is the row number for the constraint (needed for constraint coefficient matrix)
        targets_long["i"] = np.arange(len(targets_long)) + 1
        return targets_long

    def initial_weights(self):
        """
        Calculate the initial weight for each person-state combination.
        """
        states = list(self.targets_wide["STATE"])

        def state_shares():
            """
            Calculate percentage of population coming from each state
            """
            state_shares = pd.DataFrame()
            state_shares["STATE"] = self.targets_wide["STATE"]
            state_shares["AGI_STUB"] = self.targets_wide["AGI_STUB"]
            state_shares["st_share"] = (
                self.targets_wide["N1_targ"] / self.targets_wide["N1_targ"].sum()
            )
            return state_shares

        state_shares = state_shares()
        # Apportion national  weight (s006) to each state using the
        # calculted state population shares
        iweights = pd.DataFrame()
        for state in states:
            state_df = pd.DataFrame()
            state_df["pid"] = self.puf_2017_filter["pid"]
            state_df["weight_total"] = self.puf_2017_filter["s006"]
            state_df["STATE"] = state
            iweights = pd.concat([iweights, state_df])
        iweights = iweights.merge(state_shares, on="STATE")
        iweights = iweights.sort_values(by=["pid", "STATE"])
        iweights["iweight_state"] = iweights["weight_total"] * iweights["st_share"]
        return iweights

    def cc_dense(self):
        """
        Create a dense matrix of constraint coefficients, not including the
        adding-up constraints. The matrix has one record per person per state.
        """
        cc_dense = self.iweights[["pid", "STATE", "iweight_state"]].copy()
        # Use PID to match initial weights with weighted targets
        cc_dense = cc_dense.merge(self.puf_2017_filter, on="pid")
        for var in self.targ_list:
            state_var = var + "_targ"
            cc_dense[state_var] = cc_dense[var] * cc_dense["iweight_state"]
        cc_dense = cc_dense.drop(labels=self.var_list, axis=1)
        cc_dense = cc_dense.drop(labels=["s006", "AGI_STUB"], axis=1)
        # j is an index for x, the variable we will solve for
        cc_dense["j"] = np.arange(len(cc_dense)) + 1
        return cc_dense

    def cc_sparse(self):
        def cc_sparse_agg():
            """
            Sparse matrix of constraint coefficients, not including adding-up constraints
            """
            cc_sparse1 = pd.melt(
                self.dense.drop("iweight_state", axis=1), id_vars=["j", "STATE", "pid"]
            )
            # Nonzero constraint coefficient
            cc_sparse1 = cc_sparse1[cc_sparse1["value"] != 0]
            cc_sparse1["cname"] = cc_sparse1["variable"] + "_" + cc_sparse1["STATE"]
            cc_sparse1 = cc_sparse1.rename({"value": "nzcc"}, axis=1).drop(
                "variable", axis=1
            )
            return cc_sparse1

        cc_sparse1 = cc_sparse_agg()

        def cc_sparse_addup():
            """
            Sparse matrix of constraint coefficients for the adding-up constraints.
            It will have one row per person per state.
            """
            cc_sparse2 = self.dense[["j", "pid", "STATE", "iweight_state"]].copy()
            cname_pid = cc_sparse2["pid"].astype(str)
            cc_sparse2["cname"] = "p" + cname_pid.str.zfill(8)
            cc_sparse2 = cc_sparse2.rename({"iweight_state": "nzcc"}, axis=1)
            return cc_sparse2

        def combine():
            cc_sparse2 = cc_sparse_addup()
            cc_sparse = pd.concat([cc_sparse1, cc_sparse2])
            cc_sparse = cc_sparse.merge(
                self.targets_long[["cname", "i", "targtype"]], on="cname"
            )
            # Ordering is important for the Jacobian
            cc_sparse = cc_sparse.sort_values(by=["i", "j"])
            return cc_sparse

        return combine()

    def get_constraint_bounds(self):
        """
        Calculate upper and lower constraint bounds for each constraint (aggregate
        and addup).

        The tolerances for each aggregate constraint are the same if
        "Aggregate_tol_all" is True. In this case, the tolerance can be set with
        "Aggregate_tol". If "Aggregate_tol_all" is False, the tolerance for each
        constraint can be set individually with the parameters in the form
        "*_tol".

        The tolerance for the addup constraints can be set with "Addup_tol".
        """
        agg_tol_df = self.targets_long[
            ["variable", "targtype", "cname", "value"]
        ].copy()
        agg_tol_df = agg_tol_df[agg_tol_df["targtype"] == "aggregate"]
        agg_tol_df = agg_tol_df.drop("targtype", axis=1)

        # If Aggregate_tol_all is True, tolerances are the same across target
        # variables
        if self.params.to_array("Aggregate_tol_all"):
            agg_tol_df["tol"] = self.params.to_array("Aggregate_tol")
        # If Aggregate_tol_all is False, extract individual target tolerances
        # from respective parameters and merge onto target dataframe.
        else:
            tol_dict = {}
            for var in self.var_list:
                targ_var = var + "_targ"
                tol_var = var + "_tol"
                tol_dict[targ_var] = self.params.to_array(tol_var)
            tol_df = pd.DataFrame.from_dict(tol_dict, orient="index")
            tol_df = tol_df.reset_index().rename(
                {"index": "variable", 0: "tol"}, axis=1
            )
            agg_tol_df = agg_tol_df.merge(tol_df, on="variable")

        # Calculate upper and lower bounds for aggregate targets
        agg_clb = np.where(
            pd.isnull(agg_tol_df["value"]),
            -9e99,
            agg_tol_df["value"] - (abs(agg_tol_df["value"]) * agg_tol_df["tol"]),
        )
        agg_cub = np.where(
            pd.isnull(agg_tol_df["value"]),
            9e99,
            agg_tol_df["value"] + (abs(agg_tol_df["value"]) * agg_tol_df["tol"]),
        )
        agg_tol_df["clb"] = agg_clb
        agg_tol_df["cub"] = agg_cub

        # Construct dataframe for adding up constrains
        addup_tol_df = self.targets_long[["targtype", "cname", "value"]].copy()
        addup_tol_df = addup_tol_df[addup_tol_df["targtype"] == "addup"]
        addup_tol_df = addup_tol_df.drop("targtype", axis=1)

        addup_tol = self.params.to_array("Addup_tol")
        add_clb = np.where(
            pd.isnull(addup_tol_df["value"]),
            -9e99,
            addup_tol_df["value"] - (abs(addup_tol_df["value"]) * addup_tol),
        )
        add_cub = np.where(
            pd.isnull(addup_tol_df["value"]),
            9e99,
            addup_tol_df["value"] + (abs(addup_tol_df["value"]) * addup_tol),
        )
        addup_tol_df["clb"] = add_clb
        addup_tol_df["cub"] = add_cub
        addup_tol_df["tol"] = addup_tol

        tol_df = pd.concat([agg_tol_df, addup_tol_df])
        return tol_df

    def compare_national_puf_ht2(self):
        """
        Compares national totals from PUF to national totals from HT2
        for the chosen target variables and AGI group.
        """

        ht2 = pd.read_csv(self.ht2_path)

        # Filter for US
        ht2_us = ht2[ht2["STATE"] == "US"]

        puf_temp = self.puf_2017_filter.copy()
        compare_df = pd.DataFrame()
        # Loop through target variables
        for var in self.var_list:
            if var.startswith("A"):
                # HT2 amounts are in thousands
                ht2_val = ht2_us.loc[self.AGI_STUB, var] * 1000
            else:
                ht2_val = ht2_us.loc[self.AGI_STUB, var]

            puf_val = (puf_temp[var] * puf_temp["s006"]).sum()
            perc_dif = round(((puf_val - ht2_val) / ht2_val), 3)
            var_dif = pd.Series(dtype="float64")
            var_dif = pd.Series(
                data={
                    "HT2_val": ht2_val / 1e6,
                    "PUF_val": puf_val / 1e6,
                    "perc_dif": perc_dif,
                },
                name=var,
            )
            compare_df = compare_df.append(var_dif, ignore_index=False)
        return compare_df

    def compare_state_puf_ht2(self):
        """
        Compares totals by state from PUF (calculated using HT2 ratios)
        to state totals from HT2 for the chosen target variables and AGI group.
        """
        ht2 = pd.read_csv(self.ht2_path)
        # Filter HT2 by AGI stub
        ht2_stub = ht2[ht2["AGI_STUB"] == self.AGI_STUB]

        states = list(ht2["STATE"].unique())
        states.pop(0)

        # We will calculate state PUF totals from cc_dense
        cc_dense = self.dense.copy()

        diff_df = pd.DataFrame()
        # Loop through each state
        for state in states:
            # Filter HT2 and cc_dense by state
            state_df = ht2_stub[ht2_stub["STATE"] == state]
            state_dense = cc_dense[cc_dense["STATE"] == state]
            row = {}
            row["state"] = state
            row_df = pd.DataFrame()
            # Loop through each target variable
            for var in self.targ_list:
                if var.startswith("A"):
                    # HT2 amounts are in thousands
                    ht2_val = state_df[var] * 1000
                else:
                    ht2_val = state_df[var]

                targ_name = var + "_targ"
                puf_val = state_dense[targ_name].sum()
                perc_dif = round(((puf_val - ht2_val) / ht2_val), 3)
                row[var] = perc_dif
            row_df = pd.DataFrame.from_dict(row)
            diff_df = pd.concat([diff_df, row_df])
        return diff_df.set_index("state")
