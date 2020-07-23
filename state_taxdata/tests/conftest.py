import pytest
import pandas as pd
import numpy as np
from state_taxdata.prepdata import PrepData


ADJUSTMENT = {
    "AGI_STUB": [{"value": 6}],
    "MARS1_targ": [{"value": True}],
    "A00200_targ": [{"value": True}],
    "N04470_targ": [{"value": True}],
}

@pytest.fixture(scope="session")
def data_no_targs():
	return PrepData()

@pytest.fixture(scope="session")
def data_with_targs():
	return PrepData(ADJUSTMENT)

pd = PrepData(ADJUSTMENT)

@pytest.fixture(scope="session")
def puf_prep():
	return pd.prepare_puf()

@pytest.fixture(scope="session")
def puf_sum():
	return pd.puf_summary()

@pytest.fixture(scope="session")
def ratios():
	return pd.calc_ratios()

@pytest.fixture(scope="session")
def targets_wide():
	return pd.state_targets_wide()

@pytest.fixture(scope="session")
def targets_long():
	return pd.state_targets_long()

@pytest.fixture(scope="session")
def iweights():
	return pd.initial_weights()

@pytest.fixture(scope="session")
def dense():
	return pd.cc_dense()

@pytest.fixture(scope="session")
def sparse():
	return pd.cc_sparse()

@pytest.fixture(scope="session")
def compare_national():
	return pd.compare_national_puf_ht2()

@pytest.fixture(scope="session")
def compare_state():
	return pd.compare_state_puf_ht2()

