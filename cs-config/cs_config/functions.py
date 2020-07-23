import os
import paramtools
from .helpers import retrieve_puf
import state_taxdata
from state_taxdata.prepdata import TAXDATA_PARAMS, PrepData
from collections import OrderedDict


AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")


def get_version():
    version = state_taxdata.__version__
    return f"state_taxdata v{version}"


def get_inputs(meta_param_dict):
    params = TAXDATA_PARAMS()
    params.specification(serializable=True)

    filtered_params = OrderedDict()
    for k, v in params.dump().items():
        if k == "schema" or v.get("section_1") != "Tolerances":
            filtered_params[k] = v

    default_params = {"Data Preparation": filtered_params}

    return {"meta_parameters": {}, "model_parameters": default_params}


def validate_inputs(meta_param_dict, adjustment, errors_warnings):
    params = TAXDATA_PARAMS()
    params.adjust(adjustment["Data Preparation"], raise_errors=False)
    errors_warnings["Data Preparation"]["errors"].update(params.errors)

    return {"errors_warnings": errors_warnings}


def run_model(meta_param_dict, adjustment):
    params = TAXDATA_PARAMS()
    adjustment = params.adjust(adjustment["Data Preparation"])
    puf_df = retrieve_puf(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    p = PrepData(adjustment=adjustment, puf_df=puf_df)

    compare_national = p.compare_national_puf_ht2()
    compare_state = p.compare_state_puf_ht2()

    table_national = compare_national.to_html(classes="table table-striped table-hover")

    table_state = compare_state.to_html(classes="table table-striped table-hover")

    comp_dict = {
        "renderable": [
            {
                "media_type": "table",
                "title": "National PUF Totals vs National HT2 Totals",
                "data": table_national,
            },
            {
                "media_type": "table",
                "title": "State PUF Totals vs State HT2 Totals",
                "data": table_state,
            },
        ],
        "downloadable": [
            {
                "media_type": "CSV",
                "title": "compare_national",
                "data": compare_national.to_csv(),
            },
            {
                "media_type": "CSV",
                "title": "compare_state",
                "data": compare_state.to_csv(),
            },
        ],
    }

    return comp_dict
