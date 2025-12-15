#!/usr/bin/python3
import os
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

os.environ["PYART_QUIET"] = "1"

from pyiris.cli import parse_args_with_config
from pyiris.converter import run


def main():
    args = parse_args_with_config("pyiris.conf")

    # Balikin parameter list dari config (sama persis seperti script lama)
    args["select_moment"] = args.pop("_select_moment")
    args["save_moment"]   = args.pop("_save_moment")

    args["log_filter"]  = args.pop("_log_filter")
    args["sqi_filter"]  = args.pop("_sqi_filter")
    args["pmi_filter"]  = args.pop("_pmi_filter")
    args["csr_filter"]  = args.pop("_csr_filter")
    args["snlg_filter"] = args.pop("_snlg_filter")
    args["phi_filter"]  = args.pop("_phi_filter")
    args["sdz_filter"]  = args.pop("_sdz_filter")
    args["mdz_filter"]  = args.pop("_mdz_filter")
    args["spec_filter"] = args.pop("_spec_filter")

    # delta time tambahan dari config (karena -d hanya menit)
    args["delta_hours"]   = args.pop("_delta_hours")
    args["delta_seconds"] = args.pop("_delta_seconds")

    run(args)


if __name__ == "__main__":
    main()
