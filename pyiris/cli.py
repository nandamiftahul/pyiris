# pyiris/cli.py
import configparser
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def parse_args_with_config(default_config_file: str = "pyiris.conf") -> dict:
    # --- Phase 1: only config ---
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, add_help=False)
    parser.add_argument("-c", "--config", default=default_config_file, help="config file")
    known_args, remaining_argv = parser.parse_known_args()

    config = configparser.ConfigParser()
    config.read(known_args.config)

    # --- Phase 2: full parser with help ---
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config", default=default_config_file, help="config file")

    # ==== persis seperti script kamu: ambil nilai default dari config ====
    input_file       = config["FILE"]["input_file"]
    output_file      = config["FILE"]["output_file"]
    delete_file      = config["FILE"]["delete"]
    mode             = config["FILENAME"]["mode"]
    manual_taskname  = config["FILENAME"]["manual_taskname"]
    select_moment    = config["PARAMETER"]["moment_in"].split(',')
    save_moment      = config["PARAMETER"]["moment_out"].split(',')
    delta_hours      = int(config["DELTATIME"]["delta_hours"])
    delta_minutes    = int(config["DELTATIME"]["delta_minutes"])
    delta_seconds    = int(config["DELTATIME"]["delta_seconds"])

    logz   = float(config["FILTER"]["th_LOGZ"])
    loge   = float(config["FILTER"]["th_LOGE"])
    logv   = float(config["FILTER"]["th_LOGV"])
    logw   = float(config["FILTER"]["th_LOGW"])
    logd   = float(config["FILTER"]["th_LOGD"])
    sqiz   = float(config["FILTER"]["th_SQIZ"])
    sqie   = float(config["FILTER"]["th_SQIE"])
    sqiv   = float(config["FILTER"]["th_SQIV"])
    sqiw   = float(config["FILTER"]["th_SQIW"])
    sqid   = float(config["FILTER"]["th_SQID"])
    pmiz   = float(config["FILTER"]["th_PMIZ"])
    pmie   = float(config["FILTER"]["th_PMIE"])
    pmiv   = float(config["FILTER"]["th_PMIV"])
    pmiw   = float(config["FILTER"]["th_PMIW"])
    pmid   = float(config["FILTER"]["th_PMID"])
    csrz   = float(config["FILTER"]["th_CSRZ"])
    csre   = float(config["FILTER"]["th_CSRE"])
    csrv   = float(config["FILTER"]["th_CSRV"])
    csrw   = float(config["FILTER"]["th_CSRW"])
    csrd   = float(config["FILTER"]["th_CSRD"])
    snlgz  = float(config["FILTER"]["th_SNLGZ"])
    snlge  = float(config["FILTER"]["th_SNLGE"])
    snlgv  = float(config["FILTER"]["th_SNLGV"])
    snlgw  = float(config["FILTER"]["th_SNLGW"])
    snlgd  = float(config["FILTER"]["th_SNLGD"])
    snrd   = float(config["FILTER"]["th_SNRD"])
    phid1  = float(config["FILTER"]["th_PHID1"])
    phid2  = float(config["FILTER"]["th_PHID2"])
    sdz    = float(config["FILTER"]["th_SDZ"])
    mdz    = float(config["FILTER"]["th_MDZ"])
    windowSize = int(config["FILTER"]["me_windowSize"])

    econvert        = config["FILTERMODE"]["econvert"]
    speckle_filter  = config["FILTERMODE"]["speckle_filter"]
    standard_filter = config["FILTERMODE"]["standard_filter"]

    speckle_type = config["FILTERTYPE"]["speckle"].split(',')
    spec_filter  = config["FILTERTYPE"]["speckle_moment"].split(',')
    log_filter   = config["FILTERTYPE"]["LOG"].split(',')
    sqi_filter   = config["FILTERTYPE"]["SQI"].split(',')
    pmi_filter   = config["FILTERTYPE"]["PMI"].split(',')
    csr_filter   = config["FILTERTYPE"]["CSR"].split(',')
    snlg_filter  = config["FILTERTYPE"]["SNLG"].split(',')
    phi_filter   = config["FILTERTYPE"]["PHI"].split(',')
    sdz_filter   = config["FILTERTYPE"]["SDZ"].split(',')
    mdz_filter   = config["FILTERTYPE"]["MDZ"].split(',')

    moment_e     = config["ECONVERT"]["moment_e"]
    ecorrection  = config["ECONVERT"]["ecorrection"]
    ecomposite   = config["ECONVERT"]["ecomposite"]

    mod_enable      = config["MOD"]["mod"]
    interp_enable   = config["MOD"]["interpolate"]
    rpm_mod_enable  = config["MOD"]["rpm_mod"]
    target_rays     = int(config["MOD"]["target_rays"])

    # ==== persis seperti script kamu: add_argument ====
    parser.add_argument("-i", "--input", default=input_file, help="vaisala raw input file path")
    parser.add_argument("-o", "--output", default=output_file, help="vaisala raw output file path")
    parser.add_argument("-d", "--deltatime", default=delta_minutes, type=int, help="delta time from original raw data in minutes")
    parser.add_argument("--logz", default=logz, type=float, help="filter parameter")
    parser.add_argument("--sqiz", default=sqiz, type=float, help="filter parameter")
    parser.add_argument("--pmiz", default=pmiz, type=float, help="filter parameter")
    parser.add_argument("--csrz", default=csrz, type=float, help="filter parameter")
    parser.add_argument("--snlgz", default=snlgz, type=float, help="filter parameter")
    parser.add_argument("--loge", default=loge, type=float, help="filter parameter")
    parser.add_argument("--sqie", default=sqie, type=float, help="filter parameter")
    parser.add_argument("--pmie", default=pmie, type=float, help="filter parameter")
    parser.add_argument("--csre", default=csre, type=float, help="filter parameter")
    parser.add_argument("--snlge", default=snlge, type=float, help="filter parameter")
    parser.add_argument("--logv", default=logv, type=float, help="filter parameter")
    parser.add_argument("--sqiv", default=sqiv, type=float, help="filter parameter")
    parser.add_argument("--pmiv", default=pmiv, type=float, help="filter parameter")
    parser.add_argument("--csrv", default=csrv, type=float, help="filter parameter")
    parser.add_argument("--snlgv", default=snlgv, type=float, help="filter parameter")
    parser.add_argument("--logd", default=logd, type=float, help="filter parameter")
    parser.add_argument("--sqid", default=sqid, type=float, help="filter parameter")
    parser.add_argument("--pmid", default=pmid, type=float, help="filter parameter")
    parser.add_argument("--csrd", default=csrd, type=float, help="filter parameter")
    parser.add_argument("--snlgd", default=snlgd, type=float, help="filter parameter")
    parser.add_argument("--logw", default=logw, type=float, help="filter parameter")
    parser.add_argument("--sqiw", default=sqiw, type=float, help="filter parameter")
    parser.add_argument("--pmiw", default=pmiw, type=float, help="filter parameter")
    parser.add_argument("--csrw", default=csrw, type=float, help="filter parameter")
    parser.add_argument("--snlgw", default=snlgw, type=float, help="filter parameter")
    parser.add_argument("--snr", default=snrd, type=float, help="filter parameter")
    parser.add_argument("--phi1", default=phid1, type=float, help="filter parameter")
    parser.add_argument("--phi2", default=phid2, type=float, help="filter parameter")
    parser.add_argument("--sdz", default=sdz, type=float, help="filter parameter")
    parser.add_argument("--mdz", default=mdz, type=float, help="filter parameter")
    parser.add_argument("--windowSize", default=windowSize, type=int, help="speckle median filter parameter")
    parser.add_argument("--speckletype", default=speckle_type, help="speckle filter type : median or mean or ndwi")
    parser.add_argument("--econvert", default=econvert, help="econvert Th/Tv Zh/Zv replace by Te Ze enable or disable")
    parser.add_argument("--speckle", default=speckle_filter, help="speckle filter enable or disable")
    parser.add_argument("--stdfilter", default=standard_filter, help="standard filter enable or disable")
    parser.add_argument("--deletefile", default=delete_file, help="delete input file after converting enable or disable")
    parser.add_argument("--modetaskname", default=mode, help="output taskname raw or auto or manual")
    parser.add_argument("--taskname", default=manual_taskname, help="output taskname on manual mode")
    parser.add_argument("--econverttype", default=moment_e, help="econvert type H or V")
    parser.add_argument("--ecorrection", default=ecorrection, help="ecorrection disable or enable")
    parser.add_argument("--ecomposite", default=ecomposite, help="ecomposite disable or enable")
    parser.add_argument("--mod", default=mod_enable, help="mod hdf5 file into vaisala raw")
    parser.add_argument("--intp", default=interp_enable, help="interpolation missing rays")
    parser.add_argument("--rpm", default=rpm_mod_enable, help="change rpm with same value")
    parser.add_argument("--targetrays", default=target_rays, type=int, help="number of rays target to fill missing rays")

    args = vars(parser.parse_args(remaining_argv))
    # NOTE: select_moment/save_moment/filter lists tetap sumber dari config (biar 100% sama).
    # Kalau mau, kamu bisa return juga config-derived lists.
    args["_select_moment"] = select_moment
    args["_save_moment"] = save_moment
    args["_log_filter"] = log_filter
    args["_sqi_filter"] = sqi_filter
    args["_pmi_filter"] = pmi_filter
    args["_csr_filter"] = csr_filter
    args["_snlg_filter"] = snlg_filter
    args["_phi_filter"] = phi_filter
    args["_sdz_filter"] = sdz_filter
    args["_mdz_filter"] = mdz_filter
    args["_spec_filter"] = spec_filter

    args["_delta_hours"] = delta_hours
    args["_delta_seconds"] = delta_seconds
    return args
