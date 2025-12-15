# pyiris/converter.py

import os, warnings, glob, traceback, shutil
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import h5py
import numpy as np
from datetime import datetime, timedelta
from time import sleep
import pandas as pd
from math import isnan, log10
import copy

os.environ["PYART_QUIET"] = "1"

import pyart.io as radar
from scipy.ndimage import median_filter
import cv2

from .logger_utils import log_print, getnowtime
from .ray_tools import (
    fill_missing_rays_by_raycount,
    rpm_check,
    check_number_of_rays,
)
from .moments import choose_moment, emoment_check
from .filters import (
    mdfill,
    lee_filter,
    box_kernel,
    sdev_filter,
    mean_filter,
    ndwi,
)
from .odim_writer import build_hcl_legend, write_odim_h5


def list_input_files(input_path: str):
    if "RAW" in input_path:
        return [input_path]
    else:
        return glob.glob(os.path.join(input_path, "*.RAW*"))


def process_one_file(filename: str, args: dict):
    input_file = args["input"]
    output_file = args["output"]
    mode = args["modetaskname"]
    manual_taskname = args["taskname"]

    # FILTER params
    logz = args["logz"]
    sqiz = args["sqiz"]
    pmiz = args["pmiz"]
    csrz = args["csrz"]
    snlgz = args["snlgz"]
    loge = args["loge"]
    sqie = args["sqie"]
    pmie = args["pmie"]
    csre = args["csre"]
    snlge = args["snlge"]
    logv = args["logv"]
    sqiv = args["sqiv"]
    pmiv = args["pmiv"]
    csrv = args["csrv"]
    snlgv = args["snlgv"]
    logd = args["logd"]
    sqid = args["sqid"]
    pmid = args["pmid"]
    csrd = args["csrd"]
    snlgd = args["snlgd"]
    logw = args["logw"]
    sqiw = args["sqiw"]
    pmiw = args["pmiw"]
    csrw = args["csrw"]
    snlgw = args["snlgw"]
    snrd = args["snr"]
    phid1 = args["phi1"]
    phid2 = args["phi2"]
    sdz = args["sdz"]
    mdz = args["mdz"]
    windowSize = args["windowSize"]

    speckle_type = args["speckletype"]
    econvert = args["econvert"]
    speckle_filter = args["speckle"]
    standard_filter = args["stdfilter"]
    delete_file = args["deletefile"]

    moment_e = args["econverttype"]
    ecorrection = args["ecorrection"]
    ecomposite = args["ecomposite"]

    mod_enable = args["mod"]
    interp_enable = args["intp"]
    rpm_mod_enable = args["rpm"]
    target_rays = int(args["targetrays"])

    delta_minutes = args["deltatime"]
    delta_hours = args.get("delta_hours", 0)
    delta_seconds = args.get("delta_seconds", 0)

    # list dari config (dibawa dari main.py)
    select_moment = args["select_moment"]
    save_moment = args["save_moment"]
    log_filter = args["log_filter"]
    sqi_filter = args["sqi_filter"]
    pmi_filter = args["pmi_filter"]
    csr_filter = args["csr_filter"]
    snlg_filter = args["snlg_filter"]
    phi_filter = args["phi_filter"]
    sdz_filter = args["sdz_filter"]
    mdz_filter = args["mdz_filter"]
    spec_filter = args["spec_filter"]

    temp_file = "/etc/pyiris/temp"

    log_print("[{}] : starting converting {}".format(getnowtime(), os.path.basename(filename)))

    try:
        hdf_name = os.path.basename(filename)
        hdf = h5py.File(os.path.join(temp_file, hdf_name + ".h5"), 'w')
        hdf.attrs['Conventions'] = np.bytes_("ODIM_H5/V2_2")

        raw = radar.read_sigmet(filename, file_field_names=True, full_xhdr=True, time_ordered="full")

        if 'enable' in interp_enable:
            log_print('[{}] : interpolation enable'.format(getnowtime()))
            raw = fill_missing_rays_by_raycount(raw, target_rays=target_rays)

        missing_rays_flag = check_number_of_rays(raw, target_rays)

        if 'enable' in rpm_mod_enable:
            log_print('[{}] : rpm mod enable'.format(getnowtime()))
            raw = rpm_check(raw)

        fields = raw.fields

        moments = []
        for moment in select_moment:
            for m in fields:
                if moment in m:
                    moments.append(m)

        # E check
        moments = emoment_check(moments, select_moment, "DBZE")
        # moments = emoment_check(moments,select_moment,"DBZV")
        moments = emoment_check(moments, select_moment, "DBTE")
        moments = emoment_check(moments, select_moment, "VELC")

        log_moments = choose_moment(fields, log_filter)
        sqi_moments = choose_moment(fields, sqi_filter)
        pmi_moments = choose_moment(fields, pmi_filter)
        csr_moments = choose_moment(fields, csr_filter)
        snr_moments = choose_moment(fields, snlg_filter)
        phi_moments = choose_moment(fields, phi_filter)
        sdz_moments = choose_moment(fields, sdz_filter)
        mdz_moments = choose_moment(fields, mdz_filter)
        spec_moments = choose_moment(fields, spec_filter)

        log_print('[{}] : moment available -> '.format(getnowtime()), list(fields.keys()))
        log_print('[{}] : moment selected  -> '.format(getnowtime()), moments)

        altitude = raw.altitude
        fixed_angle = raw.fixed_angle
        instrument_parameters = raw.instrument_parameters
        metadata = raw.metadata
        time = raw.time
        sweep_end_ray_index = raw.sweep_end_ray_index
        sweep_start_ray_index = raw.sweep_start_ray_index
        site_name = metadata['instrument_name']

        task_name = metadata['sigmet_task_name']
        task_name = task_name.decode("utf-8").replace(' ', '')
        if 'manual' in mode:
            task_name = manual_taskname
        elif 'auto' in mode:
            task_name = 'py' + task_name

        polarization = metadata['polarization']
        site_name = site_name.decode("utf-8").replace(' ', '')

        wavelength = instrument_parameters['wavelength']['data']
        NI = instrument_parameters['nyquist_velocity']['data']
        prf_ratio = instrument_parameters['prt_ratio']['data']
        beamwidth_h = instrument_parameters['radar_beam_width_h']['data']
        beamwidth_v = instrument_parameters['radar_beam_width_v']['data']
        pulsewidth = instrument_parameters['pulse_width']['data']
        prt = instrument_parameters['prt']['data']
        prf = 1 / prt

        scan_type = raw.scan_type
        nbins = raw.ngates
        nsweeps = raw.nsweeps
        nrays = raw.nrays
        nrays_sweep = raw.rays_per_sweep['data']

        latitude = raw.latitude['data']
        longitude = raw.longitude['data']

        azimuth = raw.azimuth['data']
        azimuth_start = raw.azimuth['start']
        azimuth_stop = raw.azimuth['stop']

        elevation = raw.elevation['data'].round(decimals=2)
        elevation_start = raw.elevation['start']
        elevation_stop = raw.elevation['stop']

        first_bin_range = raw.range['meters_to_center_of_first_gate']
        range_step = raw.range['meters_between_gates']
        a1gate = raw.range['a1gate']

        radar_altitude = altitude['data']

        start_scan_time = time['units']
        start_scan_time = datetime.strptime(start_scan_time, 'seconds since %Y-%m-%dT%H:%M:%SZ') + timedelta(
            hours=delta_hours, minutes=delta_minutes, seconds=delta_seconds
        )
        delta_time = time['data']

        if nsweeps == 1:
            scan_type = 'SCAN'
        else:
            scan_type = 'PVOL'

        # FILTERING

        index_of_dbz = -1
        index_of_dbze = -1
        index_of_dbt = -1
        index_of_dbte = -1
        index_of_vel = -1
        index_of_velc = -1
        index_of_width = -1
        index_of_log = -1
        index_of_csr = -1
        index_of_pmi = -1
        index_of_sqi = -1
        index_of_snr = -1
        index_of_zdr = -1
        index_of_rho = -1
        index_of_phi = -1
        index_of_hcl = -1

        for moment in moments:
            if 'dbt' in moment.lower():
                if 'dbte' in moment.lower():
                    index_of_dbte = moments.index(moment)
                else:
                    index_of_dbt = moments.index(moment)

            if 'dbz' in moment.lower():
                if 'dbze' in moment.lower():
                    index_of_dbze = moments.index(moment)
                else:
                    index_of_dbz = moments.index(moment)

            if 'vel' in moment.lower():
                if 'velc' in moment.lower():
                    index_of_velc = moments.index(moment)
                else:
                    index_of_vel = moments.index(moment)

            if 'width' in moment.lower():
                index_of_width = moments.index(moment)

            if 'log' in moment.lower():
                index_of_log = moments.index(moment)

            if 'pmi' in moment.lower():
                index_of_pmi = moments.index(moment)

            if 'sqi' in moment.lower():
                index_of_sqi = moments.index(moment)

            if 'csp' in moment.lower():
                index_of_csr = moments.index(moment)

            if 'snr' in moment.lower():
                index_of_snr = moments.index(moment)

            if 'phidp' in moment.lower():
                index_of_phi = moments.index(moment)

            if 'zdr' in moment.lower():
                index_of_zdr = moments.index(moment)

            if 'rhohv' in moment.lower():
                index_of_rho = moments.index(moment)

            if 'class' in moment.lower():
                index_of_hcl = moments.index(moment)

        if 'enable' in standard_filter:
            log_print('[{}] : standard filter enable'.format(getnowtime()))

            if index_of_snr >= 0:
                snr = fields[moments[index_of_snr]]['data']
                if index_of_phi >= 0:
                    log_print('[{}] : SDPHI filter applied to {}'.format(getnowtime(), phi_moments))
                    phi = sdev_filter(fields[moments[index_of_phi]]['data'], (3, 3))
                    for moment in phi_moments:
                        data = fields[moment]['data']
                        mask_snrd = np.logical_or(
                            np.logical_and(snr > snrd, phi > phid1),
                            np.logical_and(snr <= snrd, phi > phid2)
                        )
                        data[mask_snrd] = np.nan
                        fields[moment]['data'] = data
                else:
                    log_print('[{}] : SDPHI filter skipped to {} , phi data unvailable'.format(getnowtime(), phi_moments))
            else:
                if index_of_phi >= 0:
                    log_print('[{}] : SDPHI filter applied to {} using th2 due to snr data unvailable'.format(getnowtime(), phi_moments))
                    phi = sdev_filter(fields[moments[index_of_phi]]['data'], (3, 3))
                    for moment in phi_moments:
                        data = fields[moment]['data']
                        data[phi < phid2] = np.nan
                        fields[moment]['data'] = data
                else:
                    log_print('[{}] : SDPHI filter skipped to {} , phi data unvailable'.format(getnowtime(), phi_moments))

            log_print('[{}] : SDZ filter applied to {}'.format(getnowtime(), sdz_moments))
            for moment in sdz_moments:
                data = fields[moment]['data']
                data[np.isnan(data)] = -327
                sd = sdev_filter(data, (3, 3))
                mask_sdz = np.logical_or(sd == 0, sd > sdz)
                data[mask_sdz] = np.nan
                data[data == -327] = np.nan
                fields[moment]['data'] = data

            log_print('[{}] : MDZ filter applied to {}'.format(getnowtime(), mdz_moments))
            for moment in mdz_moments:
                data = fields[moment]['data']
                data[np.isnan(data)] = -327
                md = median_filter(data, size=3)
                data[md < mdz] = np.nan
                data[data == -327] = np.nan
                fields[moment]['data'] = data

            if index_of_log >= 0:
                log_print('[{}] : LOG filter applied to {}'.format(getnowtime(), log_moments))
                for moment in log_moments:
                    data = fields[moment]['data']
                    log = fields[moments[index_of_log]]['data']
                    if 'DBZ' in moment:
                        if 'DBZE' in moment:
                            data[log < loge] = np.nan
                        else:
                            data[log < logz] = np.nan
                    elif 'DBT' in moment:
                        if 'DBTE' in moment:
                            data[log < loge] = np.nan
                        else:
                            data[log < logz] = np.nan
                    elif 'VEL' in moment:
                        data[log < logv] = np.nan
                    elif 'WIDTH' in moment:
                        data[log < logw] = np.nan
                    else:
                        data[log < logd] = np.nan

                    fields[moment]['data'] = data
            else:
                log_print('[{}] : LOG filter skipped to {} , LOG data unvailable'.format(getnowtime(), log_moments))

            if index_of_snr >= 0:
                log_print('[{}] : SNR filter applied to {}'.format(getnowtime(), snr_moments))

                snlgzt = 10 * log10(10 ** (snlgz / 10) - 1)
                snlget = 10 * log10(10 ** (snlge / 10) - 1)
                snlgvt = 10 * log10(10 ** (snlgv / 10) - 1)
                snlgwt = 10 * log10(10 ** (snlgw / 10) - 1)
                snlgdt = 10 * log10(10 ** (snlgd / 10) - 1)
                for moment in snr_moments:
                    data = fields[moment]['data']
                    snr = fields[moments[index_of_snr]]['data']

                    if 'DBZ' in moment:
                        if 'DBZE' in moment:
                            data[snr < snlget] = np.nan
                        else:
                            data[snr < snlgzt] = np.nan
                    elif 'DBT' in moment:
                        if 'DBTE' in moment:
                            data[snr < snlget] = np.nan
                        else:
                            data[snr < snlgzt] = np.nan
                    elif 'VEL' in moment:
                        data[snr < snlgvt] = np.nan
                    elif 'WIDTH' in moment:
                        data[snr < snlgwt] = np.nan
                    else:
                        data[snr < snlgdt] = np.nan
                    fields[moment]['data'] = data
            else:
                log_print('[{}] : SNR filter skipped to {} , SNR data unvailable'.format(getnowtime(), snr_moments))

            if index_of_csr >= 0:
                log_print('[{}] : CSR filter applied to {}'.format(getnowtime(), csr_moments))

                for moment in csr_moments:
                    data = fields[moment]['data']
                    csr = fields[moments[index_of_csr]]['data']
                    if 'DBZ' in moment:
                        if 'DBZE' in moment:
                            data[csr > csre] = np.nan
                        else:
                            data[csr > csrz] = np.nan
                    elif 'DBT' in moment:
                        if 'DBTE' in moment:
                            data[csr > csre] = np.nan
                        else:
                            data[csr > csrz] = np.nan
                    elif 'VEL' in moment:
                        data[csr > csrv] = np.nan
                    elif 'WIDTH' in moment:
                        data[csr > csrw] = np.nan
                    else:
                        data[csr > csrd] = np.nan
                    fields[moment]['data'] = data
            else:
                log_print('[{}] : CSR filter skipped to {} , CSR data unvailable'.format(getnowtime(), csr_moments))

            if index_of_sqi >= 0:
                log_print('[{}] : SQI filter applied to {}'.format(getnowtime(), sqi_moments))
                for moment in sqi_moments:
                    data = fields[moment]['data']
                    sqi = fields[moments[index_of_sqi]]['data']
                    if 'DBZ' in moment:
                        if 'DBZE' in moment:
                            data[sqi < sqie] = np.nan
                        else:
                            data[sqi < sqiz] = np.nan
                    elif 'DBT' in moment:
                        if 'DBTE' in moment:
                            data[sqi < sqie] = np.nan
                        else:
                            data[sqi < sqiz] = np.nan
                    elif 'VEL' in moment:
                        data[sqi < sqiv] = np.nan
                    elif 'WIDTH' in moment:
                        data[sqi < sqiw] = np.nan
                    else:
                        data[sqi < sqid] = np.nan
                    fields[moment]['data'] = data
            else:
                log_print('[{}] : SQI filter skipped to {} , SQI data unvailable'.format(getnowtime(), sqi_moments))

            if index_of_pmi >= 0:
                log_print('[{}] : PMI filter applied to {}'.format(getnowtime(), pmi_moments))
                for moment in pmi_moments:
                    data = fields[moment]['data']
                    pmi = fields[moments[index_of_pmi]]['data']
                    if 'DBZ' in moment:
                        if 'DBZE' in moment:
                            data[pmi < pmie] = np.nan
                        else:
                            data[pmi < pmiz] = np.nan
                    elif 'DBT' in moment:
                        if 'DBTE' in moment:
                            data[pmi < pmie] = np.nan
                        else:
                            data[pmi < pmiz] = np.nan
                    elif 'VEL' in moment:
                        data[pmi < pmiv] = np.nan
                    elif 'WIDTH' in moment:
                        data[pmi < pmiw] = np.nan
                    else:
                        data[pmi < pmid] = np.nan
                    fields[moment]['data'] = data
            else:
                log_print('[{}] : PMI filter skipped to {} , PMI data unvailable'.format(getnowtime(), pmi_moments))

        else:
            log_print('[{}] : standard filter disable'.format(getnowtime()))

        if 'enable' in speckle_filter:
            log_print('[{}] : speckle {} filter enable'.format(getnowtime(), speckle_type))
            for moment in spec_moments:
                for spec in speckle_type:
                    data = fields[moment]['data']
                    if 'median' in spec:
                        fields[moment]['data'] = np.ma.masked_array(median_filter(data, size=windowSize))
                    elif 'mean' in spec:
                        fields[moment]['data'] = np.ma.masked_array(cv2.filter2D(data, -1, box_kernel(windowSize)))
                    elif 'stdmean' in spec:
                        fields[moment]['data'] = np.ma.masked_array(mean_filter(data, windowSize))
                    elif 'lee' in spec:
                        fields[moment]['data'] = np.ma.masked_array(lee_filter(data, windowSize))
                    elif 'ndwi' in spec:
                        fields[moment]['data'] = np.ma.masked_array(ndwi(data))
                    elif 'mdfill' in spec:
                        fields[moment]['data'] = np.ma.masked_array(mdfill(data))

        else:
            log_print('[{}] : speckle filter disable'.format(getnowtime()))

        if 'enable' in ecorrection:
            log_print('[{}] : ecorrection enable'.format(getnowtime()))
            if index_of_dbze >= 0 and index_of_zdr >= 0 and index_of_rho >= 0:
                zdr = fields[moments[index_of_zdr]]['data']
                rho = fields[moments[index_of_rho]]['data']
                log_print('[{}] : DBZE found in ->'.format(getnowtime()), index_of_dbze)
                fields[moments[index_of_dbze]]['data'] += zdr / 2 - 10 * np.log10(rho)
            if index_of_dbte >= 0 and index_of_zdr >= 0 and index_of_rho >= 0:
                zdr = fields[moments[index_of_zdr]]['data']
                rho = fields[moments[index_of_rho]]['data']
                log_print('[{}] : DBTE found in ->'.format(getnowtime()), index_of_dbte)
                fields[moments[index_of_dbte]]['data'] += zdr / 2 - 10 * np.log10(rho)
        else:
            log_print('[{}] : ecorrection disable'.format(getnowtime()))

        if 'enable' in econvert:
            log_print('[{}] : econvert enable, {} choosed'.format(getnowtime(), moment_e))
            if 'enable' in ecomposite:
                log_print('[{}] : ecomposite enable'.format(getnowtime()))
            else:
                log_print('[{}] : ecomposite disable'.format(getnowtime()))

            if index_of_dbz >= 0 and index_of_dbze >= 0:
                log_print('[{}] : DBZ and DBZE found in ->'.format(getnowtime()), [index_of_dbz, index_of_dbze])
                if 'H' in moment_e:
                    if 'enable' in ecomposite:
                        data_Z = fields[moments[index_of_dbz]]['data']
                        data_ZE = fields[moments[index_of_dbze]]['data']
                        data_Z[np.isnan(data_Z)] = -327
                        data_ZE[np.isnan(data_ZE)] = -327
                        data = np.maximum(data_Z, data_ZE)
                        data[data == -327] = np.nan
                        fields[moments[index_of_dbze]]['data'] = data
                    else:
                        fields[moments[index_of_dbze]]['data'] = fields[moments[index_of_dbze]]['data']
                elif 'V' in moment_e:
                    if 'enable' in ecomposite:
                        data_Z = fields[moments[index_of_dbz]]['data']
                        data_ZE = fields[moments[index_of_dbze]]['data']
                        data_Z[np.isnan(data_Z)] = -327
                        data_ZE[np.isnan(data_ZE)] = -327
                        data = np.maximum(data_Z, data_ZE)
                        data[data == -327] = np.nan
                        fields[moments[index_of_dbze]]['data'] = data

            if index_of_dbt >= 0 and index_of_dbte >= 0:
                log_print('[{}] : DBT and DBTE found in ->'.format(getnowtime()), [index_of_dbt, index_of_dbte])
                if 'H' in moment_e:
                    if 'enable' in ecomposite:
                        data_T = fields[moments[index_of_dbt]]['data']
                        data_TE = fields[moments[index_of_dbte]]['data']
                        data_T[np.isnan(data_T)] = -327
                        data_TE[np.isnan(data_TE)] = -327
                        data = np.maximum(data_T, data_TE)
                        data[data == -327] = np.nan
                        fields[moments[index_of_dbte]]['data'] = data
                    else:
                        fields[moments[index_of_dbte]]['data'] = fields[moments[index_of_dbte]]['data']
                elif 'V' in moment_e:
                    if 'enable' in ecomposite:
                        data_T = fields[moments[index_of_dbt]]['data']
                        data_TE = fields[moments[index_of_dbte]]['data']
                        data_T[np.isnan(data_T)] = -327
                        data_TE[np.isnan(data_TE)] = -327
                        data = np.maximum(data_T, data_TE)
                        data[data == -327] = np.nan
                        fields[moments[index_of_dbte]]['data'] = data

        else:
            log_print('[{}] : econvert disable'.format(getnowtime()))

        hcl_data = fields[moments[index_of_hcl]]['data']
        hcl_data = (np.bitwise_and(hcl_data.astype(int), ~(0b11111 << 3)))
        fields[moments[index_of_hcl]]['data'] = hcl_data

        # END OF FILTERING

        # OFFSET AND GAIN
        offset = {}
        gain = {}
        for moment in moments:
            data = fields[moment]['data'].data
            offset[moment] = np.nanmin(data)
            if 'rhohv' in moment.lower() or 'phidp' in moment.lower() or 'sqi' in moment.lower() or 'pmi' in moment.lower():
                gain[moment] = abs(offset[moment])
            elif 'class' in moment.lower():
                gain[moment] = 1.0
                offset[moment] = 0.0
            else:
                gain[moment] = 0.01

            if isnan(gain[moment]) or gain[moment] == 0:
                gain[moment] = 1.0
            if isnan(offset[moment]):
                offset[moment] = 0.0

        moments_out = []
        for moment in save_moment:
            for m in fields:
                if moment in m:
                    moments_out.append(m)

        # E check
        moments_out = emoment_check(moments_out, save_moment, "DBZE")
        moments_out = emoment_check(moments_out, save_moment, "DBTE")
        moments_out = emoment_check(moments_out, save_moment, "VELC")

        log_print('[{}] : moment saved  ->'.format(getnowtime()), moments_out)

        # ===== ODIM writing dipindah ke odim_writer.py =====
        dt, legend = build_hcl_legend()

        write_odim_h5(
            hdf,
            raw,
            fields,
            moments_out,
            gain,
            offset,
            dt,
            legend,
            start_scan_time=start_scan_time,
            delta_time=delta_time,
            azimuth_start=azimuth_start,
            azimuth_stop=azimuth_stop,
            elevation=elevation,
            fixed_angle=fixed_angle,
            pulsewidth=pulsewidth,
            prf=prf,
            prf_ratio=prf_ratio,
            NI=NI,
            nbins=nbins,
            nrays_sweep=nrays_sweep,
            nsweeps=nsweeps,
            beamwidth_h=beamwidth_h,
            polarization=polarization,
            wavelength=wavelength,
            task_name=task_name,
            scan_type=scan_type,
            site_name=site_name,
            longitude=longitude,
            latitude=latitude,
            radar_altitude=radar_altitude,
            first_bin_range=first_bin_range,
            range_step=range_step,
            interp_enable=interp_enable,
            econvert=econvert,
            moment_e=moment_e,
        )

        hdf.close()
        sleep(2)
        shutil.move(os.path.join(temp_file, hdf_name + ".h5"), os.path.join(output_file, hdf_name + ".h5"))
        log_print("[{}] : finished converting {}".format(getnowtime(), filename))
        log_print("[{}] : saving file hdf5 to {}/{}.h5".format(getnowtime(), output_file, os.path.basename(filename)))

        if 'enable' in mod_enable:
            if 'enable' not in missing_rays_flag:
                log_print("[{}] : modify {}".format(getnowtime(), filename))
                os.system("hdf52mod/hdf52mod -i {}.h5 -o {}.ppp > log/hdf52mod.log".format(
                    os.path.basename(filename), os.path.basename(filename)
                ))
                log_print("[{}] : successfully mod file".format(getnowtime()))
            else:
                log_print("[{}] : mod force stop due to missing rays".format(getnowtime()))

        if 'enable' in delete_file:
            os.remove(filename)

    except Exception:
        msg_error = traceback.format_exc()
        log_print("[{}] : error {}\n\n".format(getnowtime(), msg_error))


def run(args: dict):
    filenames = list_input_files(args["input"])
    for filename in filenames:
        process_one_file(filename, args)
