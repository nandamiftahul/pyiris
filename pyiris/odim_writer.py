# pyiris/odim_writer.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def string2arrayint64(s, ls):
    a = np.zeros(ls, dtype='int64')
    i = 0
    for c in s:
        a[i] = ord(c)
        i += 1
    return a


def build_hcl_legend():
    # persis seperti script kamu
    dt = np.dtype([('key', 'int64', (64)), ('value', 'int64', (32))])
    legend = np.zeros((26,), dtype=dt)

    hydrometeor_class = [
        'THRESHOLD',
        'NON_MET',
        'RAIN',
        'WET_SNOW',
        'SNOW',
        'GRAUPEL',
        'HAIL',
        'GC/AP',
        'BIO',
        'PRECIPITATION',
        'LARGE_DROPS',
        'LIGHT_PRECIP',
        'MODERATE_PRECIP',
        'HEAVY_PRECIP',
        'STATIFORM',
        'CONVECTIVE',
        'MELTING',
        'NON_MELTING',
        'AUX3',
        'AUX4',
        'AUX5',
        'USER1',
        'USER2',
        'USER3',
        'USER4',
        'USER5'
    ]

    seq = 0
    for h_class in hydrometeor_class:
        legend[seq] = (string2arrayint64(h_class, 64), string2arrayint64('{}'.format(seq), 32))
        seq += 1

    return dt, legend


def write_odim_h5(
    hdf,
    raw,
    fields,
    moments_out,
    gain,
    offset,
    dt,
    legend,
    *,
    start_scan_time,          # datetime
    delta_time,               # raw.time['data']
    azimuth_start,            # raw.azimuth['start']
    azimuth_stop,             # raw.azimuth['stop']
    elevation,                # raw.elevation['data'] (rounded OK)
    fixed_angle,              # raw.fixed_angle
    pulsewidth,               # instrument_parameters['pulse_width']['data']
    prf,                      # 1/prt
    prf_ratio,                # instrument_parameters['prt_ratio']['data']
    NI,                       # instrument_parameters['nyquist_velocity']['data']
    nbins,                    # raw.ngates
    nrays_sweep,              # raw.rays_per_sweep['data']
    nsweeps,                  # raw.nsweeps
    beamwidth_h,              # instrument_parameters['radar_beam_width_h']['data']
    polarization,             # metadata['polarization']
    wavelength,               # instrument_parameters['wavelength']['data']
    task_name,                # cleaned string already
    scan_type,                # 'SCAN' / 'PVOL' already computed
    site_name,                # cleaned string already
    longitude,                # raw.longitude['data']
    latitude,                 # raw.latitude['data']
    radar_altitude,           # altitude['data']
    first_bin_range,          # raw.range['meters_to_center_of_first_gate']
    range_step,               # raw.range['meters_between_gates']
    interp_enable="disable",  # string enable/disable
    econvert="disable",       # string enable/disable
    moment_e="H",             # 'H' / 'V'
):
    """
    Menulis struktur ODIM_H5 persis seperti blok di script utama kamu.
    Tidak membuka/menutup file HDF (itu dilakukan di luar).
    """

    for sweep in range(nsweeps):
        start_ray = raw.get_start(sweep)
        end_ray = raw.get_end(sweep)
        slice_ray = raw.get_slice(sweep)

        d = 1
        for moment in moments_out:
            data = fields[moment]['data'].data[slice_ray]

            df = pd.DataFrame(data)
            dataset = {
                'azimuth_start': azimuth_start[slice_ray],
                'azimuth_stop': azimuth_stop[slice_ray]
            }

            polar_dataset = pd.DataFrame(dataset)
            polar_dataset = pd.concat([df, polar_dataset], axis=1)

            polar_dataset = polar_dataset.sort_values(by=['azimuth_stop'])
            az_start = polar_dataset['azimuth_start'].to_numpy()
            az_stop = polar_dataset['azimuth_stop'].to_numpy()
            polar_dataset = polar_dataset.drop(columns=['azimuth_start'])
            polar_dataset = polar_dataset.drop(columns=['azimuth_stop'])
            data = polar_dataset.to_numpy()

            data = (data - offset[moment]) / gain[moment]
            data = data.astype(np.uint16)

            main = hdf.create_group("dataset{}/data{}".format(sweep + 1, d))
            main.create_dataset("data", data=data, chunks=data.shape, compression="gzip")
            dset = hdf.get("dataset{}/data{}/data".format(sweep + 1, d))
            dset.attrs['CLASS'] = np.bytes_('IMAGE')
            dset.attrs['IMAGE_VERSION'] = np.bytes_('1.2')

            if 'class' in moment.lower():
                main.create_dataset("legend", (26,), data=legend, dtype=dt)

            hdf.create_group("dataset{}/data{}/what".format(sweep + 1, d))
            dset = hdf.get("dataset{}/data{}/what".format(sweep + 1, d))

            # quantity mapping persis seperti script asli (termasuk econvert/moment_e)
            if 'dbt' in moment.lower():
                if 'dbte' in moment.lower():
                    if 'enable' in econvert:
                        if 'H' in moment_e:
                            dset.attrs['quantity'] = np.bytes_("TX")
                        elif 'V' in moment_e:
                            dset.attrs['quantity'] = np.bytes_("TV")
                    else:
                        dset.attrs['quantity'] = np.bytes_("TX")
                elif 'dbzv' in moment.lower():
                    dset.attrs['quantity'] = np.bytes_("TV")
                else:
                    dset.attrs['quantity'] = np.bytes_("TH")
            elif 'dbz' in moment.lower():
                if 'dbze' in moment.lower():
                    if 'enable' in econvert:
                        if 'H' in moment_e:
                            dset.attrs['quantity'] = np.bytes_("DBZX")
                        elif 'V' in moment_e:
                            dset.attrs['quantity'] = np.bytes_("DBZV")
                    else:
                        dset.attrs['quantity'] = np.bytes_("DBZX")
                elif 'dbzv' in moment.lower():
                    dset.attrs['quantity'] = np.bytes_("DBZV")
                else:
                    dset.attrs['quantity'] = np.bytes_("DBZH")
            elif 'vel' in moment.lower():
                if 'velc' in moment.lower():
                    dset.attrs['quantity'] = np.bytes_("VRADDH")
                else:
                    dset.attrs['quantity'] = np.bytes_("VRADH")
            elif 'width' in moment.lower():
                dset.attrs['quantity'] = np.bytes_("WRADH")
            elif 'zdr' in moment.lower():
                dset.attrs['quantity'] = np.bytes_("ZDR")
            elif 'phidp' in moment.lower():
                dset.attrs['quantity'] = np.bytes_("PHIDP")
            elif 'rhohv' in moment.lower():
                dset.attrs['quantity'] = np.bytes_("RHOHV")
            elif 'kdp' in moment.lower():
                dset.attrs['quantity'] = np.bytes_("KDP")
            elif 'sqi' in moment.lower():
                dset.attrs['quantity'] = np.bytes_("SQIH")
            elif 'snr' in moment.lower():
                dset.attrs['quantity'] = np.bytes_("SNRH")
            elif 'class' in moment.lower():
                dset.attrs['quantity'] = np.bytes_("CLASS")
            else:
                dset.attrs['quantity'] = np.bytes_(moment)

            d += 1
            dset.attrs['gain'] = np.double(gain[moment])
            dset.attrs['offset'] = np.double(offset[moment])
            dset.attrs['nodata'] = np.double(0.0)
            dset.attrs['undetect'] = np.double(0.0)

        # dataset/what
        hdf.create_group("dataset{}/what".format(sweep + 1))
        dset = hdf.get("dataset{}/what".format(sweep + 1))
        dset.attrs['product'] = np.bytes_('SCAN')

        sweep_start_time = start_scan_time + timedelta(seconds=(delta_time[start_ray]))
        starttime = sweep_start_time.strftime("%H%M%S")
        startdate = sweep_start_time.strftime("%Y%m%d")
        dset.attrs.create('startdate', startdate, None, dtype='<S9')
        dset.attrs.create('starttime', starttime, None, dtype='<S7')

        sweep_end_datetime = start_scan_time + timedelta(seconds=(delta_time[end_ray]))
        endtime = sweep_end_datetime.strftime("%H%M%S")
        enddate = sweep_end_datetime.strftime("%Y%m%d")

        scan_time = start_scan_time.timestamp() + delta_time[slice_ray]
        dset.attrs.create('enddate', enddate, None, dtype='<S9')
        dset.attrs.create('endtime', endtime, None, dtype='<S7')

        long_scan = sweep_end_datetime - sweep_start_time
        T = 360 / int(long_scan.total_seconds())
        rpm = np.single(T / 6).round(decimals=1)

        # dataset/how
        hdf.create_group("dataset{}/how".format(sweep + 1))
        dset = hdf.get("dataset{}/how".format(sweep + 1))
        dset.attrs['scan_index'] = sweep + 1
        dset.attrs['pulsewidth'] = np.double(pulsewidth[start_ray] * 1e6)
        dset.attrs['lowprf'] = np.double(prf[start_ray] / prf_ratio[start_ray])
        dset.attrs['highprf'] = np.double(prf[start_ray])
        dset.attrs['NI'] = np.double(NI[start_ray])
        dset.attrs['rpm'] = rpm
        dset.attrs['astart'] = az_start[0].astype('float64')
        dset.attrs['startazA'] = az_start.astype('float64')
        dset.attrs['stopazA'] = az_stop.astype('float64')
        dset.attrs['startazT'] = scan_time
        dset.attrs['stopazT'] = scan_time
        dset.attrs['startelT'] = scan_time
        dset.attrs['stopelT'] = scan_time
        dset.attrs['startelA'] = elevation[slice_ray].astype('float64')
        dset.attrs['stopelA'] = elevation[slice_ray].astype('float64')

        # dataset/where
        hdf.create_group("dataset{}/where".format(sweep + 1))
        dset = hdf.get("dataset{}/where".format(sweep + 1))
        dset.attrs['elangle'] = np.single(fixed_angle['data'][sweep].round(decimals=1))
        dset.attrs['nbins'] = np.int_(nbins)
        dset.attrs['rstart'] = np.double(first_bin_range[0])
        dset.attrs['rscale'] = np.double(range_step[0])
        if 'enable' in interp_enable:
            dset.attrs['nrays'] = np.int_(np.nanmax(nrays_sweep))
        else:
            dset.attrs['nrays'] = np.int_(nrays_sweep[sweep])
        dset.attrs['a1gate'] = 0

    # how
    hdf.create_group("how")
    dset = hdf.get("how")
    dset.attrs['task'] = np.bytes_(task_name)
    dset.attrs['beamwidth'] = np.double(beamwidth_h[0])
    dset.attrs['polarization'] = np.bytes_(polarization)
    dset.attrs['scan_count'] = np.int_(nsweeps)
    dset.attrs['wavelength'] = np.double(wavelength[0] / 100)
    dset.attrs['azmethod'] = np.bytes_("AVERAGE")
    dset.attrs['binmethod'] = np.bytes_("AVERAGE")

    # what
    hdf.create_group("what")
    dset = hdf.get("what")
    dset.attrs['object'] = np.bytes_(scan_type)
    dset.attrs['version'] = np.bytes_('H5rad 2.2')
    ingest_start_time = start_scan_time
    ingesttime = ingest_start_time.strftime("%H%M%S")
    ingestdate = ingest_start_time.strftime("%Y%m%d")
    dset.attrs['date'] = np.bytes_(ingestdate)
    dset.attrs['time'] = np.bytes_(ingesttime)
    dset.attrs['source'] = np.bytes_("PLC:{}".format(site_name))

    # where
    hdf.create_group("where")
    dset = hdf.get("where")
    dset.attrs['lon'] = np.double(longitude[0])
    dset.attrs['lat'] = np.double(latitude[0])
    dset.attrs['height'] = np.double(radar_altitude[0])
