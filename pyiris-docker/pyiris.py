#! /usr/bin/python3

import os, warnings, glob, re, traceback, sys
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning)

import h5py
import numpy as np
from datetime import datetime, timedelta
from time import sleep
import pandas as pd
from statistics import mean
from math import isnan, log10, pi, sqrt
import configparser
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

os.environ["PYART_QUIET"] = "1"

import pyart.io as radar
from scipy.ndimage import median_filter, filters
from scipy.signal import medfilt2d
import cv2
from skimage import exposure
from difflib import get_close_matches

from scipy.ndimage import generic_filter
from scipy.interpolate import interp1d
import copy

sys.path.append("/etc/pyiris/plot")
from pyirisgraph import plot_ppi, plot_basic_ppi, plot_multi_ppi
#from pyirisgui import show_gui

from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance

from scipy.signal import convolve2d

def check_number_of_rays(radar, target_ray_count):
	print("📊 Number of rays per sweep (grouped by elevation):")
	sweep_starts = radar.sweep_start_ray_index['data']
	sweep_ends = radar.sweep_end_ray_index['data']
	elevations = radar.elevation['data']

	any_missing = False  # untuk global flag

	for i, (start, end) in enumerate(zip(sweep_starts, sweep_ends)):
		sweep_elev = np.round(np.median(elevations[start:end+1]), 1)
		ray_count = end - start + 1
		flag = "✅ OK" if ray_count == target_ray_count else "⚠️ Missing Rays"
		if ray_count != target_ray_count:
			any_missing = True
		print(f"  Sweep {i}: Elevation {sweep_elev:.1f}° -> {ray_count} rays [{flag}]")

	if not any_missing:
		print("🎉 All sweeps have complete rays.")
		missing_rays_status = "disable"
	else:
		print("⚠️ Some sweeps have missing rays.")
		missing_rays_status = "enable"
	return missing_rays_status

def fill_missing_rays_by_raycount(radar, target_rays=None):
	"""
	Fill missing rays for each sweep in a Py-ART radar object so that every sweep has the desired 
	number of rays. Missing rays are inserted at the largest azimuth gaps. When inserting a new ray,
	only the azimuth list is modified (i.e. “sorted” so that the new azimuth falls in order), while 
	the time and elevation arrays retain their original ordering (except at the insertion locations, 
	where a nearest-neighbor or median value is used).

	Parameters:
		radar : Py-ART Radar object
		target_rays : int, optional
			Desired number of rays per sweep. If None, no extra rays are inserted (i.e. each sweep remains 
			at its original count).

	Returns:
		new_radar : Py-ART Radar object with missing rays filled.
	"""
	fields = radar.fields.keys()
	filled_fields = {f: [] for f in fields}
	filled_azimuth = []
	filled_elevation = []
	filled_time = []
	sweep_start = []
	sweep_end = []

	ray_idx = 0

	# Process each sweep separately
	for sweep in range(radar.nsweeps):
		start = radar.sweep_start_ray_index['data'][sweep]
		end = radar.sweep_end_ray_index['data'][sweep] + 1

		# Get original arrays for this sweep
		az = radar.azimuth['data'][start:end]
		elev = radar.elevation['data'][start:end]
		t = radar.time['data'][start:end]
		
		# Set target rays for this sweep. If target_rays is not provided, keep original count.
		n_current = len(az)
		if target_rays is None:
			target_rays_sweep = n_current
		else:
			target_rays_sweep = target_rays
		n_missing = target_rays_sweep - n_current
		# Convert arrays to lists so that we can insert at specific positions.
		# (We assume the original sweep is already nearly ordered by azimuth.)
		az_list   = list(az)
		elev_list = list(elev)
		time_list = list(t)
		field_lists = {f: list(radar.fields[f]['data'][start:end]) for f in fields}
		
		# Insert missing rays one-by-one; use the largest azimuth gap each time.
		for _ in range(n_missing):
			current_az = np.array(az_list)
			# To handle wrap-around, append first angle + 360.
			az_extended = np.append(current_az, current_az[0] + 360)
			# Compute gap between consecutive angles.
			gaps = np.diff(az_extended)
			gap_idx = np.argmax(gaps)
			
			# Compute new azimuth as the midpoint of the largest gap.
			az1 = current_az[gap_idx]
			az2 = current_az[(gap_idx + 1) % len(current_az)]
			# Compute gap difference modulo 360.
			gap_diff = (az2 - az1) % 360
			new_az = (az1 + gap_diff/2) % 360
			
			# Since current_az is assumed sorted, the proper insertion point is gap_idx+1.
			insert_idx = gap_idx + 1
			
			# Insert new azimuth.
			az_list.insert(insert_idx, new_az)
			# For elevation, we use the median of existing values.
			elev_list.insert(insert_idx, np.median(elev_list))
			# For time, use the nearest original time (here we choose the time of the preceding ray).
			if insert_idx == 0:
				nearest_time = time_list[0]
			else:
				nearest_time = time_list[insert_idx - 1]
			time_list.insert(insert_idx, nearest_time)
			
			# For every field, interpolate linearly between the two adjacent rays.
			for f in fields:
				data_array = np.array(field_lists[f])
				val1 = data_array[gap_idx]
				# For wrap-around, use modulo indexing.
				val2 = data_array[(gap_idx + 1) % len(data_array)]
				new_val = (val1 + val2) / 2.0
				field_lists[f].insert(insert_idx, new_val)
		
		# Append the results of this sweep into our overall containers.
		filled_azimuth.extend(az_list)
		filled_elevation.extend(elev_list)
		filled_time.extend(time_list)
		for f in fields:
			filled_fields[f].append(np.ma.array(field_lists[f]))
		sweep_start.append(ray_idx)
		ray_idx += len(az_list)
		sweep_end.append(ray_idx - 1)
	
	# Build new radar object with the filled data.
	new_radar = copy.deepcopy(radar)
	new_radar.azimuth['data'] = np.array(filled_azimuth, dtype=np.float32)
	new_radar.elevation['data'] = np.array(filled_elevation, dtype=np.float32)
	new_radar.time['data'] = np.array(filled_time, dtype=np.float64)
	new_radar.sweep_start_ray_index['data'] = np.array(sweep_start, dtype=np.int32)
	new_radar.sweep_end_ray_index['data'] = np.array(sweep_end, dtype=np.int32)

	for f in fields:
		# Concatenate the per-sweep arrays vertically.
		new_radar.fields[f]['data'] = np.ma.array(np.concatenate(filled_fields[f]))

	print("📊 Ray count per sweep (before ➜ after), grouped by elevation:")

	orig_starts = radar.sweep_start_ray_index['data']
	orig_ends = radar.sweep_end_ray_index['data']
	orig_elevs = radar.elevation['data']

	new_starts = new_radar.sweep_start_ray_index['data']
	new_ends = new_radar.sweep_end_ray_index['data']
	new_elevs = new_radar.elevation['data']

	for i, (orig_start, orig_end, new_start, new_end) in enumerate(zip(orig_starts, orig_ends, new_starts, new_ends)):
		elev_before = np.round(np.median(orig_elevs[orig_start:orig_end+1]), 1)
		elev_after  = np.round(np.median(new_elevs[new_start:new_end+1]), 1)
		count_before = orig_end - orig_start + 1
		count_after = new_end - new_start + 1
		print(f"  Sweep {i}: Elevation {elev_before:.1f}° ➜ {elev_after:.1f}° | Rays: {count_before} ➜ {count_after}")


	print("✅ Done. Missing rays filled by ray count; azimuths are sorted by insertion, while time and elevation remain in original order except at insertions.")
	return new_radar

def fill_missing_rays(radar):
	# Configuration
	az_spacing = 1.0
	full_az = np.arange(0, 360, az_spacing)
	fields = radar.fields.keys()
	filled_fields = {f: [] for f in fields}
	filled_azimuth = []
	filled_elevation = []
	filled_time = []
	sweep_start = []
	sweep_end = []

	ray_idx = 0

	# Process each sweep
	for sweep in range(radar.nsweeps):
		print(f"Processing sweep {sweep}")
		start = radar.sweep_start_ray_index['data'][sweep]
		end = radar.sweep_end_ray_index['data'][sweep]

		az = radar.azimuth['data'][start:end+1]
		elev = radar.elevation['data'][start:end+1]
		t = radar.time['data'][start:end+1]

		# Find missing azimuths
		az_rounded = np.round(az).astype(int) % 360
		existing_az = set(az_rounded)
		missing_az = sorted(set(np.round(full_az).astype(int)) - existing_az)

		# Interpolate missing rays for each field
		for field in fields:
			orig_data = radar.fields[field]['data'][start:end+1]
			n_rng = radar.ngates

			# Interpolated values for missing azimuths
			interp_data = []

			for gate in range(n_rng):
				vals = orig_data[:, gate]
				valid = ~np.ma.getmaskarray(vals)
				if np.sum(valid) < 2:
					continue
				f_interp = interp1d(
					az[valid],
					vals[valid],
					kind='linear',
					bounds_error=False,
					fill_value=np.nan
				)
				interp_vals = f_interp(missing_az)
				interp_data.append(interp_vals)

			# Reshape interpolated data into (n_missing, n_gates)
			# Ensure correct shape (n_missing, ngates)
			interp_data = np.array(interp_data).T  # Shape: (n_missing, ?)
			interp_data = np.ma.masked_invalid(interp_data)

			# Pad if needed
			if interp_data.shape[1] < radar.ngates:
				pad_width = radar.ngates - interp_data.shape[1]
				interp_data = np.pad(
					interp_data,
					((0, 0), (0, pad_width)),
					mode='constant',
					constant_values=np.nan
				)
				interp_data = np.ma.masked_invalid(interp_data)
			elif interp_data.shape[1] > radar.ngates:
				# Truncate if somehow longer
				interp_data = interp_data[:, :radar.ngates]

			# Start with lists, not arrays
			field_data = list(orig_data)
			az_data = list(az)
			elev_data = list(elev)
			time_data = list(t)

			# For each missing azimuth, find insert index
			for i, az_miss in enumerate(missing_az):
				# Interpolated moment for this azimuth
				interp_ray = interp_data[i]  # Shape: (ngates,)
				
				# Insert position based on azimuth
				insert_idx = np.searchsorted(az_data, az_miss)

				# Insert into each list
				field_data.insert(insert_idx, interp_ray)
				az_data.insert(insert_idx, az_miss)
				elev_data.insert(insert_idx, np.median(elev))   # or interpolate if needed
				#time_data.insert(insert_idx, np.mean(t))		# or interpolate if needed
				# Use nearest time from original data to preserve RPM
				if insert_idx == 0:
					nearest_time = t[0]
				elif insert_idx >= len(t):
					nearest_time = t[-1]
				else:
					before = t[insert_idx - 1]
					after = t[insert_idx]
					nearest_time = before if abs(az_data[insert_idx - 1] - az_miss) < abs(az_data[insert_idx] - az_miss) else after

				time_data.insert(insert_idx, nearest_time)

			# Append to sweep-level containers
			filled_fields[field].append(np.ma.array(field_data))
			if field == list(fields)[0]:
				filled_azimuth.extend(az_data)
				filled_elevation.extend(elev_data)
				filled_time.extend(time_data)
				sweep_start.append(ray_idx)
				ray_idx += len(az_data)
				sweep_end.append(ray_idx - 1)

	# Build new radar
	new_radar = copy.deepcopy(radar)
	new_radar.azimuth['data'] = np.array(filled_azimuth, dtype=np.float32)
	new_radar.elevation['data'] = np.array(filled_elevation, dtype=np.float32)
	new_radar.time['data'] = np.array(filled_time, dtype=np.float64)
	new_radar.sweep_start_ray_index['data'] = np.array(sweep_start, dtype=np.int32)
	new_radar.sweep_end_ray_index['data'] = np.array(sweep_end, dtype=np.int32)

	for field in fields:
		all_data = np.ma.vstack(filled_fields[field])
		new_radar.fields[field]['data'] = all_data

	print("✅ Done. Missing rays filled. Original data preserved.")
	return new_radar

def rpm_check(radar):
	# Compute RPM per sweep
	sweep_rpms = []
	for sweep in range(radar.nsweeps):
		start = radar.sweep_start_ray_index['data'][sweep]
		end = radar.sweep_end_ray_index['data'][sweep] + 1

		az = radar.azimuth['data'][start:end]
		time_sweep = radar.time['data'][start:end]

		# One full rotation time
		time_per_rotation = time_sweep[-1] - time_sweep[0]
		sweep_rpm = 60.0 / time_per_rotation if time_per_rotation > 0 else 0
		sweep_rpms.append(sweep_rpm)

	# Choose fixed RPM
	target_rpm = min(sweep_rpms)  # or max or mean, depending on your purpose
	
	new_time_data = radar.time['data'].copy()
	ray_ptr = 0

	for sweep in range(radar.nsweeps):
		start = radar.sweep_start_ray_index['data'][sweep]
		end = radar.sweep_end_ray_index['data'][sweep] + 1
		nsweep_rays = end - start

		# Compute new total time span for this sweep based on target RPM
		target_duration = 60.0 / target_rpm
		time_per_ray = target_duration / (nsweep_rays - 1)

		# Generate new time array for this sweep
		sweep_time = np.arange(nsweep_rays) * time_per_ray

		# Offset to maintain continuity
		if sweep == 0:
			offset = 0.0
		else:
			offset = new_time_data[ray_ptr - 1] + time_per_ray

		new_time_data[ray_ptr:ray_ptr + nsweep_rays] = sweep_time + offset
		ray_ptr += nsweep_rays

	# Replace time data
	radar.time['data'] = new_time_data
	return radar

def log_print(messages, add_messages = np.empty):
	now = datetime.now()
	now = now.strftime('%d%m%y')
	#fi = open('/etc/pyiris/log/pyiris_{}.log'.format(now), 'a')
	fi = open('/etc/pyiris/log/pyiris.log'.format(now), 'a')
	if add_messages == np.empty: 
		all_messages = messages +'\n'
		print(messages)
	else:
		all_messages = messages + np.array2string(np.array(add_messages)) +'\n'
		print(messages,add_messages)
	fi.write(all_messages)
	fi.close()

def mdfill(data):
	#data = mean_filter(data,3)
	#data = median_filter(data,size = 3)
	m,n = data.shape
	np.ma.set_fill_value(data, np.nan)
	data[data >= 327] = np.nan
	data[data <= -327] = np.nan
	nanindex = np.where(np.isnan(data))
	print(nanindex)
	nanrow = nanindex[0]
	nancol = nanindex[1]

	for r,c in zip(nanrow,nancol):
		valid_n=0
		ne = []
		if r != 0 and c !=0 and r != m-1 and c != n-1:
			ne.append(data[r-1][c-1]) 
			ne.append(data[r-1][c]  )
			ne.append(data[r-1][c+1])
			ne.append(data[r][c-1]  )
			ne.append(data[r][c+1]  )
			ne.append(data[r+1][c-1])
			ne.append(data[r+1][c]  )
			ne.append(data[r+1][c+1])
			ne = np.array(ne)
			nan_ne = np.count_nonzero(np.isnan(ne))
			
			valid_n = ne.size - nan_ne
			if(valid_n >= 6):
				data[r][c] = np.nanmean(ne)
	return data

def lee_filter(img, size):
	img_mean = uniform_filter(img, (size, size))
	img_sqr_mean = uniform_filter(img**2, (size, size))
	img_variance = img_sqr_mean - img_mean**2
	overall_variance = variance(img)
	img_weights = img_variance**2 / (img_variance**2 + overall_variance**2)
	img_output = img_mean + img_weights * (img - img_mean)
	return img_output

def box_kernel(size):
	k = np.ones((size, size), np.float32) / (size ** 2)
	return k

def getnowtime():
	now	 = datetime.now()
	now_str = now.strftime("%Y-%m-%d %H:%M:%S")
	return now_str

def string2arrayint64(s,ls):
	a = np.zeros(ls, dtype='int64')
	i = 0
	for c in s:
		a[i] = ord(c)
		i+=1
	return(a)

def std_deviation(arr):
	return np.std(arr)

def std_mean(arr):
	return np.mean(arr)

def sdev_filter(array_in,window_size):
	std_dev_array = generic_filter(array_in, std_deviation, size=window_size, mode='nearest')
	return std_dev_array

def mean_filter(array_in,window_size):
	std_dev_array = generic_filter(array_in, std_mean, size=window_size, mode='nearest')
	return std_dev_array

def ndwi(data, max_iter=100, threshold=0.1):
	data_min = np.nanmin(data)
	data_max = np.nanmax(data)
	data = exposure.rescale_intensity(data, in_range=(data_min,data_max), out_range=(0,255)).astype(np.uint8)
	prev_data = data
	for i in range(max_iter):
		filtered_data = cv2.fastNlMeansDenoising(data)
		mad = np.abs(data - filtered_data).mean()
		if mad < threshold or np.var(filtered_data) < 0.01:
			break
		prev_data = data
		data = filtered_data
	filtered_data = np.interp(filtered_data, [0,255], [data_min, data_max])

	return filtered_data

def choose_moment(all_moment,std_filter):
	std_moments = []
	if std_filter != ['']:
		for moment in std_filter:
			for m in all_moment:
				if moment in m:
					std_moments.append(m)
	std_moments = emoment_check(std_moments,std_filter,"DBZE")
	#std_moments = emoment_check(std_moments,std_filter,"DBZV")
	std_moments = emoment_check(std_moments,std_filter,"DBTE")
	std_moments = emoment_check(std_moments,std_filter,"VELC")
	return std_moments

def emoment_check(inmoments,select_moment, check_moment):
	moments = list(dict.fromkeys(inmoments))
	config_moment = get_close_matches(check_moment,select_moment)
	result_moment = get_close_matches(check_moment,moments)
	cm_status = False
	for cm in config_moment:
		if check_moment in cm:
			cm_status = True
	rm_status = False
	for rm in result_moment:
		if check_moment in rm:
			rm_status = True
	if not(cm_status and rm_status):
		for rm in result_moment:
			if check_moment in rm:
				moments.remove(rm)
	return moments

config_file = '/etc/pyiris/pyiris.conf' 

# --- Phase 1: only config ---
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, add_help=False)
parser.add_argument("-c", "--config", default=config_file, help="config file")

# parse only known args so we can load config
known_args, remaining_argv = parser.parse_known_args()

config = configparser.ConfigParser()
config.read(known_args.config)

# --- Phase 2: full parser with help ---
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

# re-add config so it shows in --help
parser.add_argument("-c", "--config", default=config_file, help="config file")

input_file	 = config["FILE"]["input_file"]	
output_file	= config["FILE"]["output_file"]	  
delete_file	= config["FILE"]["delete"]	  
mode		   = config["FILENAME"]["mode"]	   
manual_taskname= config["FILENAME"]["manual_taskname"]
select_moment  = config["PARAMETER"]["moment_in"].split(',')
save_moment	= config["PARAMETER"]["moment_out"].split(',')
delta_hours	= int(config["DELTATIME"]["delta_hours"])  
delta_minutes  = int(config["DELTATIME"]["delta_minutes"])
delta_seconds  = int(config["DELTATIME"]["delta_seconds"])
logz		   = float(config["FILTER"]["th_LOGZ"])	
loge		   = float(config["FILTER"]["th_LOGE"])	
logv		   = float(config["FILTER"]["th_LOGV"])
logw		   = float(config["FILTER"]["th_LOGW"])
logd		   = float(config["FILTER"]["th_LOGD"])	
sqiz		   = float(config["FILTER"]["th_SQIZ"])	
sqie		   = float(config["FILTER"]["th_SQIE"])	
sqiv		   = float(config["FILTER"]["th_SQIV"])	
sqiw		   = float(config["FILTER"]["th_SQIW"])	
sqid		   = float(config["FILTER"]["th_SQID"])	
pmiz		   = float(config["FILTER"]["th_PMIZ"])	
pmie		   = float(config["FILTER"]["th_PMIE"])	
pmiv		   = float(config["FILTER"]["th_PMIV"])   
pmiw		   = float(config["FILTER"]["th_PMIW"])   
pmid		   = float(config["FILTER"]["th_PMID"])   
csrz		   = float(config["FILTER"]["th_CSRZ"])	
csre		   = float(config["FILTER"]["th_CSRE"])	
csrv		   = float(config["FILTER"]["th_CSRV"])   
csrw		   = float(config["FILTER"]["th_CSRW"])   
csrd		   = float(config["FILTER"]["th_CSRD"])   
snlgz		  = float(config["FILTER"]["th_SNLGZ"])	
snlge		  = float(config["FILTER"]["th_SNLGE"])	
snlgv		  = float(config["FILTER"]["th_SNLGV"])   
snlgw		  = float(config["FILTER"]["th_SNLGW"])   
snlgd		  = float(config["FILTER"]["th_SNLGD"])	  
snrd		  = float(config["FILTER"]["th_SNRD"])   
phid1		  = float(config["FILTER"]["th_PHID1"])	 
phid2		  = float(config["FILTER"]["th_PHID2"])   
sdz  		  = float(config["FILTER"]["th_SDZ"])	 
mdz 		  = float(config["FILTER"]["th_MDZ"])   
windowSize	 = int(config["FILTER"]["me_windowSize"])
econvert		= config["FILTERMODE"]["econvert"]		
speckle_filter  = config["FILTERMODE"]["speckle_filter"] 
standard_filter = config["FILTERMODE"]["standard_filter"] 
speckle_type	= config["FILTERTYPE"]["speckle"].split(',') 
spec_filter	  = config["FILTERTYPE"]["speckle_moment"].split(',')
log_filter	  = config["FILTERTYPE"]["LOG"].split(',')
sqi_filter	  = config["FILTERTYPE"]["SQI"].split(',') 
pmi_filter	  = config["FILTERTYPE"]["PMI"].split(',') 
csr_filter	  = config["FILTERTYPE"]["CSR"].split(',') 
snlg_filter	  = config["FILTERTYPE"]["SNLG"].split(',') 
phi_filter	  = config["FILTERTYPE"]["PHI"].split(',') 
sdz_filter	  = config["FILTERTYPE"]["SDZ"].split(',') 
mdz_filter	  = config["FILTERTYPE"]["MDZ"].split(',') 
index_moment	= int(config["GRAPH"]["index_moment"])
index_sweep	 = int(config["GRAPH"]["index_sweep"])
graph_mode	  = config["GRAPH"]["graph_mode"]
moment_e		= config["ECONVERT"]["moment_e"]
ecorrection		= config["ECONVERT"]["ecorrection"]
ecomposite		= config["ECONVERT"]["ecomposite"]
mod_enable		= config["MOD"]["mod"]
interp_enable		= config["MOD"]["interpolate"]
rpm_mod_enable		= config["MOD"]["rpm_mod"]
target_rays	 = int(config["MOD"]["target_rays"])

# Parse command line arguments

parser.add_argument("-i", "--input", default=input_file, help="vaisala raw input file path")
parser.add_argument("-o", "--output", default=output_file, help="vaisala raw output file path")
parser.add_argument("-g", "--graph", default=0, type=int, help="display graph after converting, put 1 for enable")
parser.add_argument("-s", "--sweep", default=index_sweep, type=int, help="index of sweep for display graph")
parser.add_argument("-m", "--moment", default=index_moment, type=int, help="index of moment for display graph")
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
parser.add_argument("--snr", default=snrd, type=float, help="filter parameter")
parser.add_argument("--phi1", default=phid1, type=float, help="filter parameter")
parser.add_argument("--phi2", default=phid2, type=float, help="filter parameter")
parser.add_argument("--sdz", default=sdz, type=float, help="filter parameter")
parser.add_argument("--mdz", default=mdz, type=float, help="filter parameter")
parser.add_argument("--windowSize", default=windowSize  , type=int, help="speckle median filter parameter")
parser.add_argument("--speckletype", default=speckle_type,  help="speckle filter type : median or mean or ndwi")
parser.add_argument("--econvert"  , default=econvert	   ,  help="econvert Th/Tv Zh/Zv replace by Te Ze enable or disable")
parser.add_argument("--speckle"   , default=speckle_filter ,  help="speckle filter enable or disable")
parser.add_argument("--stdfilter" , default=standard_filter,  help="standard filter enable or disable")
parser.add_argument("--deletefile" , default=delete_file,  help="delete input file after converting enable or disable")
parser.add_argument("--modetaskname" , default=mode,  help="output taskname raw or auto or manual")
parser.add_argument("--taskname" , default=manual_taskname,  help="output taskname on manual mode")
parser.add_argument("--econverttype" , default=moment_e,  help="econvert type H or V")
parser.add_argument("--ecorrection" , default=ecorrection,  help="ecorrection disable or enable")
parser.add_argument("--ecomposite" , default=ecomposite,  help="ecomposite disable or enable")
parser.add_argument("--graphmode" , default=graph_mode,  help="graphic mode base, map, or multi")
parser.add_argument("--mod" , default=mod_enable,  help="mod hdf5 file into vaisala raw")
parser.add_argument("--intp" , default=interp_enable,  help="interpolation missing rays")
parser.add_argument("--rpm" , default=rpm_mod_enable,  help="change rpm with same value")
parser.add_argument("--targetrays" , default=target_rays, type=int, help="number of rays target to fill missing rays")
#parser.add_argument("--gui" , default="disable",  help="gui iteraction tools")

args = vars(parser.parse_args())

# Set up parameters
input_file = args["input"]
output_file = args["output"]
mode = args["modetaskname"]
manual_taskname = args["taskname"]
graph = args["graph"]
index_moment = args["moment"]
index_sweep = args["sweep"]
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
snrd = args["snr"]
phid1 = args["phi1"]
phid2 = args["phi2"]
sdz = args["sdz"]
mdz = args["mdz"]
moment_e = args["econverttype"]
ecorrection = args["ecorrection"]
ecomposite = args["ecomposite"]
mod_enable = args["mod"]
interp_enable = args["intp"]
rpm_mod_enable = args["rpm"]
target_rays = int(args["targetrays"])

windowSize		  = args["windowSize"]
speckle_type		= args["speckletype"]
econvert		= args["econvert"]
speckle_filter  = args["speckle"]
standard_filter = args["stdfilter"]
delete_file = args["deletefile"]
delta_minutes = args["deltatime"]
graph_mode = args["graphmode"]
#gui = args["gui"]

'''
if "enable" in gui: 
	show_gui()
	exit()
'''
dt = np.dtype([ ('key','int64', (64)) , ('value','int64',(32)) ])

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
	legend[seq] = (string2arrayint64(h_class,64),string2arrayint64('{}'.format(seq),32))
	seq += 1

if "RAW" in input_file:
	filenames = [input_file]
else :
	filenames = glob.glob(os.path.join(input_file,"*.RAW*"))

for filename in filenames: 
	log_print("[{}] : starting converting {}".format(getnowtime(),os.path.basename(filename)))

	try:
		hdf_name = os.path.basename(filename)
		hdf	  = h5py.File(os.path.join(output_file,hdf_name+".h5"),'w')
		hdf.attrs['Conventions'] = np.string_("ODIM_H5/V2_2")
		raw	  = radar.read_sigmet(filename, file_field_names=True, full_xhdr=True, time_ordered="full")
		
		
		if 'enable' in interp_enable: 
			log_print('[{}] : interpolation enable'.format(getnowtime()))
			raw = fill_missing_rays_by_raycount(raw, target_rays=target_rays)
		
		missing_rays_flag = check_number_of_rays(raw, target_rays)
		
		if 'enable' in rpm_mod_enable:
			log_print('[{}] : rpm mod enable'.format(getnowtime()))
			raw = rpm_check(raw)
		
		fields				= raw.fields
		
		moments = []
		for moment in select_moment:
			for m in fields:
				if moment in m:
					moments.append(m)
		#E check
		moments = emoment_check(moments,select_moment,"DBZE")
		#moments = emoment_check(moments,select_moment,"DBZV")
		moments = emoment_check(moments,select_moment,"DBTE")
		moments = emoment_check(moments,select_moment,"VELC")
				
		log_moments = choose_moment(fields,log_filter)
		sqi_moments = choose_moment(fields,sqi_filter)
		pmi_moments = choose_moment(fields,pmi_filter)
		csr_moments = choose_moment(fields,csr_filter)
		snr_moments = choose_moment(fields,snlg_filter)
		phi_moments = choose_moment(fields,phi_filter)
		sdz_moments = choose_moment(fields,sdz_filter)
		mdz_moments = choose_moment(fields,mdz_filter)
		spec_moments = choose_moment(fields,spec_filter)
		
		
		log_print('[{}] : moment available -> '.format(getnowtime()), list(fields.keys()))
		log_print('[{}] : moment selected  -> '.format(getnowtime()), moments)
		
		altitude			  = raw.altitude
		fixed_angle		   = raw.fixed_angle
		instrument_parameters = raw.instrument_parameters
		metadata			  = raw.metadata
		time				  = raw.time
		sweep_end_ray_index   = raw.sweep_end_ray_index
		sweep_start_ray_index = raw.sweep_start_ray_index
		site_name			= metadata['instrument_name']
		
		task_name			= metadata['sigmet_task_name']
		task_name			= task_name.decode("utf-8").replace(' ','')
		if 'manual' in mode: 
			task_name = manual_taskname
		elif 'auto' in mode:
			task_name = 'py' + task_name

		
		polarization		 = metadata['polarization']
		site_name			= site_name.decode("utf-8").replace(' ','')

		wavelength		   = instrument_parameters['wavelength']['data']
		NI				   = instrument_parameters['nyquist_velocity']['data']
		prf_ratio			= instrument_parameters['prt_ratio']['data']
		beamwidth_h		  = instrument_parameters['radar_beam_width_h']['data']
		beamwidth_v		  = instrument_parameters['radar_beam_width_v']['data']
		pulsewidth		   = instrument_parameters['pulse_width']['data']
		prt				  = instrument_parameters['prt']['data']
		prf				  = 1/prt 

		scan_type			= raw.scan_type
		nbins				= raw.ngates
		nsweeps				= raw.nsweeps
		nrays				= raw.nrays
		nrays_sweep		  = raw.rays_per_sweep['data']
		#print(nrays, nrays_sweep)
		latitude			 = raw.latitude['data']
		longitude			= raw.longitude['data']
		#azimuth			  = raw.azimuth['data'].round(decimals=1)
		azimuth			  = raw.azimuth['data']
		azimuth_start		= raw.azimuth['start']
		azimuth_stop		 = raw.azimuth['stop']
		
		elevation			= raw.elevation['data'].round(decimals=2)
		elevation_start	  = raw.elevation['start']
		elevation_stop	   = raw.elevation['stop']
		
		first_bin_range	  = raw.range['meters_to_center_of_first_gate']
		range_step		   = raw.range['meters_between_gates']
		a1gate			   = raw.range['a1gate']

		
		radar_altitude	   = altitude['data']

		start_scan_time	  = time['units']
		start_scan_time	  = datetime.strptime(start_scan_time, 'seconds since %Y-%m-%dT%H:%M:%SZ') + timedelta(hours = delta_hours, minutes = delta_minutes, seconds = delta_seconds)
		delta_time		   = time['data']
		#print(delta_time)
		if nsweeps == 1: scan_type = 'SCAN'
		else		   : scan_type = 'PVOL'
		

		#FILTERING
		
		index_of_dbz   = -1
		index_of_dbze  = -1
		index_of_dbt   = -1
		index_of_dbte  = -1
		index_of_vel   = -1
		index_of_velc  = -1
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
				else :
					index_of_dbt = moments.index(moment)

			if 'dbz' in moment.lower(): 
				if 'dbze' in moment.lower():
					index_of_dbze = moments.index(moment) 
				else :
					index_of_dbz = moments.index(moment)

			if 'vel' in moment.lower(): 
				if 'velc' in moment.lower(): 
					index_of_velc = moments.index(moment)
				else :
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
				snr  = fields[moments[index_of_snr]]['data']
				#np.savetxt('temp/snr.txt', snr, delimiter=',') 
				#log_print(mask_snrz)
				if index_of_phi >= 0:
					log_print('[{}] : SDPHI filter applied to {}'.format(getnowtime(),phi_moments))
					phi  = sdev_filter(fields[moments[index_of_phi]]['data'],(3,3))
					for moment in phi_moments:
						data = fields[moment]['data']
						
						#np.savetxt('temp/phidp.txt', fields[moments[index_of_phi]]['data'], delimiter=',') 
						#np.savetxt('temp/sdp.txt', phi, delimiter=',') 
						
						mask_snrd = np.logical_or(np.logical_and(snr > snrd, phi > phid1),np.logical_and(snr <= snrd, phi > phid2))
						#log_print(mask_snrz)

						data[mask_snrd] = np.nan
						fields[moment]['data'] = data
				else :
					log_print('[{}] : SDPHI filter skipped to {} , phi data unvailable'.format(getnowtime(),phi_moments))
			else :
				if index_of_phi >= 0:
					log_print('[{}] : SDPHI filter applied to {} using th2 due to snr data unvailable'.format(getnowtime(),phi_moments))
					phi  = sdev_filter(fields[moments[index_of_phi]]['data'],(3,3))
					for moment in phi_moments:
						data = fields[moment]['data']
						
						data[phi < phid2] = np.nan
						fields[moment]['data'] = data
				else :
					log_print('[{}] : SDPHI filter skipped to {} , phi data unvailable'.format(getnowtime(),phi_moments))

			log_print('[{}] : SDZ filter applied to {}'.format(getnowtime(),sdz_moments))
			for moment in sdz_moments:
				data = fields[moment]['data']
				data[np.isnan(data)] = -327
				sd  = sdev_filter(data,(3,3))
				mask_sdz = np.logical_or(sd == 0,sd > sdz)
				data[mask_sdz] = np.nan
				data[data == -327] = np.nan
				fields[moment]['data'] = data
			
			log_print('[{}] : MDZ filter applied to {}'.format(getnowtime(),mdz_moments))
			for moment in mdz_moments:
				data = fields[moment]['data']
				data[np.isnan(data)] = -327
				md  = median_filter(data, size=3)
				data[md < mdz] = np.nan
				data[data == -327] = np.nan
				fields[moment]['data'] = data

			if index_of_log >= 0:
				log_print('[{}] : LOG filter applied to {}'.format(getnowtime(),log_moments))
				for moment in log_moments:
					data = fields[moment]['data']
					log  = fields[moments[index_of_log]]['data']
					if 'DBZ' in moment:
						if 'DBZE' in moment:
							data[log < loge] = np.nan
						else :
							data[log < logz] = np.nan
					elif 'DBT' in moment:
						if 'DBTE' in moment:
							data[log < loge] = np.nan
						else :
							data[log < logz] = np.nan
					elif 'VEL' in moment:
						data[log < logv] = np.nan
					elif 'WIDTH' in moment:
						data[log < logw] = np.nan
					else :
							data[log < logd] = np.nan
					
					fields[moment]['data'] = data
			else :
				log_print('[{}] : LOG filter skipped to {} , LOG data unvailable'.format(getnowtime(),log_moments))

			if index_of_snr >= 0:
				log_print('[{}] : SNR filter applied to {}'.format(getnowtime(),snr_moments))
				
				#dbz = fields[moments[index_of_dbz]]['data']
				#dbt = fields[moments[index_of_dbt]]['data']
				#snr = (dbt - dbz) 
				snlgzt = 10*log10(10**(snlgz/10)-1)
				snlget = 10*log10(10**(snlge/10)-1)
				snlgvt = 10*log10(10**(snlgv/10)-1)
				snlgwt = 10*log10(10**(snlgw/10)-1)
				snlgdt = 10*log10(10**(snlgd/10)-1)
				for moment in snr_moments:
					data = fields[moment]['data']
					snr  = fields[moments[index_of_snr]]['data']
					
					if 'DBZ' in moment:
						if 'DBZE' in moment:
							data[snr < snlget] = np.nan
						else :
							data[snr < snlgzt] = np.nan
					elif 'DBT' in moment:
						if 'DBTE' in moment:
							data[snr < snlget] = np.nan
						else :
							data[snr < snlgzt] = np.nan
					elif 'VEL' in moment:
						data[snr < snlgvt] = np.nan
					elif 'WIDTH' in moment:
						data[snr < snlgwt] = np.nan
					else :
							data[snr < snlgdt] = np.nan
					fields[moment]['data'] = data
			else :
				log_print('[{}] : SNR filter skipped to {} , SNR data unvailable'.format(getnowtime(),snr_moments))

			if index_of_csr >= 0:
				log_print('[{}] : CSR filter applied to {}'.format(getnowtime(),csr_moments))
				
				#dbz = fields[moments[index_of_dbz]]['data']
				#dbt = fields[moments[index_of_dbt]]['data']
				#csr = (dbt - dbz) 
				
				for moment in csr_moments:
					data = fields[moment]['data']
					csr  = fields[moments[index_of_csr]]['data']
					if 'DBZ' in moment:
						if 'DBZE' in moment:
							data[csr > csre] = np.nan
						else :
							data[csr > csrz] = np.nan
					elif 'DBT' in moment:
						if 'DBTE' in moment:
							data[csr > csre] = np.nan
						else :
							data[csr > csrz] = np.nan
					elif 'VEL' in moment:
						data[csr > csrv] = np.nan
					elif 'WIDTH' in moment:
						data[csr > csrw] = np.nan
					else :
							data[csr > csrd] = np.nan
					fields[moment]['data'] = data
			else :
				log_print('[{}] : CSR filter skipped to {} , CSR data unvailable'.format(getnowtime(),csr_moments))
			
			if index_of_sqi >= 0:
				log_print('[{}] : SQI filter applied to {}'.format(getnowtime(),sqi_moments))
				for moment in sqi_moments:
					data = fields[moment]['data']
					sqi  = fields[moments[index_of_sqi]]['data']
					if 'DBZ' in moment:
						if 'DBZE' in moment:
							data[sqi < sqie] = np.nan
						else :
							data[sqi < sqiz] = np.nan
					elif 'DBT' in moment:
						if 'DBTE' in moment:
							data[sqi < sqie] = np.nan
						else :
							data[sqi < sqiz] = np.nan
					elif 'VEL' in moment:
						data[sqi < sqiv] = np.nan
					elif 'WIDTH' in moment:
						data[sqi < sqiw] = np.nan
					else :
							data[sqi < sqid] = np.nan
					fields[moment]['data'] = data
			else :
				log_print('[{}] : SQI filter skipped to {} , SQI data unvailable'.format(getnowtime(),sqi_moments))

			if index_of_pmi >= 0:
				log_print('[{}] : PMI filter applied to {}'.format(getnowtime(),pmi_moments))
				for moment in pmi_moments:
					data = fields[moment]['data']
					pmi  = fields[moments[index_of_pmi]]['data']
					if 'DBZ' in moment:
						if 'DBZE' in moment:
							data[pmi < pmie] = np.nan
						else :
							data[pmi < pmiz] = np.nan
					elif 'DBT' in moment:
						if 'DBTE' in moment:
							data[pmi < pmie] = np.nan
						else :
							data[pmi < pmiz] = np.nan
					elif 'VEL' in moment:
						data[pmi < pmiv] = np.nan
					elif 'WIDTH' in moment:
						data[pmi < pmiw] = np.nan
					else :
							data[pmi < pmid] = np.nan
					fields[moment]['data'] = data
			else :
				log_print('[{}] : PMI filter skipped to {} , PMI data unvailable'.format(getnowtime(),pmi_moments))
					
		else :
			log_print('[{}] : standard filter disable'.format(getnowtime()))
		
		if 'enable' in speckle_filter:
			log_print('[{}] : speckle {} filter enable'.format(getnowtime(),speckle_type))
			for moment in spec_moments:
				for spec in speckle_type:
					data = fields[moment]['data']
					if 'median' in spec:
						fields[moment]['data'] = np.ma.masked_array(median_filter(data, size=windowSize))
					elif 'mean' in spec:
						fields[moment]['data'] = np.ma.masked_array(cv2.filter2D(data, -1, box_kernel(windowSize)))
					elif 'stdmean' in spec:
						fields[moment]['data'] = np.ma.masked_array(mean_filter(data,windowSize))
					elif 'lee' in spec:
						fields[moment]['data'] = np.ma.masked_array(lee_filter(data, windowSize))
					elif 'ndwi' in spec:
						fields[moment]['data'] = np.ma.masked_array(ndwi(data))
					elif 'mdfill' in spec:
						fields[moment]['data'] = np.ma.masked_array(mdfill(data))
		
		else :
			log_print('[{}] : speckle filter disable'.format(getnowtime()))
		
		
		if 'enable' in ecorrection: 
			log_print('[{}] : ecorrection enable'.format(getnowtime()))
			if index_of_dbze >= 0 and index_of_zdr >=0 and index_of_rho >=0 :
				zdr = fields[moments[index_of_zdr]]['data']
				rho = fields[moments[index_of_rho]]['data']
				log_print('[{}] : DBZE found in ->'.format(getnowtime()), index_of_dbze)
				fields[moments[index_of_dbze]]['data'] += zdr/2 - 10*np.log10(rho)
			if index_of_dbte >= 0 and index_of_zdr >=0 and index_of_rho >=0:
				zdr = fields[moments[index_of_zdr]]['data']
				rho = fields[moments[index_of_rho]]['data']
				log_print('[{}] : DBTE found in ->'.format(getnowtime()), index_of_dbte)
				fields[moments[index_of_dbte]]['data'] += zdr/2 - 10*np.log10(rho)
		else :
			log_print('[{}] : ecorrection disable'.format(getnowtime()))
		
		if 'enable' in econvert: 
			log_print('[{}] : econvert enable, {} choosed'.format(getnowtime(),moment_e))
			if 'enable' in ecomposite:
				log_print('[{}] : ecomposite enable'.format(getnowtime()))
			else :
				log_print('[{}] : ecomposite disable'.format(getnowtime()))
				
			if index_of_dbz >= 0 and index_of_dbze >= 0:
				log_print('[{}] : DBZ and DBZE found in ->'.format(getnowtime()),[index_of_dbz, index_of_dbze])
				if 'H' in moment_e:
					if 'enable' in ecomposite:
						data_Z  = fields[moments[index_of_dbz]]['data']
						data_ZE = fields[moments[index_of_dbze]]['data']
						data_Z[np.isnan(data_Z)] = -327
						data_ZE[np.isnan(data_ZE)] = -327
						data = np.maximum(data_Z,data_ZE)
						data[data == -327] = np.nan
						fields[moments[index_of_dbze]]['data'] = data
					else :
						fields[moments[index_of_dbze]]['data'] = fields[moments[index_of_dbze]]['data']
				elif 'V' in moment_e:
					if 'enable' in ecomposite:
						data_Z  = fields[moments[index_of_dbz]]['data']
						data_ZE = fields[moments[index_of_dbze]]['data']
						data_Z[np.isnan(data_Z)] = -327
						data_ZE[np.isnan(data_ZE)] = -327
						data = np.maximum(data_Z,data_ZE)
						data[data == -327] = np.nan
						fields[moments[index_of_dbze]]['data'] = data

						
			if index_of_dbt >= 0 and index_of_dbte >= 0:
				log_print('[{}] : DBT and DBTE found in ->'.format(getnowtime()),[index_of_dbt, index_of_dbte])
				if 'H' in moment_e:
					if 'enable' in ecomposite:
						data_T  = fields[moments[index_of_dbt]]['data']
						data_TE = fields[moments[index_of_dbte]]['data']
						data_T[np.isnan(data_T)] = -327
						data_TE[np.isnan(data_TE)] = -327
						data = np.maximum(data_T,data_TE)
						data[data == -327] = np.nan
						fields[moments[index_of_dbte]]['data'] = data
					else :
						fields[moments[index_of_dbte]]['data'] = fields[moments[index_of_dbte]]['data']
				elif 'V' in moment_e:
					if 'enable' in ecomposite:
						data_T  = fields[moments[index_of_dbt]]['data']
						data_TE = fields[moments[index_of_dbte]]['data']
						data_T[np.isnan(data_T)] = -327
						data_TE[np.isnan(data_TE)] = -327
						data = np.maximum(data_T,data_TE)
						data[data == -327] = np.nan
						fields[moments[index_of_dbte]]['data'] = data

		else :
			log_print('[{}] : econvert disable'.format(getnowtime()))
		
		hcl_data = fields[moments[index_of_hcl]]['data']
		#log_print(hcl_data)
		hcl_data = (np.bitwise_and(hcl_data.astype(int), ~(0b11111 << 3)))
		fields[moments[index_of_hcl]]['data'] = hcl_data
		#log_print(hcl_data)
		#END OF FILTERING
		
		#OFFSET AND GAIN
		offset = {}
		gain = {}
		for moment in moments:
			data   = fields[moment]['data'].data
			offset[moment] = np.nanmin(data)
			if 'rhohv' in moment.lower() or 'phidp' in moment.lower() or 'sqi' in moment.lower()  or 'pmi' in moment.lower():
				gain[moment] = abs(offset[moment]) 
			elif 'class' in moment.lower():
				gain[moment] = 1.0
				offset[moment] = 0.0 
			else:
				gain[moment] = 0.01
				
			if isnan(gain[moment]) or gain[moment]==0:
				gain[moment] = 1.0
			if isnan(offset[moment]):
				offset[moment] = 0.0
		#print(gain)
		#print(offset)
		
		moments_out = []
		for moment in save_moment:
			for m in fields:
				if moment in m:
					moments_out.append(m)
		#E check
		moments_out = emoment_check(moments_out,save_moment,"DBZE")
		moments_out = emoment_check(moments_out,save_moment,"DBTE")
		moments_out = emoment_check(moments_out,save_moment,"VELC")
		
		log_print('[{}] : moment saved  ->'.format(getnowtime()), moments_out)
		
		for sweep in range(nsweeps):
			start_ray = raw.get_start(sweep)
			end_ray   = raw.get_end(sweep) 
			slice_ray = raw.get_slice(sweep)
			
			d = 1
			for moment in moments_out:
				#print('[{}] : moment {} gain {} offset {}'.format(getnowtime(),moment, gain, offset))
				data   = fields[moment]['data'].data[slice_ray]
				
				
				#print(data.shape)
				#log_print(azimuth[slice_ray])
				
				
				df	 = pd.DataFrame(data)
				dataset = {
				'azimuth_start' : azimuth_start[slice_ray],
				'azimuth_stop'  : azimuth_stop[slice_ray]
				}
				
				polar_dataset = pd.DataFrame(dataset) 
				polar_dataset = pd.concat([df,polar_dataset],axis = 1)

				polar_dataset = polar_dataset.sort_values(by=['azimuth_stop'])
				az_start	  = polar_dataset['azimuth_start'].to_numpy()
				az_stop	   = polar_dataset['azimuth_stop'].to_numpy()
				polar_dataset = polar_dataset.drop(columns = ['azimuth_start'])
				polar_dataset = polar_dataset.drop(columns = ['azimuth_stop'])
				data = polar_dataset.to_numpy()
				
				data   = (data - offset[moment])/gain[moment]
				data   = data.astype(np.uint16)
				main   = hdf.create_group("dataset{}/data{}".format(sweep+1,d))
				main.create_dataset("data", data=data, chunks=data.shape, compression="gzip")
				dset						= hdf.get("dataset{}/data{}/data".format(sweep+1,d))
				dset.attrs['CLASS']		 = np.string_('IMAGE')
				dset.attrs['IMAGE_VERSION'] = np.string_('1.2')
				
				if 'class' in moment.lower():
					main.create_dataset("legend", (26,), data = legend, dtype=dt)
					
				hdf.create_group("dataset{}/data{}/what".format(sweep+1,d))
				dset = hdf.get("dataset{}/data{}/what".format(sweep+1,d))
				if 'dbt' in moment.lower():
					if 'dbte' in moment.lower():
						if 'enable' in econvert: 
							if 'H' in moment_e : 
								dset.attrs['quantity'] = np.string_("TX")
							elif 'V' in moment_e:
								dset.attrs['quantity'] = np.string_("TV")
						else:
							dset.attrs['quantity'] = np.string_("TX")
					elif 'dbzv' in moment.lower():
						dset.attrs['quantity'] = np.string_("TV")
					else :
						dset.attrs['quantity'] = np.string_("TH")
				elif 'dbz' in moment.lower(): 
					if 'dbze' in moment.lower(): 
						if 'enable' in econvert: 
							if 'H' in moment_e:
								dset.attrs['quantity'] = np.string_("DBZX")
							elif 'V' in moment_e:
								dset.attrs['quantity'] = np.string_("DBZV")
						else:
							dset.attrs['quantity'] = np.string_("DBZX")
					elif 'dbzv' in moment.lower():
						dset.attrs['quantity'] = np.string_("DBZV")
					else :
						dset.attrs['quantity'] = np.string_("DBZH")
				elif 'vel' in moment.lower():
					if 'velc' in moment.lower():
						dset.attrs['quantity'] = np.string_("VRADDH")
					else:
						dset.attrs['quantity'] = np.string_("VRADH")
				elif 'width' in moment.lower(): 
					dset.attrs['quantity'] = np.string_("WRADH")
				elif 'zdr' in moment.lower(): 
					dset.attrs['quantity'] = np.string_("ZDR")
				elif 'phidp' in moment.lower(): 
					dset.attrs['quantity'] = np.string_("PHIDP")
				elif 'rhohv' in moment.lower(): 
					dset.attrs['quantity'] = np.string_("RHOHV")
				elif 'kdp' in moment.lower(): 
					dset.attrs['quantity'] = np.string_("KDP")
				elif 'sqi' in moment.lower(): 
					dset.attrs['quantity'] = np.string_("SQIH")
				elif 'snr' in moment.lower(): 
					dset.attrs['quantity'] = np.string_("SNRH")
				elif 'class' in moment.lower(): 
					dset.attrs['quantity'] = np.string_("CLASS")
				else:
					dset.attrs['quantity'] = np.string_(moment)
				d += 1
				dset.attrs['gain']	  = np.double(gain[moment])
				dset.attrs['offset']	= np.double(offset[moment])
				dset.attrs['nodata']	= np.double(0.0)
				dset.attrs['undetect']  = np.double(0.0)

			hdf.create_group("dataset{}/what".format(sweep+1))
			dset					= hdf.get("dataset{}/what".format(sweep+1))
			dset.attrs['product']   = np.string_('SCAN')
			sweep_start_time		= start_scan_time + timedelta(seconds = (delta_time[start_ray]))
			starttime			   = sweep_start_time.strftime("%H%M%S")
			startdate			   = sweep_start_time.strftime("%Y%m%d")
			dset.attrs.create('startdate', startdate, None, dtype='<S9')
			dset.attrs.create('starttime', starttime, None, dtype='<S7')
			sweep_end_datetime	  = start_scan_time + timedelta(seconds = (delta_time[end_ray]))
			endtime				 = sweep_end_datetime.strftime("%H%M%S")
			enddate				 = sweep_end_datetime.strftime("%Y%m%d")
			scan_time = start_scan_time.timestamp() + delta_time[slice_ray]
			dset.attrs.create('enddate', enddate, None, dtype='<S9')
			dset.attrs.create('endtime', endtime, None, dtype='<S7')
			#print(sweep_end_datetime , sweep_start_time, start_ray, end_ray)
			long_scan = sweep_end_datetime - sweep_start_time
			T = 360/int(long_scan.total_seconds())
			rpm = np.single(T/6).round(decimals=1)
			
			hdf.create_group("dataset{}/how".format(sweep+1))
			dset					  = hdf.get("dataset{}/how".format(sweep+1))
			dset.attrs['scan_index']  = sweep+1
			dset.attrs['pulsewidth']  = np.double(pulsewidth[start_ray]*1e6)
			dset.attrs['lowprf']	  = np.double(prf[start_ray]/prf_ratio[start_ray])
			dset.attrs['highprf']	 = np.double(prf[start_ray])
			dset.attrs['NI']		  = np.double(NI[start_ray])
			dset.attrs['rpm']		 = rpm
			dset.attrs['astart']	  = az_start[0].astype('float64') 
			dset.attrs['startazA']	= az_start.astype('float64') 
			dset.attrs['stopazA']	 = az_stop.astype('float64') 
			dset.attrs['startazT']	= scan_time 
			dset.attrs['stopazT']	 = scan_time
			dset.attrs['startelT']	= scan_time 
			dset.attrs['stopelT']	 = scan_time
			dset.attrs['startelA']	= elevation[slice_ray].astype('float64')
			dset.attrs['stopelA']	 = elevation[slice_ray].astype('float64')
			
			
			hdf.create_group("dataset{}/where".format(sweep+1))
			dset				 = hdf.get("dataset{}/where".format(sweep+1))
			
			dset.attrs['elangle']= np.single(fixed_angle['data'][sweep].round(decimals=1))
			dset.attrs['nbins']  = np.int_(nbins)
			dset.attrs['rstart'] = np.double(first_bin_range[0])
			dset.attrs['rscale'] = np.double(range_step[0])
			if 'enable' in interp_enable:
				dset.attrs['nrays']  = np.int_(np.nanmax(nrays_sweep))
			else:
				dset.attrs['nrays']  = np.int_(nrays_sweep[sweep])
			dset.attrs['a1gate'] = np.int_(a1gate)

		hdf.create_group("how")
		dset					  = hdf.get("how")
		dset.attrs['task']		= np.string_(task_name)
		dset.attrs['beamwidth']   = np.double(beamwidth_h[0])
		dset.attrs['polarization']= np.string_(polarization)
		dset.attrs['scan_count']  = np.int_(nsweeps)
		dset.attrs['wavelength']  = np.double(wavelength[0]/100)
		dset.attrs['azmethod']	= np.string_("AVERAGE")
		dset.attrs['binmethod']	= np.string_("AVERAGE")

		hdf.create_group("what")
		dset					  = hdf.get("what")
		dset.attrs['object']	  = np.string_(scan_type)
		dset.attrs['version']	 = np.string_('H5rad 2.2')
		ingest_start_time		 = start_scan_time
		ingesttime				 = ingest_start_time.strftime("%H%M%S")
		ingestdate				 = ingest_start_time.strftime("%Y%m%d")
		dset.attrs['date']		= np.string_(ingestdate)
		dset.attrs['time']		= np.string_(ingesttime)
		dset.attrs['source']	  = np.string_("PLC:{}".format(site_name))

		hdf.create_group("where")
		dset					  = hdf.get("where")
		dset.attrs['lon']		 = np.double(longitude[0])
		dset.attrs['lat']		 = np.double(latitude[0])
		dset.attrs['height']	  = np.double(radar_altitude[0])
		
		hdf.close()
		sleep(2)
		log_print("[{}] : finished converting {}".format(getnowtime(),filename))
		log_print("[{}] : saving file hdf5 to {}/{}.h5".format(getnowtime(),output_file,os.path.basename(filename)))
		
		if graph :
			plotfilename = "{}/{}.h5".format(output_file,os.path.basename(filename))
			if 'base' in graph_mode:
				plot_basic_ppi(plotfilename, index_moment, index_sweep)
			elif 'map' in graph_mode:
				plot_ppi(plotfilename, index_moment, index_sweep)
			elif 'multi' in graph_mode:
				plot_multi_ppi(plotfilename, index_moment, index_sweep)
				
		#mod_enable = True
		
		if 'enable' in mod_enable:
			if 'enable' not in missing_rays_flag:
				log_print("[{}] : modify {}".format(getnowtime(),filename))
				os.system("/etc/pyiris/hdf52mod/hdf52mod -i {}.h5 -o {}".format(os.path.basename(filename),os.path.basename(filename)))
			else :
				log_print("[{}] : mod force stop due to missing rays".format(getnowtime()))
			
		if 'enable' in delete_file:
			os.remove(filename)

	except Exception as e:
		msg_error = traceback.format_exc()
		log_print("[{}] : error {}\n\n".format(getnowtime(),msg_error))
#'''
