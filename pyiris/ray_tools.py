# pyiris/ray_tools.py
import numpy as np
import copy
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter

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
