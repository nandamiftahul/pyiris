# pyiris/filters.py
import numpy as np
import cv2
from skimage import exposure
from scipy.ndimage import generic_filter, median_filter, uniform_filter, variance

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
