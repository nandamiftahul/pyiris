#! /usr/bin/python3
import os, warnings
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
from matplotlib.widgets import Slider
import numpy as np
os.environ["PYART_QUIET"] = "1"
import pyart
from datetime import datetime 
from mpl_toolkits.basemap import Basemap 
from math import ceil

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning)

def getnowtime():
	now     = datetime.now()
	now_str = now.strftime("%Y-%m-%d %H:%M:%S")
	return now_str

def plot_ppi(filename, index_moment, sweep): 

	radar = pyart.aux_io.read_odim_h5(filename, file_field_names=False)
	display = pyart.graph.RadarMapDisplayBasemap(radar)

	site_name    = radar.metadata['source'].split(':')[1]
	elevation = radar.fixed_angle['data']
	start_scan_time      = radar.time['units']
	start_scan_time      = datetime.strptime(start_scan_time, 'seconds since %Y-%m-%dT%H:%M:%SZ')
	scan_time    = start_scan_time.strftime('%y%m%d-%H%M')
	fields       = radar.fields
	moments      = list(fields.keys())
	moment_print = ""
	i = 0
	for mm in moments:
		moment_print += ", {} : {}".format(i, mm)
		i+=1
	sweep_print = ""
	i = 0
	for sw in elevation:
		sweep_print += ", {} : {:.1f}".format(i, sw)
		i+=1
		
	print('[{}] : index moment for plot {}'.format(getnowtime(),moment_print))
	print('[{}] : index sweep for plot {}'.format(getnowtime(),sweep_print))
	#moment       = 'reflectivity_horizontal'
	#sweep        = 0
	lat = radar.latitude["data"][0]
	lon = radar.longitude["data"][0]
	moment = moments[index_moment]
	margin = 2.5

	lat_min = lat - margin 
	lat_max = lat + margin 
	lon_min = lon - margin
	lon_max = lon + margin

	n_grid = 3
	step_lon = (lon_max - lon_min)/n_grid
	step_lat = (lat_max - lat_min)/n_grid

	plot_title = '{}-{}\n{} {:.1f} degree'.format(site_name,scan_time,moment,elevation[sweep])

	m=Basemap(llcrnrlat=lat_min,urcrnrlat=lat_max,\
				llcrnrlon=lon_min,urcrnrlon=lon_max,\
				resolution="i",projection='cass',lat_0=lat,lon_0=lon,)

	fig = plt.figure(layout = 'constrained', dpi = 100)
	display.plot_ppi_map(
		moment,
		sweep,
		min_lon=lon_min,
		max_lon=lon_max,
		min_lat=lat_min,
		max_lat=lat_max,
		lon_lines=np.arange(lon_min, lon_max, step_lon),
		lat_lines=np.arange(lat_min, lat_max, step_lat),
		fig=fig,
		cmap = 'turbo',
		title = plot_title,
		basemap = m,
		vmin = 0,
		vmax = 70
	)
	
	m.drawcoastlines()

	#plt.grid()
	plt.show()

def plot_basic_ppi(filename, index_moment, sweep): 
	radar = pyart.aux_io.read_odim_h5(filename, file_field_names=False)	
	elevation = radar.fixed_angle['data']
	fields       = radar.fields
	moments      = list(fields.keys())
	moment = moments[index_moment]
	
	moment_print = ""
	i = 0
	for mm in moments:
		moment_print += ", {} : {}".format(i, mm)
		i+=1
	sweep_print = ""
	i = 0
	for sw in elevation:
		sweep_print += ", {} : {:.1f}".format(i, sw)
		i+=1
		
	print('[{}] : index moment for plot {}'.format(getnowtime(),moment_print))
	print('[{}] : index sweep for plot {}'.format(getnowtime(),sweep_print))
	# Plot the Reflectivity Field (corrected_reflectivity_horizontal)
	fig = plt.figure(figsize=(15, 5))
	display = pyart.graph.RadarDisplay(radar)
	display.plot(moment, sweep, cmap="pyart_ChaseSpectral", vmin=-20, vmax=70)
	#plt.grid()
	plt.show()

def plot_multi_ppi(filename, index_moment, sweep): 
	radar = pyart.aux_io.read_odim_h5(filename, file_field_names=False)	
	site_name    = radar.metadata['source'].split(':')[1]
	elevation = radar.fixed_angle['data']
	start_scan_time      = radar.time['units']
	start_scan_time      = datetime.strptime(start_scan_time, 'seconds since %Y-%m-%dT%H:%M:%SZ')
	scan_time    = start_scan_time.strftime('%y%m%d-%H%M')
	fields       = radar.fields
	moments      = list(fields.keys())
	#moment = moments[index_moment]
	
	moment_print = ""
	i = 0
	for mm in moments:
		moment_print += ", {} : {}".format(i, mm)
		i+=1
	sweep_print = ""
	i = 0
	for sw in elevation:
		sweep_print += ", {} : {:.1f}".format(i, sw)
		i+=1
		
	print('[{}] : index moment for plot {}'.format(getnowtime(),moment_print))
	print('[{}] : index sweep for plot {}'.format(getnowtime(),sweep_print))

	# create the plot
	fig = plt.figure(figsize=(15, 5))
	
	display = pyart.graph.RadarDisplay(radar)
	
	num = len(moments)
	col = ceil(num/3)
	row = int(num/col)
	
	print(num,col,row)
	
	pos = []
	i = 1
	for c in range(1,col+1):
		for r in range(1,row+1):
			pos.append(col*100+row*10+i)
			i+=1
	#print(pos)
	for moment,ip in zip(moments,pos):
		ax = fig.add_subplot(ip)
		display.plot(
			moment,
			sweep,
			ax=ax,
			colorbar_label="",
			title= moment,
		)

	plot_title = '{}-{}\n {:.1f} degree'.format(site_name,scan_time,elevation[sweep])
	plt.suptitle(plot_title, fontsize=16)
	plt.show()

