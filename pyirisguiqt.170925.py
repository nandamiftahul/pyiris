#! /usr/bin/python3

import sys, os, glob, subprocess
os.environ["PYART_QUIET"] = "1"
import numpy as np, pyart, matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.ndimage import median_filter, gaussian_filter, generic_filter, uniform_filter
import cv2

class RadarViewer(QtWidgets.QMainWindow):
	def __init__(self):
		super().__init__()
		self.setWindowTitle("Radar Viewer")
		#self.resize(1650, 980)

		# ---- Matplotlib figure/canvas/toolbar
		self.fig = plt.Figure(figsize=(7, 7))
		self.ax = self.fig.add_subplot(111, projection="polar")
		self.canvas = FigureCanvas(self.fig)
		self.nav_toolbar = NavigationToolbar(self.canvas, self)
		self._cbar = None

		# ---- Title QLabel below toolbar
		self.title_label = QtWidgets.QLabel("No data loaded")
		self.title_label.setFrameShape(QtWidgets.QFrame.Panel)
		self.title_label.setFrameShadow(QtWidgets.QFrame.Sunken)
		self.title_label.setAlignment(QtCore.Qt.AlignCenter)
		self.title_label.setStyleSheet("font-weight: bold; font-size: 12pt; margin: 4px;")
		self.title_label.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
		self.title_label.setFixedHeight(24)   # keep label slim
		
		# ---- Top control toolbar
		self.ctrl = QtWidgets.QToolBar("Controls")
		self.addToolBar(self.ctrl)

		# Moment
		self.moment_combo = QtWidgets.QComboBox()
		self.ctrl.addWidget(QtWidgets.QLabel("  Moment:"))
		self.ctrl.addWidget(self.moment_combo)
		self.moment_combo.currentIndexChanged.connect(self.on_moment_change)

		# Product (now includes CMax/CAPPI/VIL/EchoTops)
		self.product_combo = QtWidgets.QComboBox()
		self.product_combo.addItems(["Normal", "Max", "Min", "Mean", "StdDev", "CMax", "CAPPI", "VIL", "EchoTops"])
		self.ctrl.addSeparator()
		self.ctrl.addWidget(QtWidgets.QLabel("  Product:"))
		self.ctrl.addWidget(self.product_combo)
		self.product_combo.currentIndexChanged.connect(self.update_plot)

		# Sweep (still useful for Normal/RainRate-style single-sweep products)
		self.ctrl.addSeparator()
		self.sweep_combo = QtWidgets.QComboBox()
		self.ctrl.addWidget(QtWidgets.QLabel("  Sweep:"))
		self.ctrl.addWidget(self.sweep_combo)
		self.sweep_combo.currentIndexChanged.connect(self.on_sweep_change)

		# Colormap
		self.ctrl.addSeparator()
		self.cmap_combo = QtWidgets.QComboBox()
		self.cmap_combo.addItems(sorted(plt.colormaps()))
		self.cmap_combo.setCurrentText("RefDiff")
		self.ctrl.addWidget(QtWidgets.QLabel("  Colormap:"))
		self.ctrl.addWidget(self.cmap_combo)
		self.cmap_combo.currentIndexChanged.connect(self.update_plot)

		# vmin/vmax
		self.ctrl.addSeparator()
		self.vmin_edit = QtWidgets.QLineEdit(); self.vmin_edit.setFixedWidth(70)
		self.vmax_edit = QtWidgets.QLineEdit(); self.vmax_edit.setFixedWidth(70)
		self.ctrl.addWidget(QtWidgets.QLabel("  vmin:"))
		self.ctrl.addWidget(self.vmin_edit)
		self.ctrl.addWidget(QtWidgets.QLabel("  vmax:"))
		self.ctrl.addWidget(self.vmax_edit)
		self.btn_apply_limits = QtWidgets.QPushButton("Apply")
		self.btn_auto_limits  = QtWidgets.QPushButton("Auto")
		self.btn_reset_limits = QtWidgets.QPushButton("Reset")
		self.ctrl.addWidget(self.btn_apply_limits)
		self.ctrl.addWidget(self.btn_auto_limits)
		self.ctrl.addWidget(self.btn_reset_limits)
		self.btn_apply_limits.clicked.connect(self.update_plot)
		self.btn_auto_limits.clicked.connect(self.autoscale_limits)
		self.btn_reset_limits.clicked.connect(self.reset_limits)

		# CAPPI height / Echo Tops threshold
		self.ctrl.addSeparator()
		self.cappi_edit = QtWidgets.QLineEdit("2.0"); self.cappi_edit.setFixedWidth(70)
		self.ctrl.addWidget(QtWidgets.QLabel("  CAPPI height (km):"))
		self.ctrl.addWidget(self.cappi_edit)

		self.echotops_thr_edit = QtWidgets.QLineEdit("20.0"); self.echotops_thr_edit.setFixedWidth(70)
		self.ctrl.addWidget(QtWidgets.QLabel("  EchoTops thr (dBZ):"))
		self.ctrl.addWidget(self.echotops_thr_edit)

		# Play/pause
		self.ctrl.addSeparator()
		self.play_btn = QtWidgets.QPushButton("Play")
		self.pause_btn = QtWidgets.QPushButton("Pause")
		self.ctrl.addWidget(self.play_btn); self.ctrl.addWidget(self.pause_btn)
		self.play_btn.clicked.connect(self.play); self.pause_btn.clicked.connect(self.pause)

		# Time slider
		self.ctrl.addSeparator()
		self.ctrl.addWidget(QtWidgets.QLabel("  Time:"))
		self.time_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
		self.time_slider.setFixedWidth(420); self.time_slider.setRange(0, 0)
		self.ctrl.addWidget(self.time_slider)
		self.time_label = QtWidgets.QLabel("0/0")
		self.ctrl.addWidget(self.time_label)
		self.time_slider.valueChanged.connect(self.on_time_change)

		# ---- Central layout
		central = QtWidgets.QWidget()
		v = QtWidgets.QVBoxLayout(central)
		v.addWidget(self.nav_toolbar)
		title_row = QtWidgets.QHBoxLayout()
		title_row.addWidget(self.title_label, alignment=QtCore.Qt.AlignHCenter)
		v.addLayout(title_row)
		v.addWidget(self.canvas)
		self.setCentralWidget(central)

		# ---- Menu
		menubar = self.menuBar()
		m_file = menubar.addMenu("File")
		act_open_one = QtWidgets.QAction("Open Single File…", self)
		act_open_many = QtWidgets.QAction("Open Files (time series)…", self)
		act_exit = QtWidgets.QAction("Exit", self)
		m_file.addAction(act_open_one); m_file.addAction(act_open_many); m_file.addSeparator(); m_file.addAction(act_exit)
		act_open_one.triggered.connect(self.open_one)
		act_open_many.triggered.connect(self.open_many)
		act_exit.triggered.connect(self.close)

		# ---- Filters dock (full set)
		self.filter_dock = QtWidgets.QDockWidget("Filters", self)
		panel = QtWidgets.QWidget()
		dock_layout = QtWidgets.QVBoxLayout(panel)

		form = QtWidgets.QFormLayout()
		dock_layout.addLayout(form)

		self.thresh_entry   = QtWidgets.QLineEdit(); self.thresh_entry.setFixedWidth(80)
		self.sqi_entry      = QtWidgets.QLineEdit(); self.sqi_entry.setFixedWidth(80)
		self.pmi_entry      = QtWidgets.QLineEdit(); self.pmi_entry.setFixedWidth(80)
		self.rhohv_entry    = QtWidgets.QLineEdit(); self.rhohv_entry.setFixedWidth(80)
		self.phidp_entry    = QtWidgets.QLineEdit(); self.phidp_entry.setFixedWidth(80)
		self.vel_min_entry  = QtWidgets.QLineEdit(); self.vel_min_entry.setFixedWidth(80)
		self.vel_max_entry  = QtWidgets.QLineEdit(); self.vel_max_entry.setFixedWidth(80)
		self.zdr_entry      = QtWidgets.QLineEdit(); self.zdr_entry.setFixedWidth(80)
		self.cp_entry       = QtWidgets.QLineEdit(); self.cp_entry.setFixedWidth(80)

		grid_entries = QtWidgets.QGridLayout()

		# Left column (col=0 label, col=1 entry)
		grid_entries.addWidget(QtWidgets.QLabel("LOG Thresh:"), 0, 0)
		grid_entries.addWidget(self.thresh_entry, 0, 1)

		grid_entries.addWidget(QtWidgets.QLabel("SQI:"), 1, 0)
		grid_entries.addWidget(self.sqi_entry, 1, 1)

		grid_entries.addWidget(QtWidgets.QLabel("PMI:"), 2, 0)
		grid_entries.addWidget(self.pmi_entry, 2, 1)

		grid_entries.addWidget(QtWidgets.QLabel("RHOHV:"), 3, 0)
		grid_entries.addWidget(self.rhohv_entry, 3, 1)

		grid_entries.addWidget(QtWidgets.QLabel("PHIDP:"), 4, 0)
		grid_entries.addWidget(self.phidp_entry, 4, 1)

		# Right column (col=2 label, col=3 entry)
		grid_entries.addWidget(QtWidgets.QLabel("VEL Min:"), 0, 2)
		grid_entries.addWidget(self.vel_min_entry, 0, 3)

		grid_entries.addWidget(QtWidgets.QLabel("VEL Max:"), 1, 2)
		grid_entries.addWidget(self.vel_max_entry, 1, 3)

		grid_entries.addWidget(QtWidgets.QLabel("ZDR:"), 2, 2)
		grid_entries.addWidget(self.zdr_entry, 2, 3)

		grid_entries.addWidget(QtWidgets.QLabel("Clutter Prob:"), 3, 2)
		grid_entries.addWidget(self.cp_entry, 3, 3)

		# Add grid to the form
		form.addRow(grid_entries)
		
		self.chk_median   = QtWidgets.QCheckBox("Median")
		self.chk_gaussian = QtWidgets.QCheckBox("Gaussian")
		self.chk_kuan     = QtWidgets.QCheckBox("Kuan")
		self.chk_frost    = QtWidgets.QCheckBox("Frost")
		self.chk_bilateral= QtWidgets.QCheckBox("Bilateral")
		self.chk_std      = QtWidgets.QCheckBox("STD Mask")

		grid = QtWidgets.QGridLayout()
		grid.addWidget(self.chk_median,    0, 0)
		grid.addWidget(self.chk_gaussian,  0, 1)
		grid.addWidget(self.chk_kuan,      1, 0)
		grid.addWidget(self.chk_frost,     1, 1)
		grid.addWidget(self.chk_bilateral, 2, 0)
		grid.addWidget(self.chk_std,       2, 1)

		form.addRow(grid)

		btns = QtWidgets.QHBoxLayout()
		apply_btn = QtWidgets.QPushButton("Apply Filters")
		reset_btn = QtWidgets.QPushButton("Reset Filters")
		btns.addWidget(apply_btn); btns.addWidget(reset_btn)
		form.addRow(btns)

		apply_btn.clicked.connect(self.update_plot)
		reset_btn.clicked.connect(self.reset_filters)

		self.filter_dock.setWidget(panel)
		self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.filter_dock)

		# Circle checkbox
		self.chk_rings = QtWidgets.QCheckBox("Show circles basemap")
		self.chk_rings.setChecked(True)
		self.chk_rings.stateChanged.connect(self.update_plot)

		# External colorbar
		self.cbar_label = QtWidgets.QLabel("No colorbar")
		self.cbar_label.setAlignment(QtCore.Qt.AlignLeft)
		self.cbar_label.setFrameShape(QtWidgets.QFrame.Panel)
		self.cbar_label.setFrameShadow(QtWidgets.QFrame.Sunken)
		#self.cbar_label.setMinimumHeight(200)

		# External note unit colorbar
		self.cbar_unit = QtWidgets.QLabel()
		self.cbar_unit.setAlignment(QtCore.Qt.AlignCenter)
		self.cbar_unit.setFrameShape(QtWidgets.QFrame.Panel)
		self.cbar_unit.setFrameShadow(QtWidgets.QFrame.Sunken)
		#self.cbar_unit.setMinimumHeight(200)

		# Grid layout for colorbar + unit
		grid_cbar = QtWidgets.QGridLayout()
		grid_cbar.addWidget(self.cbar_label, 0, 0)
		grid_cbar.addWidget(self.cbar_unit, 0, 1)

		# Wrap layout into a QWidget
		cbar_widget = QtWidgets.QWidget()
		cbar_widget.setLayout(grid_cbar)

		# Add to dock layout
		dock_layout.addWidget(cbar_widget)
		dock_layout.addWidget(self.chk_rings)
		
		self.filter_dock.setWidget(panel)


		# --- View Menu ---
		view_menu = menubar.addMenu("&View")

		self.view_group = QtWidgets.QActionGroup(self)
		self.view_group.setExclusive(True)

		polar_act = QtWidgets.QAction("Polar Plot", self, checkable=True)
		polar_act.setChecked(True)  # default
		polar_act.triggered.connect(lambda: self.set_plot_mode("polar"))
		view_menu.addAction(polar_act)
		self.view_group.addAction(polar_act)

		basemap_act = QtWidgets.QAction("Basemap Plot", self, checkable=True)
		basemap_act.triggered.connect(lambda: self.set_plot_mode("basemap"))
		view_menu.addAction(basemap_act)
		self.view_group.addAction(basemap_act)
		self.plot_mode = "polar"
		
		# Progress bar in status bar
		self.progress = QtWidgets.QProgressBar()
		self.progress.setMaximumWidth(200)
		self.progress.setVisible(False)
		self.progress.setTextVisible(True)   # show text inside bar
		self.progress.setFormat("Ready")     # initial text
		self.statusBar().addPermanentWidget(self.progress)

		#--- Basemap Colors Menu ---
		basemap_menu = menubar.addMenu("&Basemap Colors")

		land_act = QtWidgets.QAction("Set Land Color", self)
		land_act.triggered.connect(self.choose_land_color)
		basemap_menu.addAction(land_act)

		ocean_act = QtWidgets.QAction("Set Ocean Color", self)
		ocean_act.triggered.connect(self.choose_ocean_color)
		basemap_menu.addAction(ocean_act)

		# Default colors
		self.land_color = "lightgray"
		self.ocean_color = "lightblue"

		# --- Basemap Style Menu ---
		style_menu = basemap_menu.addMenu("Map Style")

		styles = {
			"Simple (Coastlines + Land/Ocean)": "simple",
			"Stamen Terrain": "stamen_terrain",
			"Stamen Toner": "stamen_toner",
			"Stamen Watercolor": "stamen_watercolor",
			"OSM": "osm",
		}

		self.basemap_style = "simple"

		self.style_group = QtWidgets.QActionGroup(self)
		self.style_group.setExclusive(True)

		for label, key in styles.items():
			act = QtWidgets.QAction(label, self, checkable=True)
			act.triggered.connect(lambda checked, k=key: self.set_basemap_style(k))
			style_menu.addAction(act)
			self.style_group.addAction(act)
			if key == "simple":
				act.setChecked(True)

		# --- Recent Files Dock ---
		self.recent_dock = QtWidgets.QDockWidget("Recent Files", self)
		self.recent_dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)

		dock_widget = QtWidgets.QWidget()
		dock_layout = QtWidgets.QVBoxLayout(dock_widget)


		# chooser for number of files
		self.file_count_combo = QtWidgets.QComboBox()
		self.file_count_combo.addItems(["10", "50", "100", "Max"])
		self.file_count_combo.setCurrentText("10")
		self.file_count_combo.currentIndexChanged.connect(self.refresh_recent_files)
		dock_layout.addWidget(self.file_count_combo)

		self.recent_list = QtWidgets.QListWidget()
		self.recent_list.setViewMode(QtWidgets.QListView.IconMode)
		self.recent_list.setFlow(QtWidgets.QListView.TopToBottom)
		self.recent_list.setIconSize(QtCore.QSize(120, 120))   # bigger thumbnail
		self.recent_list.setResizeMode(QtWidgets.QListView.Adjust)
		self.recent_list.setMovement(QtWidgets.QListView.Static)
		self.recent_list.setSpacing(10)
		self.recent_list.itemClicked.connect(self.load_recent_file)
		dock_layout.addWidget(self.recent_list)

		self.recent_dock.setWidget(dock_widget)
		self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.recent_dock)

		self.refresh_thumbs_btn = QtWidgets.QPushButton("Refresh Thumbnails")
		self.refresh_thumbs_btn.clicked.connect(self.refresh_thumbnails)
		dock_layout.addWidget(self.refresh_thumbs_btn)


		# --- Tools Menu ---
		tools_menu = menubar.addMenu("&Tools")

		convert_act = QtWidgets.QAction("Convert to HDF5", self)
		convert_act.triggered.connect(self.convert_to_hdf5)
		tools_menu.addAction(convert_act)


		# --- Theme Menu ---
		theme_menu = menubar.addMenu("&Theme")

		# Make an exclusive action group
		theme_group = QtWidgets.QActionGroup(self)
		theme_group.setExclusive(True)

		# Add available styles
		for style in QtWidgets.QStyleFactory.keys():
			act = QtWidgets.QAction(style, self, checkable=True)
			act.triggered.connect(lambda checked, s=style: self.apply_theme(s))
			theme_menu.addAction(act)
			theme_group.addAction(act)

		# Add dark theme option
		dark_act = QtWidgets.QAction("Dark Fusion", self, checkable=True)
		dark_act.triggered.connect(self.apply_dark_fusion)
		theme_menu.addAction(dark_act)
		theme_group.addAction(dark_act)

		# Set default theme checked
		for action in theme_group.actions():
			if action.text() == "Fusion":   # default
				action.setChecked(True)
				break

		# Default = Fusion (cross-platform consistent)
		QtWidgets.QApplication.setStyle("Fusion")


		# --- Default display settings (cmap, vmin, vmax) for each radar field ---
		self.default_display = {
			# Reflectivity / total power
			"DBT":   {"cmap": "RefDiff",   "vmin": 0,    "vmax": 95},
			"DBZ":   {"cmap": "RefDiff",   "vmin": 0,    "vmax": 95},

			# Velocity
			"VEL":   {"cmap": "NWSVel",    "vmin": -35,  "vmax": 35},

			# Spectrum width
			"WID":   {"cmap": "Carbone11", "vmin": 0,    "vmax": 10},

			# Differential reflectivity
			"ZDR":   {"cmap": "Carbone11", "vmin": -8,   "vmax": 8},

			# Differential phase
			"PHIDP": {"cmap": "Carbone11", "vmin": 0,    "vmax": 180},

			# Specific differential phase
			"KDP":   {"cmap": "Carbone11", "vmin": -1,   "vmax": 20},

			# Correlation coefficient / quality indicators
			"RHOHV": {"cmap": "Carbone11", "vmin": 0,    "vmax": 1},
			"SQI":   {"cmap": "Carbone11", "vmin": 0,    "vmax": 1},
			"PMI":   {"cmap": "Carbone11", "vmin": 0,    "vmax": 1},

			# Logarithmic / derived
			"LOG":   {"cmap": "Carbone11", "vmin": 0,    "vmax": 6},

			# Signal-to-Noise Ratio
			"SNR":   {"cmap": "Carbone11", "vmin": -35,  "vmax": 95},

			# Clutter Signal Ratio
			"CSR":   {"cmap": "Carbone11", "vmin": 0,    "vmax": 50},

			# Hydrometeor Classification (categorical)
			"HCLASS": {"cmap": "tab20",    "vmin": 0,    "vmax": 26},
		}

		self.moment_units = {
			"DBZ": "dBZ",
			"DBT": "dBZ",
			"VEL": "m/s",
			"WID": "m/s",
			"ZDR": "dB",
			"PHIDP": "°",
			"KDP": "°/km",
			"RHOHV": "unitless",
			"SQI": "unitless",
			"PMI": "unitless",
			"LOG": "log",
			"SNR": "dB",
			"CSR": "dB",
			"HCLASS": "class",
		}

		# ---- Data holders
		self.radars, self.files = [], []
		self.idx, self.current_sweep = 0, 0
		self.current_moment = None

		# ---- Timer
		self.timer = QtCore.QTimer(self); self.timer.setInterval(500)
		self.timer.timeout.connect(self.step)

		# ---- Initial
		th = np.linspace(0, 2*np.pi, 500); r = np.abs(np.sin(5*th))+0.5
		self.ax.plot(th, r); self.canvas.draw()
		self.statusBar().showMessage("Ready")
		
		# Maximize window on startup
		self.showMaximized()
	
	# --------- File I/O
	def open_one(self):
		f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Radar File", os.getcwd(),
													 "Radar Files (*.raw* *.h5 *.nc);;All Files (*)")
		if not f: return
		self.load_series([f])
		# Update the recent files panel to the folder of this file
		self.update_recent_files(os.path.dirname(f))

	def open_many(self):
		fs, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Open Radar Files (time series)", os.getcwd(),
													   "Radar Files (*.raw* *.h5 *.nc);;All Files (*)")
		if not fs: return
		self.load_series(sorted(fs))
		# Update the recent files panel to the folder of this file
		self.update_recent_files(os.path.dirname(fs[0]))

	def load_series(self, paths):
		self.timer.stop()
		self.files, self.radars, self.idx = paths, [], 0
		for p in self.files: self.radars.append(self.load_radar(p))

		# Moments
		self.moment_combo.blockSignals(True); self.moment_combo.clear()
		if self.radars and self.radars[0].fields:
			self.moment_combo.addItems(list(self.radars[0].fields.keys()))
			self.current_moment = self.moment_combo.currentText()
		else:
			self.current_moment = None
		self.moment_combo.blockSignals(False)

		# Sweeps for current index
		self.populate_sweeps_for_index(0)

		# Time slider
		self.time_slider.blockSignals(True)
		self.time_slider.setRange(0, max(0, len(self.radars)-1))
		self.time_slider.setValue(0)
		self.time_slider.blockSignals(False)
		self.update_time_label()

		self.reset_limits()
		self.update_plot()

	def load_radar(self, file_path):
		self.start_progress("Loading radar file...")
		try:
			ext = os.path.splitext(file_path)[1].lower()

			if ext.startswith(".raw"):
				radar = pyart.io.read_sigmet(
					file_path,
					file_field_names=True,
					full_xhdr=True,
					time_ordered="full"
				)
			elif ext == ".h5":
				radar = pyart.aux_io.read_odim_h5(file_path)
			elif ext == ".nc":
				radar = pyart.io.read(file_path)
			else:
				radar = pyart.io.read(file_path)

			# --- Fill missing rays if needed ---
			ray_counts = []
			for s in range(radar.nsweeps):
				st = radar.sweep_start_ray_index['data'][s]
				en = radar.sweep_end_ray_index['data'][s] + 1
				ray_counts.append(en - st)

			target_rays = max(ray_counts)
			if len(set(ray_counts)) > 1:   # only fix if inconsistent
				radar = self.fill_missing_rays_biggest_gap(radar, target_rays)

			return radar
		finally:
			self.stop_progress("Ready")

	def update_recent_files(self, folder):
		# Get how many files to show
		choice = self.file_count_combo.currentText()
		max_files = 150 if choice == "Max" else int(choice)

		files = sorted(
			glob.glob(os.path.join(folder, "*.RAW*")),
			key=os.path.getmtime, reverse=True
		)
		if max_files:
			files = files[:max_files]

		self.recent_list.clear()
		for f in files:
			# add item without thumbnail
			item = QtWidgets.QListWidgetItem(os.path.basename(f))
			item.setData(QtCore.Qt.UserRole, f)
			self.recent_list.addItem(item)

	def refresh_thumbnails(self):
		if self.recent_list.count() == 0:
			return

		moment = self.moment_combo.currentText() or "DBZ"
		total = self.recent_list.count()

		# Reserve space + disable updates
		self.recent_list.setUpdatesEnabled(False)

		# Setup progress bar
		self.progress.setRange(0, total)
		self.progress.setValue(0)
		self.progress.show()

		for i in range(total):
			item = self.recent_list.item(i)
			file_path = item.data(QtCore.Qt.UserRole)
			try:
				radar = self.load_radar(file_path)
				thumb = self.make_thumbnail(radar, moment)
				item.setIcon(thumb)
			except Exception:
				pass

			# Update progress bar and status
			self.progress.setValue(i + 1)
			self.statusBar().showMessage(f"Generating thumbnails {i+1}/{total}")
			QtWidgets.QApplication.processEvents()

		# Re-enable updates
		self.recent_list.setUpdatesEnabled(True)
		self.recent_list.doItemsLayout()
		self.recent_list.update()
		self.recent_list.repaint()

		# Finalize
		self.progress.hide()
		self.statusBar().showMessage("Thumbnails ready", 3000)

	def refresh_recent_files(self):
		# reload recent files for the last opened folder (if any)
		if self.files:
			folder = os.path.dirname(self.files[0])
			self.update_recent_files(folder)

	def load_recent_file(self, item: QtWidgets.QListWidgetItem):
		file_path = item.data(QtCore.Qt.UserRole)
		if not file_path:
			return

		try:
			radar = self.load_radar(file_path)
		except Exception as e:
			# show a meaningful error instead of ""
			msg = str(e) or e.__class__.__name__
			QtWidgets.QMessageBox.critical(self, "Open error",
										   f"Failed to load:\n{file_path}\n\n{msg}")
			return

		# Keep app state in sync
		self.radars = [radar]
		self.files  = [file_path]
		self.idx = 0
		self.current_sweep = 0

		# --- Moments (fields) ---
		fields = list(radar.fields.keys())
		if not fields:
			QtWidgets.QMessageBox.warning(self, "No fields", "This file has no data fields.")
			return

		# Prefer a sensible default (DBZ/DBT/reflectivity if present)
		default_moment = self._pick_default_moment(fields)

		self.moment_combo.blockSignals(True)
		self.moment_combo.clear()
		self.moment_combo.addItems(fields)
		self.moment_combo.setCurrentText(default_moment)
		self.current_moment = default_moment
		self.moment_combo.blockSignals(False)

		# Apply moment-specific defaults (cmap, vmin/vmax) to the UI
		self._apply_moment_defaults(default_moment)

		# --- Sweeps ---
		self.sweep_combo.blockSignals(True)
		self.sweep_combo.clear()
		for i in range(radar.nsweeps):
			try:
				el = float(radar.fixed_angle["data"][i])
				label = f"{el:.2f}°"
			except Exception:
				label = f"Sweep {i}"
			self.sweep_combo.addItem(label)
		self.sweep_combo.setCurrentIndex(0)
		self.current_sweep = 0
		self.sweep_combo.blockSignals(False)

		# Update title label above the plot
		base = os.path.basename(file_path)
		self.title_label.setText(f"{base}")

		# Finally draw
		self.update_plot()

	def _pick_default_moment(self, fields):
		# try nice priorities; fall back to the first field
		prio = ["DBZ", "DBT", "REF", "VEL", "ZDR", "PHIDP", "KDP", "RHOHV", "SQI"]
		upper = {f.upper(): f for f in fields}
		for p in prio:
			for uf, orig in upper.items():
				if p in uf:
					return orig
		return fields[0]

	def _apply_moment_defaults(self, moment):
		# set cmap/vmin/vmax in the UI from your default_display mapping
		fn = moment.upper()
		for key, s in self.default_display.items():
			if key in fn:
				# normalize cmap name to a valid Matplotlib name already in your combobox
				try:
					cmap_name = plt.get_cmap(s["cmap"]).name
				except Exception:
					cmap_name = self.cmap_combo.currentText()  # fallback
				self.cmap_combo.setCurrentText(cmap_name)
				self.vmin_edit.setText(str(s["vmin"]))
				self.vmax_edit.setText(str(s["vmax"]))
				break


	# --------- UI handlers
	def on_moment_change(self):
		self.current_moment = self.moment_combo.currentText()
		field_name = self.current_moment.upper()
		for key, s in self.default_display.items():
			if key in field_name:
				# normalize to Matplotlib’s canonical name
				cmap = plt.get_cmap(s["cmap"]).name
				self.cmap_combo.setCurrentText(cmap)
				self.vmin_edit.setText(str(s["vmin"]))
				self.vmax_edit.setText(str(s["vmax"]))
				
		self.update_plot()
		
	def on_sweep_change(self):
		self.current_sweep = self.sweep_combo.currentIndex()
		self.update_plot()

	def on_time_change(self, value):
		self.idx = int(value)
		self.update_time_label()
		self.populate_sweeps_for_index(self.idx)
		self.update_plot()

	def update_time_label(self):
		total = len(self.radars); self.time_label.setText(f"{self.idx+1}/{total if total else 0}")

	def play(self):
		if self.radars: self.timer.start(); self.statusBar().showMessage("Playing…")

	def pause(self):
		self.timer.stop(); self.statusBar().showMessage("Paused")

	def step(self):
		if not self.radars: self.timer.stop(); return
		self.idx = (self.idx + 1) % len(self.radars)
		self.time_slider.blockSignals(True); self.time_slider.setValue(self.idx); self.time_slider.blockSignals(False)
		self.update_time_label(); self.populate_sweeps_for_index(self.idx); self.update_plot()

	def apply_theme(self, style_name):
		QtWidgets.QApplication.setStyle(style_name)

	def apply_dark_fusion(self):
		QtWidgets.QApplication.setStyle("Fusion")
		dark_palette = QtGui.QPalette()

		dark_palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))
		dark_palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
		dark_palette.setColor(QtGui.QPalette.Base, QtGui.QColor(25, 25, 25))
		dark_palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
		dark_palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
		dark_palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
		dark_palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
		dark_palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
		dark_palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
		dark_palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
		dark_palette.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
		dark_palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
		dark_palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)

		QtWidgets.QApplication.setPalette(dark_palette)

	def set_plot_mode(self, mode):
		self.plot_mode = mode
		self.update_plot()

	def switch_to_basemap_axis(self, radar_lon, radar_lat):
		# Remove old axis
		self.fig.delaxes(self.ax)

		# Create new axis with PlateCarree projection
		self.ax = self.fig.add_subplot(111, projection=ccrs.PlateCarree())

		# Add some background features
		self.ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
		self.ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)
		self.ax.add_feature(cfeature.LAND, facecolor=self.land_color)
		self.ax.add_feature(cfeature.OCEAN, facecolor=self.ocean_color)

		# Center map around radar with ~2° buffer
		self.ax.set_extent([radar_lon - 2, radar_lon + 2,
							radar_lat - 2, radar_lat + 2])

	def start_progress(self, message="Processing data...", maximum=0):
		"""Show progress bar with a message. maximum=0 = indeterminate."""
		self.progress.setVisible(True)
		self.progress.setMaximum(maximum)
		self.progress.setValue(0)
		self.progress.setFormat(message)
		QtWidgets.QApplication.processEvents()

	def stop_progress(self, message="Ready"):
		"""Hide progress bar and reset text."""
		self.progress.setVisible(False)
		self.progress.setFormat(message)
		QtWidgets.QApplication.processEvents()

	def choose_land_color(self):
		color = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.land_color), self, "Choose Land Color")
		if color.isValid():
			self.land_color = color.name()
			self.update_plot()

	def choose_ocean_color(self):
		color = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.ocean_color), self, "Choose Ocean Color")
		if color.isValid():
			self.ocean_color = color.name()
			self.update_plot()

	def set_basemap_style(self, style_key):
		self.basemap_style = style_key
		if self.plot_mode == "basemap":
			self.update_plot()


	# --------- Helpers
	def populate_sweeps_for_index(self, i):
		if not self.radars: return
		radar = self.radars[i]; ns = radar.nsweeps
		labels = []
		for s in range(ns):
			st = radar.sweep_start_ray_index['data'][s]; en = radar.sweep_end_ray_index['data'][s]+1
			elev = np.median(radar.elevation['data'][st:en]); labels.append(f"{s}: {elev:.2f}°")
		self.sweep_combo.blockSignals(True); self.sweep_combo.clear(); self.sweep_combo.addItems(labels)
		self.current_sweep = min(self.current_sweep, ns-1); self.sweep_combo.setCurrentIndex(self.current_sweep)
		self.sweep_combo.blockSignals(False)

	def get_sweep_data(self, moment, sweep, radar_obj):
		start = radar_obj.sweep_start_ray_index['data'][sweep]
		end   = radar_obj.sweep_end_ray_index['data'][sweep] + 1
		data  = radar_obj.fields[moment]['data'][start:end]
		az    = radar_obj.azimuth['data'][start:end]
		rng   = radar_obj.range['data']

		# --- Sort by azimuth so that mesh is ordered correctly ---
		order = np.argsort(az)
		az_sorted   = az[order]
		data_sorted = data[order]

		return az_sorted, rng, data_sorted, start, end

	def autoscale_limits(self):
		if not self.radars or self.current_moment is None: return
		radar = self.radars[self.idx]
		az, rng, data, *_ = self.get_sweep_data(self.current_moment, self.current_sweep, radar)
		arr = np.array(data, dtype=float)
		self.vmin_edit.setText(f"{np.nanmin(arr):.3g}")
		self.vmax_edit.setText(f"{np.nanmax(arr):.3g}")
		self.update_plot()

	def reset_limits(self):
		self.vmin_edit.clear(); self.vmax_edit.clear()
		self.update_plot()

	def parse_limits(self):
		vmin_txt = self.vmin_edit.text().strip(); vmax_txt = self.vmax_edit.text().strip()
		if vmin_txt=="" and vmax_txt=="": return None, None
		try:
			return (float(vmin_txt) if vmin_txt!="" else None, float(vmax_txt) if vmax_txt!="" else None)
		except ValueError:
			QtWidgets.QMessageBox.warning(self, "Limits", "vmin/vmax must be numeric."); return None, None
	
	def reset_filters(self):
		# Clear all numeric/text entries
		for widget in [self.thresh_entry, self.sqi_entry, self.pmi_entry,
					   self.rhohv_entry, self.phidp_entry, self.vel_min_entry,
					   self.vel_max_entry, self.zdr_entry, self.cp_entry]:
			widget.clear()

		# Uncheck all speckle filter checkboxes
		for chk in [self.chk_median, self.chk_gaussian, self.chk_kuan,
					self.chk_frost, self.chk_bilateral, self.chk_std]:
			chk.setChecked(False)

		self.update_plot()

	def fill_missing_rays_biggest_gap(self, radar, target_rays):
		import copy
		new_radar = copy.deepcopy(radar)

		filled_azimuth = []
		filled_elevation = []
		filled_time = []
		filled_fields = {f: [] for f in radar.fields}
		sweep_start = []
		sweep_end = []

		ray_idx = 0

		for sweep in range(radar.nsweeps):
			start = radar.sweep_start_ray_index['data'][sweep]
			end = radar.sweep_end_ray_index['data'][sweep] + 1

			az = list(radar.azimuth['data'][start:end])
			elev = list(radar.elevation['data'][start:end])
			times = list(radar.time['data'][start:end])
			field_data = {f: [row for row in radar.fields[f]['data'][start:end]] for f in radar.fields}

			current_rays = len(az)
			if current_rays >= target_rays:
				print(f"Sweep {sweep+1}: {current_rays} rays (no fill needed)")
			else:
				rays_needed = target_rays - current_rays
				print(f"Sweep {sweep+1}: {current_rays} → {target_rays} rays, adding {rays_needed} rays")
				while len(az) < target_rays:
					az_sorted_idx = np.argsort(az)
					az_sorted = np.array(az)[az_sorted_idx]

					diffs = np.diff(np.unwrap(np.deg2rad(np.concatenate([az_sorted, [az_sorted[0] + 360]]))))
					max_gap_idx = np.argmax(diffs)

					insert_az = (az_sorted[max_gap_idx] + np.rad2deg(diffs[max_gap_idx]) / 2) % 360

					insert_pos_sorted = max_gap_idx + 1
					insert_pos = az_sorted_idx[insert_pos_sorted]

					if insert_pos == len(times):
						nearest_time = times[-1]
					else:
						nearest_time = times[insert_pos - 1]

					for f in field_data:
						before = field_data[f][insert_pos - 1]
						after = field_data[f][insert_pos % len(field_data[f])]
						interp_vals = np.ma.mean([before, after], axis=0)
						field_data[f].insert(insert_pos, interp_vals)

					az.insert(insert_pos, insert_az)
					elev.insert(insert_pos, elev[insert_pos - 1])
					times.insert(insert_pos, nearest_time)

			# append filled sweep to global lists
			filled_azimuth.extend(az)
			filled_elevation.extend(elev)
			filled_time.extend(times)
			for f in field_data:
				filled_fields[f].extend(field_data[f])

			sweep_start.append(ray_idx)
			ray_idx += len(az)
			sweep_end.append(ray_idx - 1)

		# Replace data in radar
		new_radar.azimuth['data'] = np.array(filled_azimuth, dtype=np.float32)
		new_radar.elevation['data'] = np.array(filled_elevation, dtype=np.float32)
		new_radar.time['data'] = np.array(filled_time, dtype=np.float64)
		new_radar.sweep_start_ray_index['data'] = np.array(sweep_start, dtype=np.int32)
		new_radar.sweep_end_ray_index['data'] = np.array(sweep_end, dtype=np.int32)

		for f in filled_fields:
			new_radar.fields[f]['data'] = np.ma.array(filled_fields[f])

		return new_radar

	def make_thumbnail(self, radar, moment):
		fig = plt.figure(figsize=(2, 2))
		# use PlateCarree projection for geographic maps
		ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
		ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
		ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)
		ax.add_feature(cfeature.LAND, facecolor=self.land_color)
		ax.add_feature(cfeature.OCEAN, facecolor=self.ocean_color)


		display = pyart.graph.RadarMapDisplay(radar)
		radar_lat = radar.latitude['data'][0]
		radar_lon = radar.longitude['data'][0]

		try:
			display.plot_ppi_map(
				moment, 0,
				ax=ax,
				title_flag=False, colorbar_flag=False,
				embellish=False,  # no labels/axes
				cmap="turbo"      # or use your default per moment
			)
			# set map extent ~2° buffer
			ax.set_extent([radar_lon - 2, radar_lon + 2,
						   radar_lat - 2, radar_lat + 2])
		except Exception:
			fig.clf()
			plt.text(0.5, 0.5, f"No {moment}", ha="center", va="center")

		# Render to QIcon
		canvas = FigureCanvasAgg(fig)
		canvas.draw()
		width, height = fig.canvas.get_width_height()
		image = QtGui.QImage(canvas.buffer_rgba(), width, height, QtGui.QImage.Format_ARGB32)
		plt.close(fig)
		return QtGui.QIcon(QtGui.QPixmap.fromImage(image))


	# --------- Derived products
	def rays_equal_across_sweeps(self, radar):
		counts = []
		for s in range(radar.nsweeps):
			st = radar.sweep_start_ray_index['data'][s]
			en = radar.sweep_end_ray_index['data'][s] + 1
			counts.append(en - st)
		return len(set(counts)) == 1

	def compute_cmax(self, radar, moment):
		if not self.rays_equal_across_sweeps(radar): return None, "inconsistent"
		stack, az, rng = [], None, radar.range['data']
		for s in range(radar.nsweeps):
			az_s, _, d_s, _, _ = self.get_sweep_data(moment, s, radar)
			if az is None: az = az_s
			stack.append(d_s)
		data = np.ma.max(np.ma.stack(stack), axis=0)
		return az, rng, data, 0, data.shape[0]

	def choose_cappi_sweep(self, radar, target_km):
		# pick sweep whose mean gate height is closest to target
		heights = [np.mean(radar.gate_z['data'][s]) / 1000.0 for s in range(radar.nsweeps)]
		return int(np.argmin([abs(h - target_km) for h in heights]))

	def compute_vil(self, radar, moment):
		# VIL ~ ∑ 3.44e-6 * Z_lin * height_km (across sweeps)
		if not self.rays_equal_across_sweeps(radar): return None, "inconsistent"
		rng = radar.range['data']; az, acc = None, None
		for s in range(radar.nsweeps):
			az_s, _, d_s, _, _ = self.get_sweep_data(moment, s, radar)
			if az is None:
				az = az_s
				acc = np.zeros_like(d_s, dtype=float)
			Z_lin = 10.0 ** (d_s / 10.0)  # if 'moment' is reflectivity in dBZ
			height_km = radar.gate_z['data'][s] / 1000.0
			acc += 3.44e-6 * Z_lin * height_km
		return az, rng, acc, 0, acc.shape[0]

	def compute_echotops(self, radar, moment, thr_dbz):
		if not self.rays_equal_across_sweeps(radar): return None, "inconsistent"
		rng = radar.range['data']; az, tops = None, None
		for s in range(radar.nsweeps):
			az_s, _, d_s, _, _ = self.get_sweep_data(moment, s, radar)
			if az is None:
				az = az_s; tops = np.zeros_like(d_s, dtype=float)
			mask = d_s >= thr_dbz
			if np.any(mask):
				height_km = radar.gate_z['data'][s] / 1000.0
				tops = np.maximum(tops, height_km * mask)
		return az, rng, tops, 0, tops.shape[0]

	def compute_stat_product(self, radar_obj, moment, product_type):
		# Check equal ray count
		ray_counts = []
		for i in range(radar_obj.nsweeps):
			start_s = radar_obj.sweep_start_ray_index['data'][i]
			end_s = radar_obj.sweep_end_ray_index['data'][i] + 1
			ray_counts.append(end_s - start_s)
		if len(set(ray_counts)) > 1:
			return None, "Inconsistent"

		all_data = []
		az = None
		rng = radar_obj.range['data']
		for s in range(radar_obj.nsweeps):
			az_s, _, d_s, _, _ = self.get_sweep_data(moment, s, radar_obj)
			if az is None:
				az = az_s
			all_data.append(d_s)

		stack = np.ma.stack(all_data)
		if product_type == "Max":
			data = np.ma.max(stack, axis=0)
		elif product_type == "Min":
			data = np.ma.min(stack, axis=0)
		elif product_type == "Mean":
			data = np.ma.mean(stack, axis=0)
		elif product_type == "StdDev":
			data = np.ma.std(stack, axis=0)

		return (az, rng, data, 0, data.shape[0]), 0

	def apply_combined_speckle_filters(self, data, filters, size=3, std_threshold=1.0):
		original_mask = np.ma.getmaskarray(data)
		data_filtered = data.copy()

		for name in ["Median", "Gaussian", "Kuan", "Frost", "Bilateral", "STD"]:
			if not filters.get(name, False):
				continue

			data_filled = data_filtered.filled(0)

			if name == "Median":
				filtered = median_filter(data_filled, size=size)

			elif name == "Gaussian":
				filtered = gaussian_filter(data_filled, sigma=size/2)

			elif name == "Kuan":
				df32 = data_filled.astype(np.float32)
				mean = uniform_filter(df32, size=size)
				mean_sq = uniform_filter(df32**2, size=size)
				variance = mean_sq - mean**2
				noise_variance = np.nanmean(variance)
				W = 1 - (noise_variance / (variance + 1e-12))
				filtered = mean + W * (df32 - mean)

			elif name == "Frost":
				df32 = data_filled.astype(np.float32)
				mean = uniform_filter(df32, size=size)
				mean_sq = uniform_filter(df32**2, size=size)
				variance = mean_sq - mean**2
				noise_variance = np.nanmean(variance)
				alpha = - (variance / (noise_variance + 1e-12))
				filtered = mean + (df32 - mean) * np.exp(alpha)

			elif name == "Bilateral":
				df32 = data_filled.astype(np.float32)
				norm = cv2.normalize(df32, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
				filtered_norm = cv2.bilateralFilter(norm, d=size, sigmaColor=75, sigmaSpace=75)
				filtered = cv2.normalize(filtered_norm.astype(np.float32), None,
										 df32.min(), df32.max(), cv2.NORM_MINMAX)

			elif name == "STD":
				std_map = generic_filter(data_filled, np.std, size=size, mode='nearest')
				data_filtered = np.ma.masked_where(std_map < std_threshold, data_filtered)
				continue

			data_filtered = np.ma.array(filtered, mask=original_mask)
			if np.all(data_filtered.mask):
				data_filtered = data

		return data_filtered

	def convert_to_hdf5(self):
		if not self.files:
			QtWidgets.QMessageBox.warning(self, "Convert to HDF5", "No radar file loaded.")
			return

		input_file = self.files[self.idx]

		# Ask for output folder
		folder = QtWidgets.QFileDialog.getExistingDirectory(
			self, "Select Output Folder", "/etc/pyiris/output/"
		)
		if not folder:
			return

		# Show progress
		self.start_progress("Converting to HDF5...")

		try:
			result = subprocess.run(
				["pyiris", "-i", input_file, "-o", folder],
				capture_output=True, text=True
			)
			if result.returncode == 0:
				self.statusBar().showMessage(f"Conversion complete → {folder}", 5000)
				QtWidgets.QMessageBox.information(
					self, "Success", f"File saved in:\n{folder}"
				)
				self.update_recent_files(folder)
			else:
				QtWidgets.QMessageBox.critical(
					self, "Conversion Error",
					f"pyiris failed:\n\n{result.stderr}"
				)
		except Exception as e:
			QtWidgets.QMessageBox.critical(self, "Error", str(e))
		finally:
			self.stop_progress("Ready")
	
	
	# --------- Plot/update
	def update_plot(self):
		if not self.radars or self.current_moment is None:
			return
		self.start_progress("Processing data...")
		radar = self.radars[self.idx]
		self.current_sweep = min(self.current_sweep, radar.nsweeps - 1)

		product = self.product_combo.currentText()
		moment  = self.current_moment

		try:
			# ---- get data based on product ----
			if product == "Normal":
				az, rng, data, start, end = self.get_sweep_data(moment, self.current_sweep, radar)

			elif product in ("Max", "Min", "Mean", "StdDev"):
				res, status = self.compute_stat_product(radar, moment, product)
				if res is None:
					QtWidgets.QMessageBox.warning(self, product, "Sweeps have different ray counts.")
					return
				az, rng, data, start, end = res

			elif product == "CMax":
				res = self.compute_cmax(radar, moment)
				if res is None or (isinstance(res, tuple) and len(res) == 2):
					QtWidgets.QMessageBox.warning(self, "CMax", "Sweeps have different ray counts.")
					return
				az, rng, data, start, end = res

			elif product == "CAPPI":
				try:
					target_km = float(self.cappi_edit.text().strip())
				except ValueError:
					QtWidgets.QMessageBox.warning(self, "CAPPI", "Invalid CAPPI height (km).")
					return
				sweep_idx = self.choose_cappi_sweep(radar, target_km)
				az, rng, data, start, end = self.get_sweep_data(moment, sweep_idx, radar)

			elif product == "VIL":
				res = self.compute_vil(radar, moment)
				if res is None or (isinstance(res, tuple) and len(res) == 2):
					QtWidgets.QMessageBox.warning(self, "VIL", "Sweeps have different ray counts.")
					return
				az, rng, data, start, end = res

			elif product == "EchoTops":
				try:
					thr = float(self.echotops_thr_edit.text().strip())
				except ValueError:
					QtWidgets.QMessageBox.warning(self, "EchoTops", "Invalid threshold (dBZ).")
					return
				res = self.compute_echotops(radar, moment, thr)
				if res is None or (isinstance(res, tuple) and len(res) == 2):
					QtWidgets.QMessageBox.warning(self, "EchoTops", "Sweeps have different ray counts.")
					return
				az, rng, data, start, end = res

			else:
				return

			# ---- apply filters ----
			try:
				if self.thresh_entry.text():
					data = np.ma.masked_where(data < float(self.thresh_entry.text()), data)

				if self.sqi_entry.text():
					sqi_field = next((f for f in radar.fields if "SQI" in f.upper()), None)
					if sqi_field:
						sqi_data = radar.fields[sqi_field]['data'][start:end]
						data = np.ma.masked_where(sqi_data < float(self.sqi_entry.text()), data)

				if self.pmi_entry.text():
					pmi_field = next((f for f in radar.fields if "PMI" in f.upper()), None)
					if pmi_field:
						pmi_data = radar.fields[pmi_field]['data'][start:end]
						data = np.ma.masked_where(pmi_data < float(self.pmi_entry.text()), data)

				if self.rhohv_entry.text():
					rho_field = next((f for f in radar.fields if "RHOHV" in f.upper()), None)
					if rho_field:
						rho_data = radar.fields[rho_field]['data'][start:end]
						data = np.ma.masked_where(rho_data < float(self.rhohv_entry.text()), data)

				if self.phidp_entry.text():
					phi_field = next((f for f in radar.fields if "PHIDP" in f.upper()), None)
					if phi_field:
						phi_data = radar.fields[phi_field]['data'][start:end]
						data = np.ma.masked_where(phi_data < float(self.phidp_entry.text()), data)

				if self.vel_min_entry.text() or self.vel_max_entry.text():
					vel_field = next((f for f in radar.fields if "VEL" in f.upper()), None)
					if vel_field:
						vel_data = radar.fields[vel_field]['data'][start:end]
						if self.vel_min_entry.text():
							data = np.ma.masked_where(vel_data < float(self.vel_min_entry.text()), data)
						if self.vel_max_entry.text():
							data = np.ma.masked_where(vel_data > float(self.vel_max_entry.text()), data)

				if self.zdr_entry.text():
					zdr_field = next((f for f in radar.fields if "ZDR" in f.upper()), None)
					if zdr_field:
						zdr_data = radar.fields[zdr_field]['data'][start:end]
						data = np.ma.masked_where(zdr_data < float(self.zdr_entry.text()), data)

				if self.cp_entry.text():
					cp_field = next((f for f in radar.fields if "CP" in f.upper()), None)
					if cp_field:
						cp_data = radar.fields[cp_field]['data'][start:end]
						data = np.ma.masked_where(cp_data < float(self.cp_entry.text()), data)

				# --- Speckle filters ---
				filters = {
					"Median": self.chk_median.isChecked(),
					"Gaussian": self.chk_gaussian.isChecked(),
					"Kuan": self.chk_kuan.isChecked(),
					"Frost": self.chk_frost.isChecked(),
					"Bilateral": self.chk_bilateral.isChecked(),
					"STD": self.chk_std.isChecked(),
				}
				if any(filters.values()):
					data = self.apply_combined_speckle_filters(data, filters, size=3, std_threshold=1.0)


			except Exception as e:
				QtWidgets.QMessageBox.warning(self, "Filter Error", str(e))

			# ---- limits
			vmin, vmax = self.parse_limits()
			
			# ---- draw plot (reset properly) ----
			self.ax.cla()

			theta = np.deg2rad(az)
			rr, th = np.meshgrid(rng, theta)

			cmap = plt.get_cmap(self.cmap_combo.currentText())

			if self.plot_mode == "polar":
				# Polar projection
				self.fig.delaxes(self.ax)
				self.ax = self.fig.add_subplot(111, polar=True)
				self.ax.set_theta_zero_location("N")
				self.ax.set_theta_direction(-1)

				pcm = self.ax.pcolormesh(
					th, rr, data,
					shading="auto", cmap=cmap,
					vmin=vmin, vmax=vmax
				)
			else:  # Basemap projection
				radar_lat = radar.latitude['data'][0]
				radar_lon = radar.longitude['data'][0]

				# Recreate axis with PlateCarree projection
				self.fig.delaxes(self.ax)
				self.ax = self.fig.add_subplot(111, projection=ccrs.PlateCarree())

				# --- Draw range rings (e.g. every 50 km) ---
				max_range_km = rng.max() / 1000.0
				if self.plot_mode == "basemap" and  self.chk_rings.isChecked():
					# --- Draw range rings ---
					for r_km in np.arange(50, max_range_km + 2, 50):
						circle_lon, circle_lat = [], []
						for az in np.linspace(0, 360, 361):
							az_rad = np.deg2rad(az)
							dx = r_km * np.sin(az_rad) / 111.0
							dy = r_km * np.cos(az_rad) / 111.0
							circle_lon.append(radar_lon + dx)
							circle_lat.append(radar_lat + dy)

						self.ax.plot(circle_lon, circle_lat, color="gray", linewidth=0.5,
								transform=ccrs.PlateCarree())
						# label
						label_angle = np.deg2rad(15)   # 15° azimuth
						dx = (r_km + 5) * np.sin(label_angle) / 111.0
						dy = (r_km + 5) * np.cos(label_angle) / 111.0

						self.ax.text(radar_lon + dx, radar_lat + dy,
								f"{int(r_km)} km",
								ha="left", va="bottom", fontsize=7, color="gray",
								transform=ccrs.PlateCarree())

					# --- Draw azimuth lines ---
					for az in range(0, 360, 30):
						az_rad = np.deg2rad(az)
						x = (max_range_km * np.sin(az_rad)) / 111.0
						y = (max_range_km * np.cos(az_rad)) / 111.0

						self.ax.plot([radar_lon, radar_lon + x],
								[radar_lat, radar_lat + y],
								color="gray", linewidth=0.5, transform=ccrs.PlateCarree())
						# label
						self.ax.text(radar_lon + x, radar_lat + y,
								f"{az}°", ha="center", va="center",
								fontsize=7, color="gray", transform=ccrs.PlateCarree())

				# Apply chosen basemap style
				if self.basemap_style == "simple":
					self.ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
					self.ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)
					self.ax.add_feature(cfeature.LAND, facecolor=self.land_color)
					self.ax.add_feature(cfeature.OCEAN, facecolor=self.ocean_color)

				elif self.basemap_style == "stamen_terrain":
					stamen = cimgt.Stamen("terrain-background")
					self.ax.add_image(stamen, 8)

				elif self.basemap_style == "stamen_toner":
					stamen = cimgt.Stamen("toner-lite")
					self.ax.add_image(stamen, 8)

				elif self.basemap_style == "stamen_watercolor":
					stamen = cimgt.Stamen("watercolor")
					self.ax.add_image(stamen, 8)

				elif self.basemap_style == "osm":
					osm = cimgt.OSM()
					self.ax.add_image(osm, 8)

				# Center map around radar
				# --- Dynamically set extent based on radar max range ---
				max_range_km = rng.max() / 1000.0
				km2deg = 1.0 / 111.0  # rough conversion, valid near equator

				extent_buffer = max_range_km * km2deg + 0.2
				self.ax.set_extent([radar_lon - extent_buffer, radar_lon + extent_buffer,
									radar_lat - extent_buffer, radar_lat + extent_buffer],
									crs=ccrs.PlateCarree())

				# Convert radar polar coords → lon/lat
				x = rr * np.sin(th) / 1000.0  # km east
				y = rr * np.cos(th) / 1000.0  # km north
				lon = radar_lon + (x / 111.0)
				lat = radar_lat + (y / 111.0)

				pcm = self.ax.pcolormesh(
					lon, lat, data,
					shading="auto", cmap=cmap,
					vmin=vmin, vmax=vmax,
					transform=ccrs.PlateCarree()
				)
			# Update or create colorbar
			# Remove inline colorbar if any
			if self._cbar:
				try:
					self._cbar.remove()
				except Exception:
					pass
			self._cbar = None

			# --- external colorbar in filter dock ---
			cbar_fig = plt.Figure(figsize=(1.0, 3.0), dpi=100)   # make it tall enough
			cax = cbar_fig.add_axes([0.25, 0.05, 0.5, 0.9])      # [left, bottom, width, height]

			# Use ScalarMappable (pcm) to generate ticks + labels
			cbar = cbar_fig.colorbar(pcm, cax=cax, orientation="vertical")

			# Force tick placement
			cbar.locator = plt.MaxNLocator(nbins=6)   # max 6 ticks
			cbar.update_ticks()

			# Tick font size
			cbar.ax.tick_params(labelsize=8)

			# Add label with unit
			moment_name = self.current_moment or "Unknown"
			moment_upper = moment_name.upper()

			# Strip trailing digits (e.g. DBZ2 → DBZ, VEL3 → VEL)
			moment_base = ''.join([c for c in moment_upper if not c.isdigit()])

			unit = ""
			try:
				unit = radar.fields[moment_name].get("units", "")
			except Exception:
				pass

			if not unit:
				for key, u in self.moment_units.items():
					#print(key,u, moment_base)
					if key in moment_base:   # use cleaned name
						unit = u
						break
			#print(unit)
			# Update note/unit label (moment + unit)
			self.cbar_unit.setText(f"{moment} [{unit}]")

			# Render into QImage for dock
			canvas = FigureCanvasAgg(cbar_fig)
			canvas.draw()
			w, h = canvas.get_width_height()
			img = QtGui.QImage(canvas.buffer_rgba(), w, h, QtGui.QImage.Format_ARGB32)
			self.cbar_label.setPixmap(QtGui.QPixmap.fromImage(img))

			# Title / status
			base = os.path.basename(self.files[self.idx]) if self.files else ""
			title_text = f"{product} {moment} | Sweep {self.current_sweep} | {base}"
			self.title_label.setText(title_text)
			
			# Make plot fill entire canvas
			self.fig.tight_layout(pad=0)  
			self.ax.set_position([0.05, 0.05, 0.85, 0.9])  # [left, bottom, width, height]
			self.canvas.draw_idle()

			try:
				ts = radar.time["units"].split("since")[-1].strip()
			except Exception:
				ts = "Unknown time"
			self.statusBar().showMessage(f"Timestamp units: {ts}")

			self.canvas.draw_idle()
		except Exception as e:
			QtWidgets.QMessageBox.critical(self, "Update Error", str(e))
		finally:
				self.stop_progress("Ready")

# ---- Run
if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	w = RadarViewer()
	w.show()
	sys.exit(app.exec_())
