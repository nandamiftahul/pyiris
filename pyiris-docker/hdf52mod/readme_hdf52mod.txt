hdf52mod

Preparation:
1) copy the exe file (hdf52mod) to /etc/pyiris/hdf52mod
2) copy the conf file (hdf52mod.conf) to /etc/pyiris/hdf52mod
3) check default configuration of the hdf52mod.conf
   hdf5 file to be at /etc/pyiris/output
   original IRIS RAW to be at /etc/pyiris/input
   create "temp" directory under /etc/pyiris
   create "raw_out" directory under /etc/pyiris
4) modified IRIS RAW to be generated under /etc/pyiris/raw_out directory

Usage:
$ hdf52mod -i (hdf5 filename) -o (modified IRIS RAW filename)
Note: the "hdf5 filename" must be only filename (directory path not needed)
      the "modified IRIS RAW filename" must be only filename (directory path not needed)

Version check:
$ hdf52mod -y
	
04/17/2025     Naga
04/22/2025     Program name changed from hdf52iris_mod to hdf52mod

