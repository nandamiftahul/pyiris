# pyiris/moments.py
import numpy as np
from difflib import get_close_matches

def string2arrayint64(s,ls):
	a = np.zeros(ls, dtype='int64')
	i = 0
	for c in s:
		a[i] = ord(c)
		i+=1
	return(a)

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
