# pyiris/logger_utils.py
from datetime import datetime
import numpy as np

def getnowtime():
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

def log_print(messages, add_messages=np.empty):
    now = datetime.now().strftime('%d%m%y')
    fi = open('log/pyiris.log'.format(now), 'a')
    if add_messages == np.empty:
        all_messages = messages + '\n'
        print(messages)
    else:
        all_messages = messages + np.array2string(np.array(add_messages)) + '\n'
        print(messages, add_messages)
    fi.write(all_messages)
    fi.close()
