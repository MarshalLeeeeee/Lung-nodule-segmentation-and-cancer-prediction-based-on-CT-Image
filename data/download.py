"""import os
import sys
#from datatime import datatime

file_id = [
	'0Bz-jINrxV740dVhQRHl60WNRanc',
	'0Bz-jINrxV740QTZNenZadmw4NEE',
	'0Bz-jINrxV740OFRGZ0lvZipZU28',
	'0Bz-jINrxV740Z0NoSW9JRE9RRVk',
	'0Bz-jINrxV740SDB3R0hYalJGa2M',
	'0Bz-jINrxV740Nm1fQWhTZUVia2S',
	'0Bz-jINrxV740YXBpYV9MQkVxSk0',
	'0Bz-jINrxV740YTM0X2dQcGZ1eFU',
	'0Bz-jINrxV740WEpXakYyRWxaSm8',
	'0Bz-jINrxV740YVZFMEdpQmZrOTQ',
]

fid = '0Bz-jINrxV740dVhQRHl60WNRanc'
filename = 'subset0.zip'

command = 'python google_drive.py ' + fid + ' ' + filename
os.system(command)"""


import os
import sys
#from datetime import datetime


file_id = [
        '0Bz-jINrxV740dVhQRHl6OWNRanc',
        '0Bz-jINrxV740QTZNenZadmw4NEE',
        '0Bz-jINrxV740OFRGZ0lvZ1pZU28',
        '0Bz-jINrxV740Z0NoSW9JRE9RRVk',
        '0Bz-jINrxV740SDB3R0hYalJGa2M',
        '0Bz-jINrxV740Nm1fQWhTZUVia2s',
        '0Bz-jINrxV740YXBpYV9MQkVxSk0',
        '0Bz-jINrxV740YTM0X2dQcGZ1eFU',
        '0Bz-jINrxV740WEpXakYyRWxaSm8',
        '0Bz-jINrxV740YVZFMEdpQmZrOTQ'
]

file_name = ['subset'+str(i)+'.zip' for i in range(0,10)]

for fid, fname in zip(file_id, file_name):
        print(fname)
        print(fid)
        #print datetime.now()

        command = 'python google_drive.py '+fid+' '+fname
        os.system(command)
