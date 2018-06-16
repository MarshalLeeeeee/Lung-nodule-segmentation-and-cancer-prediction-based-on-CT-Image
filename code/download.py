"""
	a simple automatic script for download LUNA16 dataset from google drive
	
	Usage: python download.py 
"""

from __future__ import print_function
import sys
import os 
from datetime import datetime

file_id = [
'https://drive.google.com/open?id=0Bz-jINrxV740dVhQRHl6OWNRanc', 
'https://drive.google.com/open?id=0Bz-jINrxV740QTZNenZadmw4NEE',
'https://drive.google.com/open?id=0Bz-jINrxV740OFRGZ0lvZ1pZU28',
'https://drive.google.com/open?id=0Bz-jINrxV740Z0NoSW9JRE9RRVk',
'https://drive.google.com/open?id=0Bz-jINrxV740SDB3R0hYalJGa2M',
'https://drive.google.com/open?id=0Bz-jINrxV740Nm1fQWhTZUVia2s',
'https://drive.google.com/open?id=0Bz-jINrxV740YXBpYV9MQkVxSk0',
'https://drive.google.com/open?id=0Bz-jINrxV740YTM0X2dQcGZ1eFU',
'https://drive.google.com/open?id=0Bz-jINrxV740WEpXakYyRWxaSm8',
'https://drive.google.com/open?id=0Bz-jINrxV740YVZFMEdpQmZrOTQ'
]

file_id = [fid[fid.find('=')+1:] for fid in file_id]

file_name = ['subset'+str(i)+'.zip' for i in range(1, 10)]

for fid, fname in zip(file_id, file_name):
	print fname 
	print datetime.now()
	command = 'python google_drive.py '+fid+ ' '+fname 
	os.sys(command)