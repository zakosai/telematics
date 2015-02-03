'''
Created on Jan 29, 2015

@author: linh
'''

import csv

f = open('/home/linh/workspace/telematics/data/features_all_13/1.csv', 'rb')
rows = csv.reader(f)

title = list(rows.next())

f = open('/home/linh/workspace/telematics/data/features.csv', 'wb')
a = csv.writer(f)
for i in range(len(title)):
    w = [i, title[i]]
    a.writerow(w)