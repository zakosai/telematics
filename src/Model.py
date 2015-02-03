'''
Created on Jan 28, 2015

@author: linh
'''

import pandas as pd
import numpy as np
import csv
import os
import random as rd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import time


def getFeatureZero(driverfile, trip):
    f = open(driverfile, 'rb')
    rows = csv.reader(f)
    
    rows = list(rows)
    feature = []
    rows[trip][3] = '0'
    row = map(float, rows[trip])
    feature.append(row[4])    # 1 - duration
#     feature.append(row[7])    # 4- avg_correct_speed
#     feature.append(row[8])    # 5- avg_corrected_acc
   
#     feature.append(row[9]) # 6b- avg_angle_change

#     feature.append(row[11])   #17 - total distance
#     feature.append(row[12])   # 8 - fastest speed
#     feature.append(row[15])   #10 = time speed up
#     feature.append(row[16])   #11 - time of speed down
#    
#     feature.append(row[19]) # 12- var_instant speed
#     feature.append(row[20]) # 13- var_instant acc
# 
#     feature.append(row[21]) # 16- highway
#     feature.append(row[22]) # 18- num_of_acc more than 4


    feature.append(row[23])   #23 - hard acceleration 2
    feature.append(row[24])   #22 - hard deceleration 2

#     feature.extend(row[25:30])#20 - five turning
#     feature.extend(row[35:40])#21 - five turning

#     feature.append(row[47])   #24 - per_speed_15 
#     feature.append(row[48])   #25 - per_speed_11_15 
#     feature.append(row[49])   #26 - per_speed_7_11 
#     feature.append(row[50])   #27 - per_speed_5_7 
#     feature.append(row[51])   #28 - per_speed_5
#     feature.append(row[54])   #35 - time of turning
#     feature.append(row[55])   #36 - avg speed turning  
    feature.append(row[56])   #31 - avg speed excluding stop 
#     feature.append(row[57]) # 34 - Proportion of time at constant speed
# 
#     feature.append(row[58])   #32 - 10s acceleration
#     feature.append(row[59])   #33 - 10s deceleration   
#     feature.append(row[60]) # 39 - time taken to stop
#     feature.append(row[61]) # 40 - max dec
#     feature.append(row[62]) # 41 - number of stop
#     feature.append(row[63]) # 42 - avg_period
#     feature.append(row[65]) # 44 - max_period
#     feature.append(row[67]) #46 - var_max_p
    
#     feature.append(row[68]) # 48 - std_distance travel
#     feature.append(row[69]) # 47 - avg_distance travel
#     
#     feature.append(row[72]) # 51 - pro of acc below 1m/s2
#     feature.append(row[73]) # 52 - pro of dec below 1m/s2
#     
#     feature.append(row[74]) # 18b num of acc more than 3
#     feature.append(row[76]) # 23b hard acc 1.5
#     feature.append(row[77]) # 22b hard braking 1.5 
    
#     feature.append(row[80]) # per straight 
#     feature.append(row[81]) # per 10_45
#     feature.append(row[82]) # per 45_90
#     feature.append(row[83]) # per 90_120  
    return feature

    
def getFeatureOne(driverfile):
    f = open(driverfile, 'rb')
    rows = csv.reader(f)
    
    rows.next()
    features = []
    driver_trip = []
    for row in rows:
        feature = []
        driver_trip.append(row[3])
        row[3] = '0'
        row = map(float, row)
        feature.append(row[4])    # 1 - duration
#         feature.append(row[7])    # 4- avg_correct_speed
#         feature.append(row[8])    # 5- avg_corrected_acc
        
#         feature.append(row[9]) # 6b- avg_angle_change

#         feature.append(row[11])   #17 - total distance
#         feature.append(row[12])   # 8 - fastest speed
#         feature.append(row[15])   #10 = time speed up
#         feature.append(row[16])   #11 - time of speed down
#         
#         feature.append(row[19]) # 12- var_instant speed
#         feature.append(row[20]) # 13- var_instant acc
#  
#         feature.append(row[21]) # 16- highway
#         feature.append(row[22]) # 18- num_of_acc more than 4
 

        feature.append(row[23])   #23 - hard acceleration 2
        feature.append(row[24])   #22 - hard deceleration 2
#      
#         feature.extend(row[25:30])#20 - five turning
#         feature.extend(row[35:40])#21 - five turning

#         feature.append(row[47])   #24 - per_speed_15 
#         feature.append(row[48])   #25 - per_speed_11_15 
#         feature.append(row[49])   #26 - per_speed_7_11 
#         feature.append(row[50])   #27 - per_speed_5_7 
#         feature.append(row[51])   #28 - per_speed_5
#         feature.append(row[54])   #35 - time of turning
#         feature.append(row[55])   #36 - avg speed turning  
        feature.append(row[56])   #31 - avg speed excluding stop 
#         feature.append(row[57]) # 34 - Proportion of time at constant speed
# 
#         feature.append(row[58])   #32 - 10s acceleration
#         feature.append(row[59])   #33 - 10s deceleration   
#         feature.append(row[60]) # 39 - time taken to stop
#         feature.append(row[61]) # 40 - max dec
#         feature.append(row[62]) # 41 - number of stop
#         feature.append(row[63]) # 42 - avg_period
#         feature.append(row[65]) # 44 - max_period
#         feature.append(row[67]) #46 - var_max_p
         
#         feature.append(row[68]) # 48 - std_distance travel
#         feature.append(row[69]) # 47 - avg_distance travel
#          
#         feature.append(row[72]) # 51 - pro of acc below 1m/s2
#         feature.append(row[73]) # 52 - pro of dec below 1m/s2
#          
#         feature.append(row[74]) # 18b num of acc more than 3
#         feature.append(row[76]) # 23b hard acc 1.5
#         feature.append(row[77]) # 22b hard braking 1.5 
         
#         feature.append(row[80]) # per straight 
#         feature.append(row[81]) # per 10_45
#         feature.append(row[82]) # per 45_90
#         feature.append(row[83]) # per 90_120
        
        
        
        features.append(feature)
    
    return features, driver_trip


def logisticRegression(train, test, label):
    clf = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    
    clf.fit_transform(train, label)
    pre = clf.predict_proba(np.array(test))
    
    return pre
def randomForest(train, test, labels):
    clf = RandomForestClassifier(n_estimators=5000,n_jobs=-1)
    
    clf.fit_transform(train, labels)
    pre = clf.predict_proba(np.array(test))
    
    return pre    


def main():
    start_time = time.time()
    path = '/home/linh/workspace/telematics/data/features_all_16/'
    
    drivers = os.listdir('/home/linh/drivers/')
        
    f = open('/home/linh/workspace/telematics/data/result_pre/rf_rf_v16_1_22_23_31.csv', 'wb')
    a = csv.writer(f)
    
    w = ['driver_trip', 'prob']
    a.writerow(w)
    
    drivercount = 0
    driverlist = []
    
    for driverpredict in drivers:
        drivercount += 1
        if (driverpredict in driverlist) | (len(driverlist) == 0) :
            tmp = list(drivers)
            tmp.remove(driverpredict)
            
            #step 1
            driverlist = rd.sample(tmp, 400)
            trip = np.random.randint(1, 200, size=400)
            label = np.zeros(400)
            
            label1 = np.append(label, np.ones(200))
            
            featuresZero = []
            for i in range(len(driverlist)):
                feature = getFeatureZero(path+driverlist[i]+'.csv', trip[i])
                featuresZero.append(feature)
            
        featureOne, driver_trip = getFeatureOne(path+driverpredict+'.csv')
        train = featuresZero + featureOne
        
        pre = randomForest(train, featureOne, label1)
        
        count = 0
        listprob = []
        for i in range(len(pre)):
            listprob.append(pre[i][1])
            if pre[i][1] > 0.66:
                count += 1
        
        print count

        
        #step 2
        threshold = 0.66
        
        features = []
        for i in range(len(pre)):
            if pre[i][1] >= threshold:
                features.append(featureOne[i])
        
        train = featuresZero + features 
        label2 = np.append(label, np.ones(len(features)))
        
        pre = randomForest(train, featureOne, label2)
        
        
        for i in range (len(pre)):
            w = [driver_trip[i],str(pre[i][1]) ]
            a.writerow(w)
        
        print drivercount, time.time() - start_time

        
    f.close()


main()
    
    
    
    
    
    
    
    
        