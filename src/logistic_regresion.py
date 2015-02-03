import csv
import glob
import os
import numpy as np
import time
import random as rd
from sklearn import linear_model
import operator
from sklearn.ensemble import RandomForestClassifier 

def openFile(filename):
    matfeature = []
    with open(filename,'rb') as f:
		f.next() # skip header line
		for line in f:
			lstring = line.split(",")
			lstring[3] = '0'
			listfeature = map(float,lstring)#run in python ver 2.x
			#listfeature = list(map(int,line.split(",").pop(3)))#run in python ver 3.x
			matfeature.append(listfeature)
    return matfeature

def openDiverTrip(filename):
	matfeature=[]
	with open(filename,'rb') as f:
		f.next() # skip header line
		for line in f:
			listfeature = line.split(",")
			matfeature.append(listfeature)
	return matfeature
    
def getsinglefeature(drivername,tripid):
	with open(drivername,'rb') as f:
		f.next() # skip header line
		for line in f:
			lstring = line.split(",")
			if lstring[2]==tripid:
				lstring[3] = '0'
				listfeature = map(float,lstring)#run in python ver 2.x
				#break;
			
	feature =[]
	label = str(int(listfeature[1]))+'_'+str(int(listfeature[2]))
	feature.append(listfeature[4]) # Duration	
	feature.append(listfeature[77])	# hard acceleration 2.5
	feature.append(listfeature[78])	# hard deceleration 2.5
	feature.append(listfeature[56]) # Average speed (excluding STOP)
	
	return label,feature
	
def getfeature(drivername):
    matfeature = openFile(drivername)
    labels = []
    neededfeature =[]

    for listfeature in matfeature:
		feature =[]
		labels.append(str(int(listfeature[1]))+'_'+str(int(listfeature[2])))
		feature.append(listfeature[4]) # Duration
		#feature.append(listfeature[23])	# hard acceleration
		#feature.append(listfeature[24])	# hard deceleration
		feature.append(listfeature[77])	# hard acceleration 2.5
		feature.append(listfeature[78])	# hard deceleration 2.5
		feature.append(listfeature[56]) # Average speed (excluding STOP)
		#feature.append(listfeature[16])
		#feature.append(listfeature[55]) # Average speed turning
		#feature.append(listfeature[15])	# time of speed up
		#feature.append(listfeature[16])	# time of speed down 
		#feature.append(listfeature[57]) # Proportion of time at constant speed
		#feature.append(listfeature[61]) # max deceleration value to stop
		#feature.append(listfeature[62]) # number of stop
		neededfeature.append(feature)
    return labels,neededfeature

def runLogisticRegression(training_set, test_set, labels):

	model = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
	model.fit_transform(training_set, labels)
	prob = model.predict_proba(np.array(test_set))
	
	return prob
	
def runRandomForest(training_set, test_set, labels):
	model = RandomForestClassifier(n_estimators=5000,n_jobs=-1)
	model.fit(training_set, labels)
	prob = model.predict_proba(np.array(test_set))
	
	return prob	
	
def main():
	
	#featurepath = "F:/Kaggle/Driver Telematics Analysis/data/data_test/"
	featurepath = "/home/linh/workspace/telematics/data/features_all_13/"
	drivertriptable = openDiverTrip("/home/linh/driver_trip.csv")
	
	f = open("/home/linh/workspace/telematics/lr_features_v9_1_22c_23c_31.csv",'w')
	f.write('driver_trip,prob\n')
	start_time = time.time()
	listDrivers = os.listdir(featurepath)
	numrandomtrip = 1000
	targets =np.zeros(numrandomtrip)
	label_1 = 200
	label_2 = 180 # label_1 = label_2 + num_reject
	num_reject = label_1 - label_2
	
	targets1=np.append(targets,np.ones(label_1)) 
	targets2=np.append(targets,np.ones(label_2)) 
	
	randomTrips = rd.sample(xrange(len(drivertriptable)),numrandomtrip)
	reffeature = []
    
	for rdtrip in randomTrips:
		drivertrip = drivertriptable[rdtrip]
		#print drivertrip
		label,tripfeature = getsinglefeature(featurepath+drivertrip[0]+'.csv',drivertrip[1].strip())
		reffeature.append(tripfeature)

	reffeature = np.array(reffeature) 	
	print ('-%s seconds --- loading ref data complete' %str(time.time() - start_time))
	
	step = 1
	for driver in listDrivers:
		if step%10==0:
			print ('-%s seconds --- %s driver -- %s' %(str(time.time() - start_time),driver,step))
		labels,curfeature = getfeature(featurepath+driver)
		#print labels[0]
		#print curfeature[0]
		#break
		features = np.concatenate((reffeature,curfeature))
		# Running model predicted first
		prob_1 = runLogisticRegression(features, curfeature, targets1)
		
		### sort prediction result 
		prob_s = []		
		for p in prob_1:
			prob_s.append(p[1])
		
		# Find probability will be rejected
		threshold = sorted(prob_s)[num_reject]
		
		list_index = []
		for index_p in range(len(prob_s)):
			if(prob_s[index_p]>=threshold):
				list_index.append(index_p)
				
		if len(list_index)>label_2:
			list_index = list_index[:label_2]
			
		# extract features for current driver after remove num_reject trips
		curfeature2 = []
		for index_c in range(len(curfeature)):
			if index_c in list_index:
				curfeature2.append(curfeature[index_c])
		
		features2 = np.concatenate((reffeature,curfeature2))
		
		# Running model predicted second
		prob_2 = runLogisticRegression(features2, curfeature, targets2)
		
		# Writing ouput file
		for i,j in zip(labels,prob_2):
			f.write(str(i)+','+str(j[1])+'\n')
		
		#break
		step = step+1
	
	print "Done"
	f.close()   
	
main()


