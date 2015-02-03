# @Day 12/31/2014
# @ Author: Keziah Do

# Caluate features for each driver

#!/usr/bin/python

import csv
import glob
import os
import numpy as np
import time
import random
from sklearn import linear_model
import operator



def openFile(filename):
    listUsers = []
    with open(filename,'rb') as rfile:
        reader = csv.reader(rfile)
        header = reader.next()
        x = []
        y = []
        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))
    return x,y

def calculateDistance(x,y):
    distance = [0]
    for i in range(len(x)-1):
        d = ((x[i+1]-x[i])**2 +(y[i+1]-y[i])**2)**0.5
        distance.append(d)   
    return distance

def calculateVerlocity(x,y):
    velocity = calculateDistance(x,y)
    return velocity

def calculateAcceleration(v):
    acceleration = [0]
    for i in range(len(v)-1):
        acc = v[i+1] - v[i]
        acceleration.append(acc)
        
    return acceleration

def calculateAngleHorizontal(x,y):
    angle = []
    angle.append(0)
    for i in range(len(x)-1):
        if(x[i+1] - x[i]) == 0:
            angle.append(float(90))
        else:
            tan = (y[i+1]-y[i])/(x[i+1]-x[i])
            angle.append(np.arctan(tan)*180/np.pi)
        
    return angle
        
def calculateAngleChange(x,y,distance):
    angleT = []
    
    angleT.append(0)
    angleT.append(0)
 
    for i in range(len(x)-2):
        dot = (x[i+1]-x[i])*(x[i+2]-x[i+1]) + (y[i+1]-y[i])*(y[i+2]-y[i+1])
        d1 = distance[i+1]
        d2 = distance[i+2]
        
        if(d1*d2!=0):
            cos = dot/(d1*d2)
            if (cos>1.0 or cos <-1.0) :
                cos = 1.0   
            angleT.append(np.arccos(cos)*180/np.pi)
        else:
            angleT.append(0)
    return angleT    


def correctInstantSpeed(velocity):
	int_speed = velocity
	for i in range(len(int_speed)):
		if int_speed[i] >=60:
			int_speed [i] = int_speed[i-1]
	return int_speed
        
def isHighwayTrip(velocity):
    count = 0
    for i in range(len(velocity)):
        if velocity[i] >= (12.5) :
            count += 1
    if count/(len(velocity)*1.) > 0.65:
        return 1
    return 0
	
def features(x,y):
    distance = calculateDistance(x,y)
    velocity = calculateDistance(x,y)
    acceleration = calculateAcceleration(velocity)
    angleH = calculateAngleHorizontal(x,y)
    angleT = calculateAngleChange(x,y,distance)
    correted_instant_speed = correctInstantSpeed(velocity)
    correted_instant_acc = calculateAcceleration(correted_instant_speed)
    
    return distance,velocity,acceleration,angleH,angleT,correted_instant_speed,correted_instant_acc

def extractTenSequenceTurnAngle(angleT):  
    angles = np.unique(angleT)
   
    dict_angle = {a:angleT.count(a) for a in angles}
    if len(dict_angle)>=12:
        ten_list_angle = sorted(dict_angle, key = dict_angle.get, reverse = True)[-12:-2]
    else:
        ten_list_angle = [0]*10

    return ten_list_angle

def extractTenSequenceAngleAndSpeed(angleT,speed):
    ten_angle = extractTenSequenceTurnAngle(angleT)
    speed_of_ten_angle = []
    for angle in ten_angle:
        speed_of_angle = []
        for i in range(len(angleT)):
            if angleT[i] == angle:
                speed_of_angle.append(speed[i])
                
        speed_of_ten_angle.append(round(np.mean(speed_of_angle),5))
    return ten_angle, speed_of_ten_angle

	
def extractPerOfAngleChange(angleT):
	count_straight = len([agl for agl in angleT if agl <10]) 
	count_10_45 = len([agl for agl in angleT if (agl >=10 and agl <45)]) 
	count_45_90 = len([agl for agl in angleT if (agl >=45 and agl <90)]) 
	count_90_120 = len([agl for agl in angleT if (agl >=90 and agl <120)]) 
	count_120_170 = len([agl for agl in angleT if (agl >=120 and agl <170)])
	count_back = len([agl for agl in angleT if (agl >=170)]) 
	n = len(angleT)
	per_straight = count_straight*100./n
	per_10_45 = count_10_45*100./n
	per_45_90 = count_45_90*100./n
	per_90_120 = count_90_120*100./n
	per_120_170 = count_120_170*100./n
	per_back = count_back*100./n
	
	return per_straight,per_10_45,per_45_90,per_90_120,per_120_170,per_back
	
def extractTenLargestAngleChangeAndSpeed(angleT, speed):
	angles = sorted(angleT)
	speed_of_ten_angle = []
	if len(angles)>=10:
		ten_angle = angles[-11:-1]
		for angle in ten_angle:
			index = angleT.index(angle)
			speed_of_ten_angle.append(speed[index])	
	else:
		ten_angle = [0]*10
		speed_of_ten_angle = [0]*10
	return ten_angle, speed_of_ten_angle

def countTimeOfAccelerationMoreThan4(acceleration):
	
	count3 = len([acc for acc in acceleration if acc >= 3]) 
	count35 = len([acc for acc in acceleration if acc >= 3.5]) 
	count4 = len([acc for acc in acceleration if acc >= 4]) 

	#print count3,count35,count4
	return count3,count35,count4
	
def countDiffBreakingAndAccelerating(acceleration):
	n = len(acceleration)*1.0
	
	acceleating_hard_15 = len([acc for  acc in np.diff(acceleration) if acc >= 1.5]) 
	breaking_hard_15 = len([acc for  acc in np.diff(acceleration) if acc <= -1.5]) 
	
	acceleating_hard_2 = len([acc for  acc in np.diff(acceleration) if acc >= 2]) 
	breaking_hard_2 = len([acc for  acc in np.diff(acceleration) if acc <= -2])  
	
	acceleating_hard_25 = len([acc for  acc in np.diff(acceleration) if acc >= 2.5]) 
	breaking_hard_25 = len([acc for  acc in np.diff(acceleration) if acc <= -2.5]) 
	

	acc_15 = round(100.*acceleating_hard_15/n,3)
	acc_2 = round(100.*acceleating_hard_2/n,3)
	acc_25 = round(100.*acceleating_hard_25/n,3)
	
	brak_15 = round(100.*breaking_hard_15/n,3)
	brak_2 = round(100.*breaking_hard_2/n,3)
	brak_25 = round(100.*breaking_hard_25/n,3)

	return 	acc_15,acc_2,acc_25,brak_15,brak_2,brak_25

def speedClassify (speeds):
	threshold = 45
	partition = 0
	i = 0
	speed_15 = []
	speed_11_15 = []
	speed_7_11 = []
	speed_5_7 = []
	speed_5 = []
	var_speed = []
	while i<(len(speeds)-threshold):
		sub_speed = speeds[i:i+threshold]
		avg_speed = np.mean(sub_speed)
		var_speed.append(np.var(sub_speed))
		if(avg_speed>=15):
			speed_15.append(avg_speed)
		elif(avg_speed>=11 and avg_speed<15):
			speed_11_15.append(avg_speed)
		elif(avg_speed>=7 and avg_speed<11):
			speed_7_11.append(avg_speed)
		elif(avg_speed>=5 and avg_speed<7):
			speed_5_7.append(avg_speed)
		else:
			speed_5.append(avg_speed)
		partition +=1
		i = i+threshold
	#print len(speed_15)/(partition*1.0)	
	n = partition*1.0
	per_of_speed_15 = round(100*len(speed_15)/n,3)
	per_of_speed_11_15 = round(100*len(speed_11_15)/n,3)
	per_of_speed_7_11 = round(100*len(speed_7_11)/n,3)
	per_of_speed_5_7 = round(100*len(speed_5_7)/n,3)
	per_of_speed_5 = round(100*len(speed_5)/n,3)
	max_var = round(np.max(var_speed),5)
	min_var = round(np.min(var_speed),5)
	
	return per_of_speed_15,per_of_speed_11_15,per_of_speed_7_11,per_of_speed_5_7,per_of_speed_5,max_var,min_var
'''	
def detectStopPosition(velocity):
	timeToStop = 5
	velocityToStop = 1
	stopPosition = [index for index, obj in enumerate(velocity) if obj < velocityToStop] 

	arrToStop = []
	subArrToStop = []
	for index, obj in enumerate(stopPosition):
		if index != len(stopPosition) - 1:
			if stopPosition[index] + 1 == stopPosition[index+1] :
				subArrToStop.append(obj)
			else:
				subArrToStop.append(obj)
				arrToStop.append(subArrToStop)
				subArrToStop = []
		else:
			subArrToStop.append(obj)
			arrToStop.append(subArrToStop)

	stopPosition = [obj for obj in arrToStop if len(obj) >= timeToStop]
	if len(stopPosition) == 0:
		return []
	#print stopPosition
	return np.concatenate(stopPosition)
 '''
def detectTurning(x, y):
    numOfPoints = 11
    angleTuring = 15
    arr = [range(i, i + numOfPoints) for i in range(len(x) - numOfPoints + 1)]
    angle = []
    for i in arr:
        dot = (x[i[numOfPoints/2]]-x[i[0]])*(x[i[-1]]-x[i[numOfPoints/2]]) + (y[i[numOfPoints/2]]-y[ i[0]])*(y[i[-1]]-y[i[numOfPoints/2]])
        d1 = ((x[i[numOfPoints/2]]-x[i[0]])**2 +(y[i[numOfPoints/2]]-y[i[0]])**2)**0.5
        d2 = ((x[i[-1]]-x[i[numOfPoints/2]])**2 +(y[i[-1]]-y[i[numOfPoints/2]])**2)**0.5
        if(d1*d2!=0):
            cos = dot/(d1*d2)
            if (np.abs(cos) > 1.0) :
                cos = 1.0   
            angle.append(np.arccos(cos)*180/np.pi)
        else:
            angle.append(0)
            
    turningPosition = np.array([i for i,x in enumerate(angle) if angleTuring <= x and x <= 180 - angleTuring])
    turningPosition = turningPosition + numOfPoints/2
    turningPosition = np.unique(np.concatenate([turningPosition, turningPosition+1, turningPosition+2, turningPosition+3]))
    
    return turningPosition

def detectTurningTime(turningPosition):
    if len(turningPosition) != 0:
        zerosArray = np.zeros(turningPosition[-1] + 2)
        zerosArray[turningPosition] = 1
        return len([obj for index, obj in enumerate(zerosArray) if obj == 1 and zerosArray[index+1] == 0])
    else:
        return 0

def calculateAvgSpeedInTurning(velocity, turningPosition):
	if len(turningPosition) == 0:
		return 0
	return round(np.average(np.array(velocity)[turningPosition]),3)
    
def calculateAvgVerlocityExcludingStop(velocity, stopPosition):
	notStopPosition = list(set(range(len(velocity))) - set(stopPosition))
	if len(notStopPosition) == 0:
		return 0
	return round(np.average(np.array(velocity)[notStopPosition]),5)

def calculateProportionOfAccelerationTime(acceleration):
    acceleration = np.array(acceleration)
    return round(len([obj for obj in (acceleration[1:] - acceleration[:-1]) if obj > 0]) * 100. / (len(acceleration) - 1), 3)

def calculateProportionOfDecelerationTime(acceleration):
    acceleration = np.array(acceleration)
    return round(len([obj for obj in (acceleration[1:] - acceleration[:-1]) if obj < 0]) * 100. / (len(acceleration) - 1), 3)

def calculateProportionOfTimeAtConstantSpeed(velocity):
    allowance = 1
    frame = 30
    arrConstantSpeed = []
    subArrConstantSpeed = [velocity[0]]
    for index, obj in enumerate(velocity):
        if index == 0:
            continue
        if index != len(velocity) - 1:
            if abs(obj - subArrConstantSpeed[0]) < allowance:
                subArrConstantSpeed.append(obj)
            else:
#                 subArrConstantSpeed.append(obj)
                arrConstantSpeed.append(subArrConstantSpeed)
                subArrConstantSpeed = [obj]
        else:
            subArrConstantSpeed.append(obj)
            arrConstantSpeed.append(subArrConstantSpeed)
    return round(sum(len(obj) for obj in arrConstantSpeed if len(obj) > frame) * 100. / len (velocity),5) 	

def timeTakenToCometoAStop(velocity):
    velocityToStop = [0.2, 0.3]
    #get all position have velocity < 1 (velocityToStop)
    stopPosition = [index for index, obj in enumerate(velocity) if  velocityToStop[0] < obj and obj < velocityToStop[1]] 
    acceptVelocity = .5
    arrToStop = [[0]]
    subArrToStop = []
 
    #segment continuous positions in a array
    for index, obj in enumerate(stopPosition):
        if index != len(stopPosition) - 1:
            if stopPosition[index] + 1 == stopPosition[index+1] :
                subArrToStop.append(obj)
            else:
                subArrToStop.append(obj)
                arrToStop.append(subArrToStop)
                subArrToStop = []
        else:
            subArrToStop.append(obj)
            arrToStop.append(subArrToStop)
            
    ################
    previousItem = arrToStop[0]
    if  len(arrToStop) - 1 == 0:
        return []
	
   # print arrToStop
    #check with each continuous positions
    arrTimeToStop = []
    for item in arrToStop[1:]:
        for i in reversed(range(previousItem[-1] + 1, item[0])):
            if velocity[i+1] < velocity[i] or velocity[i+1] - velocity[i] < acceptVelocity:
                item.insert(0, i)
            else:
                break
        previousItem = item
        if len([index for index, obj in enumerate(velocity[item]) if obj > 2.]) == 0:
            continue
        arrTimeToStop.append(item)
    return arrTimeToStop
    
def avgTimeTakenToCometoAStop(arrTimeToStop):
    if len(arrTimeToStop) == 0:
        return 0
    elif len(arrTimeToStop) == 1:
        return len(arrTimeToStop[0])
    else:
        return len(np.concatenate(arrTimeToStop)) * 1. / len(arrTimeToStop)
    
def maximumDecelerationTimeTakenToCometoAStop(arrTimeToStop, velocity):
    if len(arrTimeToStop) == 0:
        return 0
    return max([(velocity[item[-1]] - velocity[item[0]]) * 1. / len(item) for item in arrTimeToStop])

def detectDrivingPeriod(velocity,distance):
	#print velocity[:30]
	velocityToStop = 2
	stopPosition = [index for index, obj in enumerate(velocity) if obj < velocityToStop]
	#print stopPosition
	arrToStop = []
	subArrToStop = []
    
	for index, obj in enumerate(stopPosition):
		if index != len(stopPosition) - 1:
			if stopPosition[index] + 1 == stopPosition[index+1] :
				subArrToStop.append(obj)
			else:
				subArrToStop.append(obj)
				arrToStop.append(subArrToStop)
				subArrToStop = []
		else:
			subArrToStop.append(obj)
			arrToStop.append(subArrToStop)
    
	previousItem = arrToStop[0]
	if  len(arrToStop) - 1 == 0:
		return 0, 0, 0, 0, 0, 0, 0
		
	#check with each continuous positions
	drivingPeriod = []
	for item in arrToStop[1:]:
		duration = item[0] - previousItem[-1]
		if duration <=5:
			previousItem = item
			continue
			
		drivingPeriod.append(range(previousItem[-1]+1,item[0]))
		previousItem = item
		
	sum_list_distance = []	
	if len(drivingPeriod)==0:
		return 0, 0, 0, 0, 0, 0, 0	
		
	else:
		drivingPeriodLen = map(len, drivingPeriod)	
		#print drivingPeriodLen
		minDrivingPeriod = min(drivingPeriodLen)
		indexMinDrivingPeriod = drivingPeriodLen.index(minDrivingPeriod)
		maxDrivingPeriod = max(drivingPeriodLen)
		indexMaxDrivingPeriod = drivingPeriodLen.index(maxDrivingPeriod)
		
		#print drivingPeriod
		for drivingP in drivingPeriod:
			sum_list_distance.append(sum(distance[drivingP]))

		#print sum_list_distance
		
		var_min_dec = np.var(velocity[drivingPeriod[indexMinDrivingPeriod]])
		var_max_dec = np.var(velocity[drivingPeriod[indexMaxDrivingPeriod]])
		n_period = len(drivingPeriodLen)
		agv_period = np.mean(drivingPeriodLen)

	return n_period,agv_period,minDrivingPeriod,maxDrivingPeriod,var_min_dec,var_max_dec,sum_list_distance
	
def detectStopTimeDuration(velocity):
	timeToStop = 5
	velocityToStop = 1
	stopPosition = [index for index, obj in enumerate(velocity) if obj < velocityToStop] 

	arrToStop = []
	subArrToStop = []
	for index, obj in enumerate(stopPosition):
		if index != len(stopPosition) - 1:
			if stopPosition[index] + 1 == stopPosition[index+1] :
				subArrToStop.append(obj)
			else:
				subArrToStop.append(obj)
				arrToStop.append(subArrToStop)
				subArrToStop = []
		else:
			subArrToStop.append(obj)
			arrToStop.append(subArrToStop)

	stopPosition = [obj for obj in arrToStop if len(obj) >= timeToStop]
	if len(stopPosition) == 0:
		return 0,0,[]
	#print stopPosition
	
	duration_stop = map(len,stopPosition)
	#print duration_stop	
	var_dur_stop = np.var(duration_stop)
	std_dur_stop = np.std(duration_stop)
	
	return var_dur_stop, std_dur_stop,np.concatenate(stopPosition)
	
def calculateProportionAccAndDecBelow1(accereration, stop_pos):
	# proportion acceleration below 1
	
	acc_below1 = [acc for index, acc in enumerate(accereration) if (acc <=1 and acc>0 and index not in (stop_pos))] 
	dec_below1 = [acc for index, acc in enumerate(accereration) if (acc <0 and acc>=-1 and index not in (stop_pos))] 
	# Exclude stop position
	n = len([acc for index, acc in enumerate(accereration) if index not in (stop_pos)])
	if n == 0:
		return 0,0
		
	pro_acc_below1 = round(len(acc_below1)*100./n,5)
	pro_dec_below1 = round(len(dec_below1)*100./n,5)
	#print pro_acc_below1,pro_dec_below1
	return pro_acc_below1,pro_dec_below1
	
def extractFeaturesOfTrip(filename):
	# Read data of each trip
	x,y = openFile(filename)
	
	# calculate simple features
	distance,velocity,acceleration,angleH,angleT,corrected_instant_speed,corrected_instant_acc = features(x,y)
	list_feature = []
			
	# calculate avg summary features
	avg_speed = np.mean(distance)
	avg_acc = np.mean(acceleration)
	avg_corrected_speed = np.mean(corrected_instant_speed)
	avg_corrected_acc = np.mean(corrected_instant_acc)
	avg_angle_change = np.mean(angleT)
	avg_distance = np.mean(distance)
	# calculate summary or max value features
	total_distance = sum(distance)
	fastest_speed = max(corrected_instant_speed)
	longest_distance = max(distance)
	max_angle_change = max(angleT)

	time_sp_up,per_sp_up = detectSpeedUp(np.array(corrected_instant_acc))
	time_sp_down,per_sp_down = detectSpeedDown(np.array(corrected_instant_acc))
	
	highway = isHighwayTrip(corrected_instant_speed)
	num_of_acc_3,num_of_acc_35,num_of_acc_4 = countTimeOfAccelerationMoreThan4(corrected_instant_acc)

	# ten largest turn angle in a trip and their speed
	ten_angle, speed_of_ten_angle = extractTenLargestAngleChangeAndSpeed(angleT,corrected_instant_speed)
	#print speed_of_ten_angle
	acc_15,acc_2,acc_25,brak_15,brak_2,brak_25 = countDiffBreakingAndAccelerating(corrected_instant_acc)
	per_of_speed_15,per_of_speed_11_15,per_of_speed_7_11,per_of_speed_5_7,per_of_speed_5,max_var,min_var = speedClassify(corrected_instant_speed)
	
	# Turning time
	turningPosition = detectTurning(x,y)
	time_of_turning = detectTurningTime(turningPosition)
	avg_speed_turning = calculateAvgSpeedInTurning(corrected_instant_speed, turningPosition)
	# calculate Proportion
	#stopPosition = detectStopPosition(corrected_instant_speed)
	var_dur_stop, std_dur_stop, stop_dur_pos = detectStopTimeDuration(np.array(corrected_instant_speed))
	avg_speed_excluding_stop = calculateAvgVerlocityExcludingStop(corrected_instant_speed, stop_dur_pos)
	
	proportion_time = calculateProportionOfTimeAtConstantSpeed_1(corrected_instant_speed, corrected_instant_acc)
	#speed_fil, acc_fil = filterSpeedAndAcc(corrected_instant_speed)
	speed_fil = 0
	acc_fil = 0
	# type is array
	
	arr_time_to_stop = timeTakenToCometoAStop(np.array(corrected_instant_speed))
	time_taken_stop = avgTimeTakenToCometoAStop(arr_time_to_stop)
	max_dec = maximumDecelerationTimeTakenToCometoAStop(arr_time_to_stop, corrected_instant_speed)
	num_stop,avg_num_period,min_period,max_period,var_min_p,var_max_p,dist_tra_bet_stop  = detectDrivingPeriod(np.array(corrected_instant_speed),np.array(distance))
	avg_dist_traveled = np.mean(dist_tra_bet_stop)
	sd_distance = np.std(dist_tra_bet_stop)
	
	pro_acc_below1,pro_dec_below1 = calculateProportionAccAndDecBelow1(corrected_instant_acc,stop_dur_pos)
	
	per_straight,per_10_45,per_45_90,per_90_120,per_120_170,per_back = extractPerOfAngleChange(angleT)
	
	list_feature.append(len(x))
	#list_feature.append(round(avg_speed,5))
	list_feature.append(avg_speed)
	list_feature.append(round(avg_acc,5))
	list_feature.append(avg_corrected_speed)
	#list_feature.append(round(avg_corrected_speed,5))
	list_feature.append(round(avg_corrected_acc,5))
	list_feature.append(round(avg_angle_change,3))
	list_feature.append(round(avg_distance,5))

	list_feature.append(round(total_distance,5))
	list_feature.append(round(fastest_speed,5))
	list_feature.append(round(longest_distance,5))
	list_feature.append(round(max_angle_change,3))

	list_feature.append(time_sp_up)
	list_feature.append(time_sp_down)
	
	list_feature.append(per_sp_up)
	list_feature.append(per_sp_down)
	
	list_feature.append(np.var(corrected_instant_speed))
	list_feature.append(np.var(corrected_instant_acc))

	list_feature.append(highway)

	# number of acc > 4 
	list_feature.append(num_of_acc_4)

	# breaking and accelerating hard
	list_feature.append(acc_2)
	list_feature.append(brak_2)
	# ten largest angle change and speed
	list_feature.extend(ten_angle)
	list_feature.extend(speed_of_ten_angle)
	
	list_feature.append(round(np.mean(speed_fil),5))
	list_feature.append(round(np.mean(acc_fil),5))
	
	list_feature.append(per_of_speed_15)
	list_feature.append(per_of_speed_11_15)
	list_feature.append(per_of_speed_7_11)
	list_feature.append(per_of_speed_5_7)
	list_feature.append(per_of_speed_5)
	list_feature.append(min_var)
	list_feature.append(max_var)
	
	list_feature.append(time_of_turning)
	list_feature.append(avg_speed_turning)
	list_feature.append(avg_speed_excluding_stop)
	list_feature.extend(proportion_time)
	
	list_feature.append(round(time_taken_stop,2))
	list_feature.append(max_dec)
	
	list_feature.append(num_stop)
	list_feature.append(avg_num_period)
	list_feature.append(min_period)
	list_feature.append(max_period)
	list_feature.append(var_min_p)
	list_feature.append(var_max_p)
	
	# standard deviation of distance between stops 
	list_feature.append(sd_distance)
	list_feature.append(avg_dist_traveled)
	
	#########
	
	list_feature.append(var_dur_stop)
	list_feature.append(std_dur_stop)
	list_feature.append(pro_acc_below1)
	list_feature.append(pro_dec_below1)
	
	list_feature.append(num_of_acc_3)
	list_feature.append(num_of_acc_35)
	
	list_feature.append(acc_15)
	list_feature.append(brak_15)
	list_feature.append(acc_25)
	list_feature.append(brak_25)
	
	list_feature.append(per_straight)
	list_feature.append(per_10_45)
	list_feature.append(per_45_90)
	list_feature.append(per_90_120)
	list_feature.append(per_120_170)
	list_feature.append(per_back)

	return list_feature
   
#Wilson do this feature
def detectSpeedUp(acceleration):
	timeSpeedUp = 5
	# Acceleration = 5
	posion_Acceleration = np.where(acceleration > 0)
	count = 1
	fist = posion_Acceleration[0][0]		
	SpeedUp_time = {}
	for item in posion_Acceleration[0][1::]:
		if count >= timeSpeedUp:
			if item == fist+count and item != posion_Acceleration[0][-1]:
				count += 1
			else:
				SpeedUp_time[fist] = count
				count = 1
				fist = item
		else:
			if item == fist+count:
				count += 1
			else:
				count = 1
				fist = item
	time_sp_up = len(SpeedUp_time)
	per_sp_up = round(time_sp_up*100.0/len(acceleration),3)
	return time_sp_up,per_sp_up

#Bao do this feature
def detectSpeedDown(acceleration):
	timeSpeedDown = 5
	
	posion_Acceleration = np.where(acceleration < 0)
	count = 1
	fist = posion_Acceleration[0][0]		
	SpeedDown_time = {}
	for item in posion_Acceleration[0][1::]:
		if count >= timeSpeedDown:
			if item == fist+count and item != posion_Acceleration[0][-1]:
				count += 1
			else:
				SpeedDown_time[fist] = count
				count = 1
				fist = item
		else:
			if item == fist+count:
				count += 1
			else:
				count = 1
				fist = item
	time_sp_down = len(SpeedDown_time)
	per_sp_down = round(time_sp_down*100.0/len(acceleration),3)
	return time_sp_down,per_sp_down
	
def calculateProportionOfTimeAtConstantSpeed_1(velocity, acceleration):
    velocity = np.array(velocity)
    allowance = 1
    frames = [10]
    returnValue = []
    for frame in frames:
        constantSpeedPosition = []
        listVelocity = list(chunks(velocity, frame))
        for index, obj in enumerate(listVelocity):
            minSubVelocity = min(obj)
            maxSubVelocity = max(obj)
            if (maxSubVelocity - minSubVelocity) < allowance:
                constantSpeedPosition.extend(range(index * frame, (index + 1) * len(obj)))
        
        accelerationPosition = set([index for index, obj in enumerate(acceleration) if obj >= 0]) - set(constantSpeedPosition)
        decelerationPosition = set([index for index, obj in enumerate(acceleration) if obj < 0]) - set(constantSpeedPosition)
        constantSpeed = round(len(constantSpeedPosition) * 100. / len(velocity), 3)
        accelerationSpeed = round(len(accelerationPosition) * 100. / len(velocity), 3)
        decelerationSpeed = round(len(decelerationPosition) * 100. / len(velocity), 3)
        returnValue.append([constantSpeed, accelerationSpeed, decelerationSpeed])

    return np.concatenate(returnValue)
        
def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]       

def filterSpeedAndAcc(speed):
	w_size = 11
	speed_fil = [0]
	for i in range(len(speed)-w_size):
		sp = np.mean(speed[i:i+w_size])
		speed_fil.append(sp)
		
	#speed_fill.extend([0]*10)
	acc_fil = [0]*2
	sp_diff = np.diff(speed_fil)
	acc_fil.extend(sp_diff)
	
	return speed_fil, acc_fil
	
def main():

	path = "F:/Kaggle/Driver Telematics Analysis/data/drivers/"
	#path = "F:/Kaggle/Driver Telematics Analysis/drivers/data_test/"
	path_out = "F:/Kaggle/Driver Telematics Analysis/features_all_18/"
	start_time = time.time()

	listDrivers = os.listdir(path)
	step = 1
	id = 1
	
	for driver in listDrivers:
		
		with open(path_out+driver+'.csv','wb') as f:
			writer = csv.writer(f)
			writer.writerow(['id','driver','trip','driver_trip','duration','avg_speed','avg_acc',\
					'avg_corrected_speed', 'avg_corrected_acc',\
					'avg_angle_change','avg_distance','total_distance','fastest_speed','longest_distance',\
					'max_angle_change','times_speed_up','times_speed_down','per_speed_up','per_speed_down',\
					'var_corrected_instant_speed',\
					'var_corrected_instant_acc','highway','num_of_acc_4',\
					'hard_acc2','hard_brak2',\
					'angle1','angle2','angle3','angle4','angle5','angle6','angle7','angle8','angle9','angle10',\
					'speed1','speed2','speed3','speed4','speed5','speed6','speed7','speed8','speed9','speed10',\
					'speed_fil','acc_fil',
					'per_of_speed_15', 'per_of_speed_11_15', 'per_of_speed_7_11', 'per_of_speed_5_7','per_of_speed_5',\
					'min_var','max_var','time_of_turning', 'avg_speed_turning','avg_speed_excluding_stop',\
					'10s_constant','10s_acceleration','10s_deceleration','time_taken_stop', 'max_dec', \
					'num_stop', 'avg_num_period','min_period','max_period','var_min_p','var_max_p',\
					'sd_distance', 'avg_dist_traveled','var_dur_stop','std_dur_stop','pro_acc_below1','pro_dec_below1',\
					'num_of_acc_3','num_of_acc_35','hard_acc15','hard_brak15','hard_acc25','hard_brak25',\
					'per_straight','per_10_45','per_45_90','per_90_120','per_120_170','per_back'])
			
			if step%10==0:
				print ('-%s seconds --- %s driver --- %s' %(str(time.time() - start_time),driver,step))
			for trip in os.listdir(path+str(driver)):
				
				driver_trip = str(driver)+'_'+trip[:-4]	
				#print driver_trip
				tripfeature = extractFeaturesOfTrip(path+str(driver)+'/'+trip)
				#print len(tripfeature)
				
				#break
				writer.writerow([id,driver,trip[:-4],driver_trip,tripfeature[0],tripfeature[1],tripfeature[2],\
								tripfeature[3],\
								tripfeature[4],tripfeature[5],tripfeature[6],tripfeature[7],tripfeature[8],\
								tripfeature[9],tripfeature[10],tripfeature[11],tripfeature[12],tripfeature[13],\
								tripfeature[14],tripfeature[15],tripfeature[16],tripfeature[17],tripfeature[18],\
								tripfeature[19],tripfeature[20],tripfeature[21],tripfeature[22],tripfeature[23],\
								tripfeature[24],tripfeature[25],tripfeature[26],tripfeature[27],tripfeature[28],\
								tripfeature[29],tripfeature[30],tripfeature[31],tripfeature[32],tripfeature[33],\
								tripfeature[34],tripfeature[35],tripfeature[36],tripfeature[37],tripfeature[38],\
								tripfeature[39],tripfeature[40],tripfeature[41],tripfeature[42],tripfeature[43],\
								tripfeature[44],tripfeature[45],tripfeature[46],tripfeature[47],tripfeature[48],\
								tripfeature[49],tripfeature[50],tripfeature[51],tripfeature[52],tripfeature[53],\
								tripfeature[54],tripfeature[55],tripfeature[56],tripfeature[57],tripfeature[58],\
								tripfeature[59],tripfeature[60],tripfeature[61],tripfeature[62],tripfeature[63],\
								tripfeature[64],tripfeature[65],tripfeature[66],tripfeature[67],tripfeature[68],\
								tripfeature[69],tripfeature[70],tripfeature[71],tripfeature[72],tripfeature[73],\
								tripfeature[74],tripfeature[75],tripfeature[76],tripfeature[77],tripfeature[78],
								tripfeature[79],tripfeature[80],tripfeature[81]])
				id = id + 1
		step = step+1

	print "Done"
	f.close()

main()

