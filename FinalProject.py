from pyspark import SparkConf,SparkContext
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest

import numpy as np
import re

path = "file:///Users/stellali/Desktop/Data/anonymisedData/"

#fileStudent_Assessment = path+"studentAssessment.csv"
fileStudent_Registration= path+"studentRegistration.csv"
fileVLE = path + "vle.csv"
#fileCourses = path + "courses.csv"
fileStudent_Info = path + "studentInfo.csv"
fileStudent_Vle = path + "studentVle.csv"

def getRDD(mfile, sc):
	lines = sc.textFile(mfile)
	linesHeader = lines.first()
	header = sc.parallelize([linesHeader])
	linesWithOutHeader = lines.subtract(header)
	myRDD = linesWithOutHeader.map(lambda x: quote.sub('', x).split(','))
	return myRDD

def buildArray(clickHistory):
	returnVal = np.zeros (100)
	indexlist = len(clickHistory)
   	for index in range(indexlist):
   		i = clickHistory[index][0]
   		returnVal[i] = int(clickHistory[index][1])
   	return returnVal

def getLastItem(itemlist):
	l = len(itemlist)
	largest = (0,0)
	for i in range(l):
		if itemlist[i][1] > largest[1]: largest = itemlist[i]
	return largest



if __name__ == "__main__":
		

		sc = SparkContext()
		quote = re.compile('\"')

		# Create RDDs from files
		#Student_Assessment = getRDD(fileStudent_Assessment, sc)
		#Courses = getRDD(fileCourses, sc)
		Student_Course = getRDD(fileStudent_Registration, sc) #student registration info
		VLE = getRDD(fileVLE, sc) #course material info
		Student_Info = getRDD(fileStudent_Info, sc) #student demographic info
		Student_VLE = getRDD(fileStudent_Vle, sc) #students' interaction with materials

		#filter to get BBB 2014B student, their final result and some demo info
		#studentID, (result, sex, edu, age, disability)
		BBBstudent = Student_Info.filter(lambda x: x[0]=="BBB" and x[1] == "2014B")\
			.map(lambda x: (x[2], (x[11], x[3], str(x[5]), x[7], x[10])))

		#need to transfer categorical variables to numeric...
		#sex
		BBBstudent = BBBstudent.map(lambda x: (x[0], (x[1][0], 1, x[1][2], x[1][3], x[1][4])) if x[1][1] == "F"\
			else (x[0], (x[1][0], 0, x[1][2], x[1][3], x[1][4])))
		#disability
		BBBstudent = BBBstudent.map(lambda x: (x[0], (x[1][0:4], 1)) if x[1][4] == "Y" else (x[0], (x[1][0:4], 0)))
		#edu
		BBBstudent = BBBstudent.map(lambda x: (x[0], (x[1][0][0], x[1][0][1], 0, x[1][0][3], x[1][1])) if x[1][0][2].startswith("N")\
			else (x[0], (x[1][0][0], x[1][0][1], 1, x[1][0][3], x[1][1])) if x[1][0][2].startswith("L")\
			else (x[0], (x[1][0][0], x[1][0][1], 2, x[1][0][3], x[1][1])) if x[1][0][2].startswith("A")\
			else (x[0], (x[1][0][0], x[1][0][1], 3, x[1][0][3], x[1][1])) if x[1][0][2].startswith("H")\
			else (x[0], (x[1][0][0], x[1][0][1], 4, x[1][0][3], x[1][1])))
		#age
		BBBstudent = BBBstudent.map(lambda x: (x[0], (x[1][0], x[1][1], x[1][2], 0, x[1][4])) if x[1][3] == "0-35"\
			else (x[0], (x[1][0], x[1][1], x[1][2], 1, x[1][4])) if x[1][3] == "35-55"\
			else (x[0], (x[1][0], x[1][1], x[1][2], 2, x[1][4]))) 


		#BBBwithdrawn = BBBstudent.filter(lambda x: x[1][0] == "Withdrawn") #students who withdrawn from the course, 490
		BBBcompleted = BBBstudent.filter(lambda x: x[1][0] != "Withdrawn") #students who completed the course, 1123

		#1. Analyze what factors/materials are most important for predicting whether a student will get pass/fail/disctinct

		#For students who finish the course, get their student IDs
		studentCompleted = BBBstudent.filter(lambda x: x[1][0] != "Withdrawn").keys().collect()
		#studentCompleted.count() 1123 students completed the course

		#Get BBB2014B VLE info (studentID, (materialID, date, numClick))
		BBB_VLE = Student_VLE.filter(lambda x: x[0]=="BBB" and x[1] == "2014B").map(lambda x: (x[2], (x[3], x[4], x[5])))

		#For students who completed the course, find the materials they have most interacted
		studentVLEcount = BBB_VLE.filter(lambda x: x[0] in studentCompleted).map(lambda x: (x[1][0], x[0]))\
			.distinct().map(lambda x: (x[0], 1)).reduceByKey(lambda x,y: x + y)
		topVLE = studentVLEcount.top(100, lambda x: x[1])
		VLEindex = sc.parallelize(range(100))
		#(materialID, index)
		VLE_list = VLEindex.map(lambda x: (topVLE[x][0], x))

		#For students who completed the course, count their total time of interacting with a certain material
		#(materialID, (studentID, numclick))
		BBB_VLE_completed = BBB_VLE.filter(lambda x: x[0] in studentCompleted)\
			.map(lambda x: ((x[0], x[1][0]), x[1][2])).map(lambda x: (x[0], int(x[1]))).reduceByKey(lambda x, y: x+y)\
			.map(lambda x: (x[0][1], (x[0][0], x[1])))
		#VLElist = VLE.filter(lambda x: x[1]=="BBB" and x[2] == "2014B").map(lambda x: x[0]).collect()  #there are 311 different VLEs in BBB 2014B
		
		#join, then get (materialID, (materialIndx, (studentID, numClick))) -> studentID, (materialIndex, numClick)
		VLE_completed = VLE_list.join(BBB_VLE_completed).map(lambda x: (x[1][1][0], (x[1][0], x[1][1][1])))\
			.groupByKey().map(lambda x : (x[0], list(x[1])))
		studentVLEclick = VLE_completed.map(lambda x: (x[0], buildArray(x[1])))

		#gather all information
		studentFull = BBBcompleted.join(studentVLEclick).map(lambda x: (x[0], (list(x[1][0]), x[1][1])))\
			.map(lambda x: (x[0], list(x[1]))).map(lambda x: (x[0],np.concatenate((x[1][0], x[1][1]), axis=None)))

		#seperate train and test data
		trainID = studentFull.keys().sample(False, 0.7, 1017).collect()
		
		#2 labels
		traindata1 = studentFull.filter(lambda x: x[0] in trainID).map(lambda x: (0, x[1]) if x[1][0] == "Fail" else (1, x[1]))\
			.map(lambda x: (x[0], np.delete(x[1], 0))).map(lambda x: LabeledPoint(x[0], x[1]))

		testdata1 = studentFull.filter(lambda x: x[0] not in trainID).map(lambda x: (0, x[1]) if x[1][0] == "Fail" else (1, x[1]))\
			.map(lambda x: (x[0], np.delete(x[1], 0)))

		md_lr1 = LogisticRegressionWithLBFGS.train(traindata1, iterations=100)
		lr_labelandpred1 = testdata1.map(lambda x: (x[0], md_lr1.predict(x[1])))
		# lr_labelandpred1.filter(lambda x: x[0] == x[1] and x[0] == 0).count() 82
		# lr_labelandpred1.filter(lambda x: x[0] == x[1] and x[0] == 1).count() 183
		# lr_labelandpred1.filter(lambda x: x[1] == 0 and x[0] == 1).count() 21
		# lr_labelandpred1.filter(lambda x: x[1] == 1 and x[0] == 0).count() 36
		coef = md_lr1.weights.toArray()
		ind = np.argpartition(coef, -10)[-10:]
		ind = ind - 4
		importantVLE = VLE_list.filter(lambda x: x[1] in ind).keys().collect()

		#3 labels
		traindata = studentFull.filter(lambda x: x[0] in trainID).map(lambda x: (0, x[1]) if x[1][0] == "Fail"\
			else (1, x[1]) if x[1][0] == "Pass" else (2, x[1]))\
			.map(lambda x: (x[0], np.delete(x[1], 0))).map(lambda x: LabeledPoint(x[0], x[1]))

		testdata = studentFull.filter(lambda x: x[0] not in trainID).map(lambda x: (0, x[1]) if x[1][0] == "Fail"\
			else (1, x[1]) if x[1][0] == "Pass" else (2, x[1]))\
			.map(lambda x: (x[0], np.delete(x[1], 0)))

		rfm = RandomForest.trainClassifier(traindata, 3, {0:2, 1:5, 2:3, 3:2}, 3, seed = 0527)
		rdd = testdata.map(lambda x: x[1])
		pred = rfm.predict(rdd)
		rf_labelandpred = testdata.map(lambda x: x[0]).zip(pred)
		# rf_labelandpred.filter(lambda x: x[0]==x[1] and x[0]==0).count() 79
		# rf_labelandpred.filter(lambda x: x[0]==x[1] and x[0]==1).count() 138
		# rf_labelandpred.filter(lambda x: x[0]==x[1] and x[0]==2).count() 4
		# rf_labelandpred.filter(lambda x: x[0]==0 and x[1]==1).count() 38
		# rf_labelandpred.filter(lambda x: x[0]==0 and x[1]==2).count() 1
		# rf_labelandpred.filter(lambda x: x[0]==1 and x[1]==2).count() 5
		# rf_labelandpred.filter(lambda x: x[0]==1 and x[1]==0).count() 7
		# rf_labelandpred.filter(lambda x: x[0]==2 and x[1]==0).count() 0
		# rf_labelandpred.filter(lambda x: x[0]==2 and x[1]==1).count() 50

		md_lr = LogisticRegressionWithLBFGS.train(traindata, iterations=100, numClasses=3)
		lr_labelandpred = testdata.map(lambda x: (x[0], md_lr.predict(x[1])))
		#lr_labelandpred.filter(lambda x: x[0] == x[1] and x[0] == 0).count() 83
		# lr_labelandpred.filter(lambda x: x[0] == x[1] and x[0] == 1).count() 112
		# lr_labelandpred.filter(lambda x: x[0] == x[1] and x[0] == 2).count() 10
		# lr_labelandpred.filter(lambda x: x[0] == 0 and x[1] ==1).count() 27
		# lr_labelandpred.filter(lambda x: x[0] == 0 and x[1] ==2).count() 8
		# lr_labelandpred.filter(lambda x: x[0] == 1 and x[1] ==2).count() 22
		# lr_labelandpred.filter(lambda x: x[0] == 1 and x[1] ==0).count() 16
		# lr_labelandpred.filter(lambda x: x[0] == 2 and x[1] ==0).count() 6
		# lr_labelandpred.filter(lambda x: x[0] == 2 and x[1] ==1).count() 38


		#2. For students who withdrawn, check where did most students stopped. 

		#For students who withdrawn from the course, get their info as well as the time when they register and unregister
		#Get students withdrawn date: studentID, withdrawn time
		StudentWithdrawnTime = Student_Course.filter(lambda x: x[0]=="BBB" and x[1] == "2014B")\
			.map(lambda x: (x[2], x[4])).filter(lambda x: x[1] != '')\
			.map(lambda x: (x[0], int(x[1]))).filter(lambda x: int(x[1]) > 0)
		StudentWithdrawnTime.values().histogram(32)
		#[18, 13, 8, 3, 7, 6, 13, 7, 3, 5, 7, 16, 5, 4, 6, 9, 3, 3, 4, 3, 3, 2, 7, 3, 6, 6, 2, 6, 3, 5, 6, 8])
		wdStudentID = StudentWithdrawnTime.keys().collect()
		#studentID, (materialID, date)
		lastTouch = BBB_VLE.filter(lambda x: x[0] in wdStudentID).map(lambda x: (x[0], (x[1][0], int(x[1][1]))))\
			.groupByKey().map(lambda x : (x[0], list(x[1]))).map(lambda x: (x[0], getLastItem(x[1])))
		#(materialID, timeDiff)
		lastItemandTimeDiff = lastTouch.join(StudentWithdrawnTime).map(lambda x: (x[1][0][0], x[1][1] - x[1][0][1]))
		lastItems = lastItemandTimeDiff.map(lambda x: (x[0], 1)).reduceByKey(lambda x,y: x+y).sortBy(lambda x: x[1]).collect()
		#print(lastItems[-1]) #'768351', 84 students read it and then withdrawn
		lastItemandTimeDiff.filter(lambda x: x[1] > 0).values().stats()
		#(count: 155, mean: 33.5870967742, stdev: 40.5261822333, max: 190.0, min: 1.0)
		lastItemandTimeDiff.filter(lambda x: x[1] > 0).values().histogram(27)
		#[44, 26, 13, 15, 8, 10, 5, 5, 3, 3, 4, 2, 2, 0, 1, 2, 2, 1, 2, 1, 1, 1, 1, 0, 2, 0, 1]


		

