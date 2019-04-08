import pandas as pd
import numpy as np
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import confusion_matrix

from sklearn import  metrics
from sklearn.metrics import accuracy_score


trainData = pd.read_csv("kddcup.data_10_percent_corrected")
testData = pd.read_csv("test.corrected")



# #################################
# ## Without Dimension Reduction ##
# ################################# 
# start = time.time()

# trainFeatures = trainData.drop(columns=[str(trainData.shape[1]-1)])
# trainFeatures = pd.get_dummies(trainFeatures)

# testFeatures = testData.drop(columns=[str(testData.shape[1]-1)])
# testFeatures = pd.get_dummies(testFeatures)

# trainDTest = trainFeatures.columns.difference(testFeatures.columns)
# # print(trainDTest)
# # print(len(trainDTest))
# if len(trainDTest) is not 0:
#     for difTrain in trainDTest:
#         testFeatures[str(difTrain)] = 0
#         # print(difTrain)

# testDTrain = testFeatures.columns.difference(trainFeatures.columns)
# # print(testDTrain)
# # print(len(testDTrain))
# if len(testDTrain) is not 0:
#     for difTest in testDTrain:
#         trainFeatures[str(difTest)] = 0
#         # print(difTest)

# # print(trainFeatures)
# # print(testFeatures)

# trainFeaturesNP = np.array(trainFeatures)
# # print(trainFeaturesNP)
# testFeaturesNP = np.array(testFeatures)
# # print(testFeaturesNP)

# trainLabels = pd.DataFrame(trainData, columns=[str(trainData.shape[1]-1)])
# trainLabels = np.array(trainLabels)

# testLabels = pd.DataFrame(testData, columns=[str(testData.shape[1]-1)])
# testLabels = np.array(testLabels)

# print(testFeatures)

# gnb = MultinomialNB().fit(trainFeaturesNP , trainLabels)
# #gnb = GaussianNB().fit(trainFeaturesNP, trainLabels) 
# gnb_predictions = gnb.predict(testFeaturesNP) 
  
# # accuracy on X_test 
# accuracy = gnb.score(testFeaturesNP, testLabels) 
# print ("The accuracy without dimension reduction " + str(accuracy)) 

# end = time.time()
# print("Time taken " + str(end - start))

# # creating a confusion matrix 
# cm = confusion_matrix(testLabels, gnb_predictions) 


###############################
## After Dimension Reduction ##
###############################
start = time.time()

trainFeatures = trainData.drop(columns=['0' , '6' , '7' , '8' , '9' , '10', '12', '13', '14', '16', '17', '18', '19', '20', '21', '26', '27', '30', '39' , '40' ,  str(trainData.shape[1]-1)])
trainFeatures = pd.get_dummies(trainFeatures)

testFeatures= testData.drop(columns=['0' , '6' , '7' , '8' , '9' , '10', '12', '13', '14', '16', '17', '18', '19', '20', '21', '26', '27', '30', '39' , '40' ,  str(trainData.shape[1]-1)])
testFeatures = pd.get_dummies(testFeatures)

trainDTest = trainFeatures.columns.difference(testFeatures.columns)
# print(trainDTest)
# print(len(trainDTest))
if len(trainDTest) is not 0:
    for difTrain in trainDTest:
        testFeatures[str(difTrain)] = 0
        # print(difTrain)

testDTrain = testFeatures.columns.difference(trainFeatures.columns)
# print(testDTrain)
# print(len(testDTrain))
if len(testDTrain) is not 0:
    for difTest in testDTrain:
        trainFeatures[str(difTest)] = 0
        # print(difTest)


#####################################################THSI IS ADDED RIGHT NOW
trainFeatures = trainFeatures.sort_index(axis = 1)
testFeatures = testFeatures.sort_index(axis = 1)


trainFeaturesNP = np.array(trainFeatures)
# print(trainFeaturesNP)
testFeaturesNP = np.array(testFeatures)
# print(testFeaturesNP)

trainLabels = pd.DataFrame(trainData, columns=[str(trainData.shape[1]-1)])
trainLabels = np.array(trainLabels)

testLabels = pd.DataFrame(testData, columns=[str(testData.shape[1]-1)])
testLabels = np.array(testLabels)

gnb = GaussianNB().fit(trainFeaturesNP, trainLabels) 
gnb_predictions = gnb.predict(testFeaturesNP) 
  
# accuracy on X_test 
accuracy = gnb.score(testFeaturesNP, testLabels) 
print ("\n The accuracy with dimension reduction :" + str(accuracy)) 
  
# creating a confusion matrix 
cm = confusion_matrix(testLabels, gnb_predictions) 

end = time.time()
print("Time taken " + str(end - start))