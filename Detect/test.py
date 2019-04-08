import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA


#trainData = pd.read_csv("kddcup.data.corrected")
trainData = pd.read_csv("kddcup.data_10_percent_corrected")
testData = pd.read_csv("test.corrected")

trainLabel = trainData.iloc[:,-1]

trainFeatures = trainData.drop(columns=['0' , '1' , '2' , '3' ,  str(trainData.shape[1]-1)])
#testFeatures = testData.drop(columns=['0' , '1' , '2' , '3' ,  str(testData.shape[1]-1)])

##using Ipca on 10% test Funtion

row_size =trainFeatures.shape[0]
chunk_size = 500
ipca = IncrementalPCA(n_components=20 ,  batch_size=500)
for i in range(0, row_size//chunk_size):
    print("For I = " + str(i))
    tempFeatures = trainFeatures[i*chunk_size : (i+1)*chunk_size]
    x_std_Train = StandardScaler().fit_transform(tempFeatures)
    print(x_std_Train.shape)
    # if(x_std_Train.values.np)

    print("From "+ str(i*chunk_size))
    print("To " + str((i+1)*chunk_size))
    # print("Shape of std is " + str(x_std_Train.shape))
    ipca.partial_fit(x_std_Train)

print(ipca.explained_variance_ratio_)
print(sum(ipca.explained_variance_ratio_))







