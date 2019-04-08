import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA

trainData = pd.read_csv("kddcup.data_10_percent_corrected")
testData = pd.read_csv("test.corrected")

trainLabel = trainData.iloc[:,-1]

trainFeatures = trainData.drop(columns=['0' , '1' , '2' , '3' ,  str(trainData.shape[1]-1)])

##using Ipca on 10% test Funtion

row_size =trainFeatures.shape[0]
chunk_size = 500
ipca = IncrementalPCA(n_components=20 ,  batch_size=500)
for i in range(0, row_size//chunk_size):
    tempFeatures = trainFeatures[i*chunk_size : (i+1)*chunk_size]
    x_std_Train = StandardScaler().fit_transform(tempFeatures)
    # print("Shape of std is " + str(x_std_Train.shape))
    X_Ipca = ipca.partial_fit(x_std_Train)


x = ipca.explained_variance_ratio_
PCA_Label = ['PC0','PC1' , 'PC2' , 'PC3' , 'PC4' , 'PC5' , 'PC6', 'PC7', 'PC8' , 'PC9' , 'PC10' , 'PC11' , 'PC12', 'PC13', 'PC14', 'PC15', 'PC16', 'PC18' , 'PC19' ,'PC20']

plt.plot(PCA_Label , x)
plt.xlabel("PCA")
plt.ylabel("Variance")
plt.title("PCA compoent variance")
print(x)
print("The Sum of the variance is " + str(sum(ipca.explained_variance_ratio_)))


plt.show()






