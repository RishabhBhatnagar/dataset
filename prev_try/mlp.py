from pandas import DataFrame, read_csv
from matplotlib import pyplot as plt

def normalise(data):
    mi = data.min()
    ma = data.max()
    diff = ma - mi
    return data.apply(lambda x: (x-mi)/diff)


df = read_csv('train_extracted_feature.csv')

header = list(df)
header.pop(header.index('essay'))

df = df[header]


for key in header:
    df[key] = normalise(df[key])

data_train = df.head(len(df)*8//10)
target_train = data_train.pop('score')

data_test = df.tail(len(df)*2//10)
target_test = data_test.pop('score')


print(header)
plt.scatter(df['spelling_mistakes_count'], df['score'])
plt.show()







import numpy as np
from sklearn import datasets, linear_model
regr = linear_model.LinearRegression()
regr.fit(data_train, target_train)
prediction = regr.predict(data_test)
print(sum([i for i in abs((prediction - target_test)/target_test) if i!=__import__('math').inf])/len(target_test))






'''
#import the necessary module
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#create an object of the type GaussianNB
gnb = GaussianNB()

#train the algorithm on training data and predict using the testing data
pred = gnb.fit(data_train, target_train).predict(data_test)
#print(pred.tolist())

#print the accuracy score of the model
print("Naive-Bayes accuracy : ",accuracy_score(target_test, pred, normalize = True))






#import the necessary modules
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

#create an object of type LinearSVC
svc_model = LinearSVC(random_state=0)

#train the algorithm on training data and predict using the testing data
pred = svc_model.fit(data_train, target_train).predict(data_test)

#print the accuracy score of the model
print("LinearSVC accuracy : ",accuracy_score(target_test, pred, normalize = True))









#import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#create object of the lassifier
neigh = KNeighborsClassifier(n_neighbors=5)

#Train the algorithm
neigh.fit(data_train, target_train)

# predict the response
pred = neigh.predict(data_test)

# evaluate accuracy
print ("KNeighbors accuracy score : ",accuracy_score(target_test, pred))

'''

