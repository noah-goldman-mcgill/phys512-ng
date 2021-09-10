#
# KNN Example: implement KNN using a sample dataset and the scikitlearn
# implimentation
#
# imports
#
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
#
# get the dataset we're using
#
dataset = datasets.load_wine()
#
# set input and classes
#
x, y     = dataset['data'], dataset['target']
(N,D), C = x.shape, np.max(y)+1
print('instances (N) \t {} \n features (D) \t {} \n classes (C) \t {}'.format(N,D,C))
#
# split dataset into training and test
#
inds             = np.random.permutation(N)
split            = int(np.floor(2*N/3))
x_train, y_train = x[inds[:split]], y[inds[:split]]
x_test, y_test   = x[inds[split:]], y[inds[split:]]
#
# visualize data
#
"""
plt.scatter(x_train[:,0], x_train[:,1], c=y_train, marker='o', label='train')
plt.scatter(x_test[:,0], x_test[:,1], c=y_test, marker='s', label='test')
plt.legend()
plt.ylabel('x_1')
plt.xlabel('x_0')
plt.show()
"""
#
# fit the classification
#
K = 3
KNN = KNeighborsClassifier(n_neighbors=3)
KNN.fit(x_train,y_train)
knns, y_pred = KNN.kneighbors(x_test,K,return_distance=False), KNN.predict(x_test)
# y_pred       = np.argmax(y_prob,axis=1)
correct      = np.zeros((len(y_pred)))
for i in range(len(y_pred)):
    if y_pred[i]==y_test[i]:
        correct[i] = 1
accuracy     = np.sum(correct)/y_test.shape[0]
print(accuracy)
#
# visualize result
#
corr   = y_test == y_pred
incorr = np.logical_not(corr)
plt.scatter(x_train[:,0],x_train[:,1],c=y_train,marker='o',alpha=0.2,label='train')
plt.scatter(x_test[corr,0],x_test[corr,1],marker='.',c=y_pred[corr],label='correct')
plt.scatter(x_test[incorr,0],x_test[incorr,1],marker='x',c=y_test[incorr],label='misclassified')
for i in range(x_test.shape[0]):
    for k in range(K):
        hor = x_test[i,0], x_train[knns[i,k],0]
        ver = x_test[i,1], x_train[knns[i,k],1]
        plt.plot(hor, ver, 'k-', alpha=0.1)
plt.ylabel('x_1')
plt.xlabel('x_0')
plt.legend()
plt.show()
#
#
#
