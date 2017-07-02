#reading data
import numpy as np
import matplotlib.pyplot as plt
import pandas
d = pandas.read_csv('slump_data.text')

#constructing
T = d.iloc[:,8:].values
X = d.iloc[:,1:8].values
names =  list(d.columns.values)
Xnames = names[1:8]
Tname = names[8:]
# include bias in input matrix
#X1 = np.hstack((np.ones((X.shape[0],1)), X))
#Xnames.insert(0, 'bias')

#visualization
for bb in range(T.shape[1]):
    plt.figure(bb,figsize=(10,10))
    for c in range(X.shape[1]):
        plt.subplot(3,3, c+1)
        plt.plot(X[:,c], T[:,0], 'o', alpha=0.5)
        plt.ylabel(Tname[bb])
        plt.xlabel(Xnames[c])

#random data partitioning
nrows = X.shape[0]
nTrain = int(round(nrows*0.8))
nTest = nrows - nTrain
rows = np.arange(nrows)
np.random.shuffle(rows)
trainIndices = rows[:nTrain]
testIndices = rows[nTrain:]
Xtrain = X[trainIndices,:]
Ttrain = T[trainIndices,:]
Xtest = X[testIndices,:]
Ttest = T[testIndices,:]

# standardization of data
def makeStandardize(X):
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    def standardize(origX):
        return (origX - means) / stds
    def unStandardize(stdX):
        return stds * stdX + means
    return (standardize, unStandardize)

# makeLLS function
def makeLLS(X,T,lambdaw):
    (standardizeF, unstandardizeF) = makeStandardize(X)
    X = standardizeF(X)
    (nRows,nCols) = X.shape
    X = np.hstack((np.ones((X.shape[0],1)), X))    
    penalty = lambdaw * np.eye(nCols+1)
    penalty[0,0]  =0  # don't penalize the bias weight
    w = np.linalg.lstsq(np.dot(X.T,X)+penalty, np.dot(X.T,T))[0]
    return (w, standardizeF, unstandardizeF)

# useLLS function
def useLLS(model,X):
    w, standardizeF, _ = model
    X = standardizeF(X)
    X = np.hstack((np.ones((X.shape[0],1)), X))
    return np.dot(X,w)

#defining lambdaw value
l=10

# calling the function 
model=makeLLS(Xtrain,Ttrain,l)
predictions=useLLS(model,Xtest)

print(predictions)


