import numpy as np
import matplotlib.pyplot as plt
import random
def ranPartition(X,T,tesFrac):
	(nrow,ncol)=X.shape
	nTrain=int(round(nrow*tesFrac))
	nTest=nrow-nTrain
	rowIndicesRandomlyOrdered=random.sample(xrange(nrow),nrow)
	trainIndices=rowIndicesRandomlyOrdered[:nTrain] #row indices for training sample
	testIndices=rowIndicesRandomlyOrdered[nTrain:]#row indices for testing sample
	#getting training and test data
	Xtrain=X[trainIndices,:]
	Ttrain=T[trainIndices,:]
	Xtest=X[testIndices,:]
	Ttest=T[testIndices,:]
	return (Xtrain,Ttrain,Xtest,Ttest)
def percentCorrect(p,t):
    return np.sum(p.ravel()==t.ravel()) / float(len(t)) * 100

def addOnes(X):
    return np.hstack((np.ones((X.shape[0],1)), X))
def g(X,beta):
    fs = np.exp(np.dot(X, beta))  # N x K-1
    denom = (1 + np.sum(fs,axis=1)).reshape((-1,1))
    gs = fs / denom
    return np.hstack((gs,1/denom))

def makeIndicatorVars(T):
    # Make sure T is two-dimensiona. Should be nSamples x 1.
    if T.ndim == 1:
        T = T.reshape((-1,1))    
    return (T == np.unique(T)).astype(int)


Xnames=['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue' ,'OD280/OD315 of diluted wines' ,'Proline']
Tnames=['class1','class2','class3']
data = np.loadtxt("wine.data" ,delimiter=',')
X = data[:,1:]
T = data[:,0]

class1rows=T==1
class2rows=T==2
class3rows=T==3

X1train,T1train,X1test,T1test=ranPartition(X[class1rows,:],T[class1rows,:],0.8)
X2train,T2train,X2test,T2test=ranPartition(X[class2rows,:],T[class2rows,:],0.8)
X3train,T3train,X3test,T3test=ranPartition(X[class3rows,:],T[class3rows,:],0.8)

T1train=T1train.reshape(-1,1)
T2train=T2train.reshape(-1,1)
T3train=T3train.reshape(-1,1)


T1test=T1test.reshape(-1,1)
T2test=T2test.reshape(-1,1)
T3test=T3test.reshape(-1,1)

Xtrain=np.vstack((X1train,X2train,X3train))
Ttrain=np.vstack((T1train,T2train,T3train))

Xtest=np.vstack((X1test,X2test,X3test))
Ttest=np.vstack((T1test,T2test,T3test))
'''
train = np.loadtxt("zip.train.gz")
test = np.loadtxt("zip.test.gz")
Xtrain=train[:,1:]
Ttrain = train[:,0:1]
Xtest=test[:,1:]
Ttest=test[:,0:1]
'''
def makeStandardizeF(X):
    means = X.mean(axis=0)
    stds = X.std(axis=0,ddof=1)
    def standardize(origX):
        return (origX - means) / stds
    def unStandardize(stdX):
        return stds * stdX + means
    return (standardize, unStandardize)

# Fit generative models (Normal distributions) to each class
standardize,_ = makeStandardizeF(Xtrain)
Xtrains = standardize(Xtrain)

class1rows=Ttrain==1
class2rows=Ttrain==2
class3rows=Ttrain==3

mu1=np.mean(Xtrains[class1rows,:],axis=0)
mu2=np.mean(Xtrains[class2rows,:],axis=0)
mu3=np.mean(Xtrains[class3rows,:],axis=0)


Sigma1=np.cov(Xtrains[class1rows,:].T)
Sigma2=np.cov(Xtrains[class2rows,:].T)
Sigma3=np.cov(Xtrains[class3rows,:].T)

N1=np.sum(class1rows)
N2=np.sum(class2rows)
N3=np.sum(class3rows)

#doubt
N=len(T)
prior1=N1/float(N)
prior2=N2/float(N)
prior3=N3/float(N)


# Form the QDA discriminant functions.

def discQDA(X, standardize, mu, Sigma, prior):
    Xc = standardize(X) - mu
    if Sigma.size == 1:
        Sigma = np.asarray(Sigma).reshape((1,1))
    det = np.linalg.det(Sigma)        
    if det == 0:
        raise np.linalg.LinAlgError('discQDA(): Singular covariance matrix')
    SigmaInv = np.linalg.inv(Sigma)     # pinv in case Sigma is singular
    return -0.5 * np.log(det) \
           - 0.5 * np.sum(np.dot(Xc,SigmaInv) * Xc, axis=1) \
           + np.log(prior)

def discLDA(X,standardize,mu,Sigma,prior):
	Xc=standardize(X) #-mu
	if Sigma.size==1:
		Sigma=np.asarray(Sigma).reshape((1,1))
	det=np.linalg.det(Sigma)
	if det==0:
		raise np.linalg.LinAlgError('discLDA():Sigular covarience matrix')
	SigmaInv=np.linalg.inv(Sigma)
	return np.dot(X,np.dot(SigmaInv,mu))-0.5*np.dot(np.dot(Xc,SigmaInv),mu)+np.log(prior)



d1_train=discQDA(Xtrain,standardize,mu1,Sigma1,prior1)
d2_train=discQDA(Xtrain,standardize,mu2,Sigma2,prior2)
d3_train=discQDA(Xtrain,standardize,mu3,Sigma3,prior3)
predictedTrain=np.argmax(np.vstack((d1_train,d2_train,d3_train)),axis=0).reshape(-1,1)


d1_test=discQDA(Xtest,standardize,mu1,Sigma1,prior1)
d2_test=discQDA(Xtest,standardize,mu2,Sigma2,prior2)
d3_test=discQDA(Xtest,standardize,mu3,Sigma3,prior3)
predictedTest=np.argmax(np.vstack((d1_test,d2_test,d3_test)),axis=0).reshape(-1,1)
Sigma=prior1 * Sigma1 + prior2 * Sigma2 + prior3 * Sigma3
#Sigma=(Sigma1+Sigma2+Sigma3)/3.0


dl1_train=discLDA(Xtrain,standardize,mu1,Sigma,prior1)
dl2_train=discLDA(Xtrain,standardize,mu2,Sigma,prior2)
dl3_train=discLDA(Xtrain,standardize,mu3,Sigma,prior3)
predictedTrainLDA=np.argmax(np.vstack((dl1_train,dl2_train,dl3_train)),axis=0).reshape(-1,1)


dl1_test=discLDA(Xtest,standardize,mu1,Sigma,prior1)
dl2_test=discLDA(Xtest,standardize,mu2,Sigma,prior2)
dl3_test=discLDA(Xtest,standardize,mu3,Sigma,prior3)
predictedTestLDA=np.argmax(np.vstack((dl1_test,dl2_test,dl3_test)),axis=0).reshape(-1,1)

Xtests=standardize(Xtest)

Xtrains1=addOnes(Xtrains)
Xtests1=addOnes(Xtests)
TtrainI=makeIndicatorVars(Ttrain)
TtestI=makeIndicatorVars(Ttest)
beta=np.zeros((Xtrains1.shape[1],TtrainI.shape[1]-1))
alpha=0.0001

for i in range(1000):
	gs=g(Xtrains1,beta)
	beta=beta+alpha*np.dot(Xtrains1.T,TtrainI[:,:-1]-gs[:,:-1])
	likelihoodPerSample=np.exp(np.sum(TtrainI*np.log(gs))/Xtrains.shape[0])
logregOutputTrain=g(Xtrains1,beta)
predictedTrainLLR=np.argmax(logregOutputTrain,axis=1)
logregOutputTest=g(Xtests1,beta)
predictedTestLLR=np.argmax(logregOutputTest,axis=1)




print "QDA Percent correct: Train",percentCorrect(predictedTrain+1,Ttrain),"Test",percentCorrect(predictedTest+1,Ttest)
print "LDA Percent correct: Train",percentCorrect(predictedTrainLDA+1,Ttrain),"Test",percentCorrect(predictedTestLDA+1,Ttest)
print "LLR Percent correct:Train",percentCorrect(predictedTrainLLR+1,Ttrain),"Test",percentCorrect(predictedTestLLR+1,Ttest)



