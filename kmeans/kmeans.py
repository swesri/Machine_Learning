import pandas
import numpy as np
import matplotlib.pyplot as plt

data = pandas.read_csv(open('tae.data'))
data = data[data.isnull().any(axis=1)==False]

d=data.iloc[90:130,0:2].values
d[:,0]=data.iloc[90:130,5].values
d[:,1]=data.iloc[90:130,4].values
#print(d)

def calcJ(d,centers):
    diffsq = (centers[:,np.newaxis,:] - d)**2
    return np.sum(np.min(np.sum(diffsq,axis=2), axis=0))

def kmeans(d, k , n):
    # Initialize centers and list J to track performance metric
    centers = d[np.random.choice(range(d.shape[0]),k,replace=False), :]
    J = []
    
    # Repeat n times
    for iteration in range(n):
        
        # Which center is each sample closest to?
        sqdistances = np.sum((centers[:,np.newaxis,:] - d)**2, axis=2)
        closest = np.argmin(sqdistances, axis=0)
        
        # Calculate J and append to list J
        J.append(calcJ(d,centers))
        
        # Update cluster centers
        for i in range(k):
            centers[i,:] = d[closest==i,:].mean(axis=0)
            
    # Calculate J one final time and return results
   
    J.append(calcJ(d,centers))

    return centers,J,closest


c,J,closest = kmeans(d,2,6)
print("centers");print(c);
print("J=");print(J);

plt.figure(1)
plt.subplot(511)
plt.title('plot-of-2D-data')
plt.set_yticks(scipy.arange(0,5,0.5))
plt.subplot(51(n+2))
plt.title('clustering of 2D data')
plt.set_yticks(scipy.arange(0,5,0.5))
plt.scatter(d[:,0], d[:,1], s=50,c=closest, alpha=0.5);
plt.scatter(c[:,0],c[:,1],s=80,c="green",alpha=0.5);
plt.subplots_adjust(hspace=0.8,wspace=0.5)
plt.savefig('ssdemo6.png');
plt.savefig('plot-of-2D-data-k-means.png');
plt.figure(2)
plt.scatter(d[:,0],d[:,1],s=80);
print("closest=");print(closest);

plt.subplot(213)
plt.set_yticks(scipy.arange(4,10,0.5))
plt.title('J versus iterations')
plt.plot(J)

plt.subplots_adjust(hspace=0.8,wspace=0.5)
plt.savefig('ssdemo1.png');
