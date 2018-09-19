import numpy as np
import matplotlib.pyplot as plt

def _getInitValues(num,mini=-100,maxi=100):
    centroids = []
    if mini > maxi:
        raise ValueError('maxi should be greater than mini')
    else:
        for i in range(num):
            meanBtw = (maxi - mini)/ 2 + mini
            distance = maxi - meanBtw
            result = (np.random.rand(2) - 0.5) * distance * 2 + meanBtw
            centroids.append(result)
        return centroids

def getBestCentroidAll(data,centroids):
    pair = []
    for i in range(len(data)):
        cId = _getBestCentroid(one_data=data[i],centroids=centroids)
        result = [data[i], cId]
        pair.append(result)
    return pair

def _getBestCentroid(one_data,centroids):
    tmp = []
    for j in range(len(centroids)):
        dis = _distance(one_data,centroids[j])
        tmp.append(dis)
    result = min(tmp)
    index = tmp.index(result)
    return index

def  _distance(arr,ini):
    return np.linalg.norm(np.abs(ini - arr))

def _clusterise(data_with_cl,numCl):
    clusters = [[]] * numCl
    for i in range(numCl):
        tmp = []
        for data, cl in data_with_cl:
            if cl == i:
                tmp.append(data)
            else:
                pass
        clusters[i] = tmp
    return clusters

def meanCluster(data_with_cl,numCl):
    meanClusters = [[]]*numCl
    dim = 2
    clusters = _clusterise(data_with_cl,numCl)
    for i in range(numCl):
        mean = [[]] * dim
        for l in range(dim):
            arr = np.array([i for i in clusters[i]])
            mean[l] = np.mean(arr[:,l])
        meanClusters[i] = mean
    return meanClusters

def plot(centroids,label=''):
    arr = arr = np.array([i for i in centroids])
    x = arr[:,0]
    y = arr[:,1]
    plt.plot(x,y,'o',label=label)
    print(centroids)
    plt.legend()

#収束判定をするために、評価関数を作る
def J(data_with_cl,centroids):
    clusters = _clusterise(data_with_cl=data_with_cl,numCl=2)
    tmp = []
    for i,cluster in enumerate(clusters):
        arr = np.array([val - centroids[i]  for val in cluster])
        tmp.append(np.sum(arr **2))
    value = np.sum(tmp)
    return value

if __name__ == '__main__':
  #initial centroids
  centroids = _getInitValues(3,mini=-20,maxi=20)
  #sample data
  data1 = np.random.randn(100) * 10
  data1 = data1.reshape((50,2))
  data2 = np.random.binomial(100,0.1,size=(50,2))
  data3 = np.random.beta(a=4,b=9,size=(50,2)) * 10 + 5
  data = np.concatenate((data1,data2,data3))

  diff = 100
  centroids = _getInitValues(3,mini=10,maxi=30)
  pairs = getBestCentroidAll(data,centroids)
  plot(data)
  preJ = J(pairs,centroids)
  plot(centroids,"centroids1")
  count = 1
  while diff > .0001:
      count += 1
      pairs = getBestCentroidAll(data,centroids)
      centroids = meanCluster(pairs,3)
      plot(centroids,"centroids{}".format(count))
      newJ = J(pairs,centroids)
      diff = preJ - newJ
      preJ = newJ
  print(preJ)
  print(newJ)
