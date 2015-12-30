import pickle
from clustering.dbscan import *
import numpy as np
import timeit
from util.quality import entropy


minPts=10
eps=3.0


feature_vector_file=open("data/document_matrix_256_cont_np.pickle")
print("Beginning euclidian density based clustering on the first 5000 documents from sgml files")
docs=pickle.load(feature_vector_file)
docs=docs[0:5000]

startO=timeit.default_timer()
partition=randomSample(docs)
restRecords=partition[1]
docsSample=partition[0]



distMatrix=createMatrix(docsSample)  #creates the emtpy distance matrix.


print("Clustering with minpts= "+ str(minPts) + " and eps= "+ str(eps))
corePoints=augmentedDistanceMatrixNP(docsSample,distMatrix,eps,minPts,0)


distMatrix=stripDistMatrix(distMatrix)
clusterMapSample=obtainLabels(distMatrix,corePoints,minPts)


repMap=determineReps(docsSample,clusterMapSample)
if len(repMap)>0:

    clusterMapRest=assignToNearestCluster(repMap,restRecords,0) #returns in the indices

    clusterLabelList=grabLabels(clusterMapSample,clusterMapRest,docsSample,restRecords)
    endO=timeit.default_timer()
    print("Time of euclidian density based clustering is: " + str(endO-startO)+" seconds")
    print("Number of clusters produced: "+str(len(clusterLabelList)))
    print("Number of noise points is: "+ str(len(clusterMapSample['nil'])))
    print("Entropy of the euclidian clustering is "+str(entropy(clusterLabelList)))
else:
    print("all points are noise points.  No further calculations.  The clustering is useless")



eps=.4
minPts=10



#all cosine stuff below
feature_vector_file=open("data/document_matrix_256_binary_np.pickle")
print("Beginning cosine density based clustering on the first 5000 documents from sgml files")
docs=pickle.load(feature_vector_file)
docs=docs[0:5000]

startO=timeit.default_timer()
partition=randomSample(docs)
restRecords=partition[1]
docsSample=partition[0]



distMatrix=createMatrix(docsSample)  #creates the emtpy distance matrix.


print("Clustering with minpts= "+ str(minPts) + " and eps= "+ str(eps))
corePoints=augmentedDistanceMatrixNP(docsSample,distMatrix,eps,minPts,1)  #gets corepoints and updates Dist matrix to have all
distMatrix=stripDistMatrix(distMatrix)  #gets rid of unncessary informaiton.
clusterMapSample=obtainLabels(distMatrix,corePoints,minPts)   #clusterMap has indexes of records in docMatrix stored!!!

repMap=determineReps(docsSample,clusterMapSample)
if len(repMap)>0:
    clusterMapRest=assignToNearestCluster(repMap,restRecords,1) #returns in the indices
    clusterLabelList=grabLabels(clusterMapSample,clusterMapRest,docsSample,restRecords)
    endO=timeit.default_timer()
    print("Time of cosine density based clustering is: " + str(endO-startO)+" seconds")
    print("Number of clusters produced: "+str(len(clusterLabelList)))
    print("Number of noise points is: "+ str(len(clusterMapSample['nil'])))
    print("Entropy of the euclidian clustering is "+str(entropy(clusterLabelList)))
else:
    print("all points are noise points.  No further calculations.  The clustering is useless")
