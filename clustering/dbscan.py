
import math
import numpy as np
import timeit
import random
from scipy import spatial


#partitions the dataSpace into two sets.  One set (1/7th of the data) is used for clustering.
def randomSample(documentMatrix):
    
    restRecords=list()
    randomSample=list()
    
    index=0
    for doc in documentMatrix:
        rand=random.randint(0,7)
        if rand==7:
            randomSample.append(doc)
        else:
            restRecords.append(doc)

    return [randomSample,restRecords]



#creates an empty matrix that will later hold the distance matrix
def createMatrix(docs):

    matrix=list()
    counter=0
    for doc in docs:
        next=list()
        vitalInfo=list()
        vitalInfo.append(0)
        vitalInfo.append(list())
        vitalInfo.append('nil')
        vitalInfo.append(doc[0])
        vitalInfo.append(counter)
        Dists=list()
        counter=counter+1
        next.append(vitalInfo)
        next.append(Dists)
        matrix.append(next)

    return matrix


#computes n choose 2 distances, determining epsilon neighbors of every point, and whether or not every point is a core point
#a list of corePoints is returned.
def augmentedDistanceMatrixNP(docMatrix,emptyMatrix,eps,minPts,method):
    
    #matrix=list()
    corePoints=list()
    
    counter=0
    index=0
    for docA in docMatrix:
        counter=counter+1
        #Dists=list()
        
        #vitalInfo=list()
        #vitalInfo.append(0)
        #vitalInfo.append(list())
        #vitalInfo.append('nil')
        #vitalInfo.append(docA[0])
        #vitalInfo.append(counter-1)  #index of docA in the distMatrix
        #Dists.append(vitalInfo)
        
        #matrix[docA[0]]=mapDists  #each doc id is a key.  its value is the map to hold distances.
        holder=counter
        for docB in docMatrix[counter:len(docMatrix)]:
            #distM=minkowski(docA[3],docB[3],1)
            
            if method==0:
                dist=np.linalg.norm(docA[3]-docB[3])
            else:
                dist=spatial.distance.cosine(docA[3],docB[3])
            
            #Dists.append(distE)
            emptyMatrix[index][1].append(dist)
            if dist<=eps:
                emptyMatrix[index][0][1].append(holder)
                emptyMatrix[index][0][0]=emptyMatrix[index][0][0]+1
                emptyMatrix[holder][0][1].append(index)
                emptyMatrix[holder][0][0]=emptyMatrix[holder][0][0]+1
            holder=holder+1
    #matrix.append(Dists)
    
        if emptyMatrix[index][0][0]>=minPts:
            corePoints.append(index)
        index=index+1
    
    
    return corePoints


#from the distance matrix computed, take only the neighbors of each record, its index in the document matrix, and whether it is a core point or not (some other info also)
def stripDistMatrix(distMatrix):
    
    newMatrix=list()
    
    for doc in distMatrix:
        newMatrix.append(doc[0])
    
    
    return(newMatrix)



#find all density connected points from id and given them label
def densityConnected(distanceMatrix,index,label,minPts):
    
    neighList=distanceMatrix[index][1]
    
    #neighbor is an index in the distance matrix
    for neighbor in neighList:
        if distanceMatrix[neighbor][2]=='nil':
            distanceMatrix[neighbor][2]=label
            #the below line checks if the neighbor is a core point.  If it is, then recursively call densityConnected
            if distanceMatrix[neighbor][0]>=minPts:
                densityConnected(distanceMatrix,neighbor,label,minPts)

#corePoints is the list of corePoints
#returns a map of cluster labels as keys, and value is list of indices in the sample data set of records in that cluster
def obtainLabels(distanceMatrix,corePoints,minPts):
    label=0
    
    #point is an index in the distanceMatrix
    for point in corePoints:
        if distanceMatrix[point][2]=='nil':
            label=label+1
            distanceMatrix[point][2]=label
            densityConnected(distanceMatrix,point,label,minPts)

    clusterMap={}
    for doc in distanceMatrix:
        label=doc[2]
        if clusterMap.has_key(label):
            clusterMap[label].append(doc[4])   #takes in index in original structure
        else:
            clusterMap[label]=list()
            clusterMap[label].append(doc[4])   #takes in index in original structure


    return clusterMap




#returns a central representative from every cluster defined in clusterMap (key is cluster, value is representative)
def determineReps(documentMatrix,clusterMap):
    
    
    repMap={}

    for clustering, points in clusterMap.items():
        if clustering!='nil':
            rep=[0]*256
            for index in points:
                docFrequencies=documentMatrix[index][3]
                wordI=0
                for count in docFrequencies:
                    rep[wordI]=rep[wordI]+count
                    wordI=wordI+1
            wordI=0
            for count in rep:
                rep[wordI]=round(count/float(len(points)))
                wordI=wordI+1
            repMap[clustering]=rep

    return repMap



#returns a map where key is the clustering, and value is the list of indices in restOfData of records in that cluster
#method=0 indicates cosine.  Method=1 indicates euclidian
#this function determines which cluster the record in rest of data is closest to, and it puts that record in that cluster.
def assignToNearestCluster(repMap,restOfData,method):
    
    

    clusterMapOther={}

    index=0
    for record in restOfData:
        
        #label=0
        minDist=float('inf')
        for clustering,rep in repMap.items():
            if method==0:
                dist=np.linalg.norm(record[3]-rep)
            else:
                dist=spatial.distance.cosine(record[3],rep)
            if dist<minDist:
                #print(clustering)
                label=clustering
        if clusterMapOther.has_key(label):
            clusterMapOther[label].append(index)
        else:
            clusterMapOther[label]=list()
            clusterMapOther[label].append(index)
                    
        index=index+1

    return clusterMapOther


#returns a list.  Each value is a map with key as label, and value as frequency. Each map corresponds to one cluster
def grabLabels(clusterMapSample,clusterMapRest,sampleData,restData):
    
    clusterLabelList=list()

    for clustering, indices in clusterMapSample.items():
        if clustering!='nil':
            classLabels={}
            for index in indices:
                labels=sampleData[index][1]
                for label in labels:
                    if classLabels.has_key(label):
                        classLabels[label]=classLabels[label]+1
                    else:
                        classLabels[label]=1
        
        if clusterMapRest.has_key(clustering):
            for ind in clusterMapRest[clustering]:
                labels=restData[ind][1]
                for label in labels:
                    if classLabels.has_key(label):
                        classLabels[label]=classLabels[label]+1
                    else:
                        classLabels[label]=1
        if clustering!='nil':
            clusterLabelList.append(classLabels)

    return clusterLabelList






































