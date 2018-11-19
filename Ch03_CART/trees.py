'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''

from math import log
import operator
from numpy import random

def createDataSet():
    '''
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels
    '''
    data = [
        [2, 2, 2, 3, 3, 2, 'yes'],
        [3, 2, 3, 3, 3, 2, 'yes'],
        [3, 2, 2, 3, 3, 2, 'yes'],
        [2, 1, 2, 3, 2, 1, 'yes'],
        [3, 1, 2, 3, 2, 1, 'yes'],
        [2, 3, 1, 3, 1, 1, 'no'],
        [1, 1, 3, 2, 3, 2, 'no'],
        [3, 1, 2, 3, 2, 1, 'no'],
        [1, 2, 2, 1, 1, 2, 'no'],
        [2, 2, 3, 2, 2, 2, 'no'],
        [2, 2, 3, 3, 3, 2, 'yes'],
        [1, 2, 2, 3, 3, 2, 'yes'],
        [3, 1, 2, 3, 2, 2, 'yes'],
        [3, 1, 3, 2, 2, 2, 'no'],
        [1, 3, 1, 1, 1, 2, 'no'],
        [1, 2, 2, 1, 1, 1, 'no'],
        [2, 1, 2, 2, 3, 2, 'no']
        ]

    feaList = ['color', 'root', 'sound', 'details', 'mid', 'touch']
    
    return data, feaList

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet: #the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): 
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2) #log base 2
    return shannonEnt
    

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:   
            # pick out the feature that locat in cur_axis 
            reducedFeatVec = featVec[:axis]     
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

    
def chooseBestFeatureToSplit(dataSet, feaList):
    numFeatures = len(feaList)      #the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    
    print('numFeatures : %d '% numFeatures)
    
    bestFeature = random.randint(0, numFeatures, 1)[0]
    for i in range(numFeatures):        #iterate over all the features
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        uniqueVals = set(featList)       #get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)     
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
        print('feature idx : %s \t, info_gain : %f' %(feaList[i],infoGain))
    print('\nbest feature : %s\n' % feaList[bestFeature])
    return bestFeature                      #returns an integer

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        classCount[vote] =  classCount.get(vote, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, feaList):
    classList = [example[-1] for example in dataSet]
    
    if classList.count(classList[0]) == len(classList): 
        return classList[0]  #stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    
    bestFeat = chooseBestFeatureToSplit(dataSet, feaList)
    bestFeatLabel = feaList[bestFeat]
    
    myTree = {bestFeatLabel:{}}  # initial a tree
    
    del(feaList[bestFeat])          # pick out the feature that used to split cur_node
    
    featValues = [example[bestFeat] for example in dataSet]
    
    uniqueVals = set(featValues)
    
    for value in uniqueVals:  # build the tree
        subfeaList = feaList[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subfeaList)
    return myTree                            
    
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: 
        classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'wb+')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
    
