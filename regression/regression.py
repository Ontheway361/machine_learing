'''
Created on 2018/12/05
@author: lujie
'''
import os
import numpy as np
from time import sleep
import json
# import urllib2
import math
import matplotlib.pyplot as plt

from IPython import embed

class Regression(object):
    def __init__(self):
        pass

    def get_root(self):
        return os.getcwd()

    def loadDataSet(self,fileName):      #general function to parse tab -delimited floats
        numFeat = len(open(fileName).readline().split('\t')) - 1 #get number of fields
        dataMat = []; labelMat = []
        fr = open(fileName)
        for line in fr.readlines():
            lineArr =[]
            curLine = line.strip().split('\t')
            for i in range(numFeat):
                lineArr.append(float(curLine[i]))
            dataMat.append(lineArr)
            labelMat.append(float(curLine[-1]))
        return dataMat,labelMat

    def show_dataset(self, x, y, w = None):
        xarr, yarr = np.array(x), np.array(y)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(xarr[:,1].tolist(), yarr, 'b.')
        plt.xlabel('x-axis'), plt.ylabel('y-axis'), plt.title('dataset')
        save_name = 'data.png'
        plt.savefig(get_root() + '/' +save_name)

    def standRegres(self, xArr, yArr):
        xMat = np.mat(xArr); yMat = np.mat(yArr).T
        xTx = xMat.T*xMat
        if np.linalg.det(xTx) == 0.0:
            print("This matrix is singular, cannot do inverse")
            return
        ws = xTx.I * (xMat.T*yMat)
        embed()
        return ws

    def lwlr(self,testPoint, xArr, yArr, k = 1.0):
        xMat = np.mat(xArr); yMat = np.mat(yArr).T
        m = shape(xMat)[0]
        weights = np.mat(eye((m)))
        for j in range(m):                      #next 2 lines create weights matrix
            diffMat = testPoint - xMat[j,:]     #
            weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
        xTx = xMat.T * (weights * xMat)
        if np.linalg.det(xTx) == 0.0:
            print("This matrix is singular, cannot do inverse")
            return
        ws = xTx.I * (xMat.T * (weights * yMat))
        return testPoint * ws

    def lwlrTest(self,testArr,xArr,yArr,k=1.0):  #loops over all the data points and applies lwlr to each one
        m = shape(testArr)[0]
        yHat = np.zeros(m)
        for i in range(m):
            yHat[i] = self.lwlr(testArr[i],xArr,yArr,k)
        return yHat

    def lwlrTestPlot(self,xArr,yArr,k=1.0):  #same thing as lwlrTest except it sorts X first
        yHat = np.zeros(shape(yArr))       #easier for plotting
        xCopy = np.mat(xArr)
        xCopy.sort(0)
        for i in range(shape(xArr)[0]):
            yHat[i] = lwlr(xCopy[i],xArr,yArr,k)
        return yHat,xCopy

    def rssError(self,yArr,yHatArr): #yArr and yHatArr both need to be arrays
        return ((yArr-yHatArr)**2).sum()

    def ridgeRegres(self,xMat,yMat,lam=0.2):
        xTx = xMat.T*xMat
        denom = xTx + np.eye(shape(xMat)[1])*lam
        if np.linalg.det(denom) == 0.0:
            print("This matrix is singular, cannot do inverse")
            return
        ws = denom.I * (xMat.T*yMat)
        return ws

    def ridgeTest(self,xArr,yArr):
        xMat = np.mat(xArr); yMat = np.mat(yArr).T
        yMean = mean(yMat,0)
        yMat = yMat - yMean     #to eliminate X0 take mean off of Y
        #regularize X's
        xMeans = mean(xMat,0)   #calc mean then subtract it off
        xVar = var(xMat,0)      #calc variance of Xi then divide by it
        xMat = (xMat - xMeans)/xVar
        numTestPts = 30
        wMat = zeros((numTestPts,shape(xMat)[1]))
        for i in range(numTestPts):
            ws = ridgeRegres(xMat,yMat,exp(i-10))
            wMat[i,:]=ws.T
        return wMat

    def regularize(self,xMat):#regularize by columns
        inMat = xMat.copy()
        inMeans = mean(inMat,0)   #calc mean then subtract it off
        inVar = var(inMat,0)      #calc variance of Xi then divide by it
        inMat = (inMat - inMeans)/inVar
        return inMat

    def stageWise(self,xArr,yArr,eps=0.01,numIt=100):
        xMat = np.mat(xArr); yMat = np.mat(yArr).T
        yMean = np.mean(yMat,0)
        yMat = yMat - yMean     #can also regularize ys but will get smaller coef
        xMat = regularize(xMat)
        m,n = shape(xMat)
        #returnMat = zeros((numIt,n)) #testing code remove
        ws = np.zeros((n,1)); wsTest = ws.copy(); wsMax = ws.copy()
        for i in range(numIt):
            print(ws.T)
            lowestError = inf;
            for j in range(n):
                for sign in [-1,1]:
                    wsTest = ws.copy()
                    wsTest[j] += eps*sign
                    yTest = xMat*wsTest
                    rssE = rssError(yMat.A,yTest.A)
                    if rssE < lowestError:
                        lowestError = rssE
                        wsMax = wsTest
            ws = wsMax.copy()
            #returnMat[i,:]=ws.T
        return ws

    def searchForSet(self,retX, retY, setNum, yr, numPce, origPrc):
        sleep(10)
        myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
        searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum)
        pg = urllib2.urlopen(searchURL)
        retDict = json.loads(pg.read())
        for i in range(len(retDict['items'])):
            try:
                currItem = retDict['items'][i]
                if currItem['product']['condition'] == 'new':
                    newFlag = 1
                else: newFlag = 0
                listOfInv = currItem['product']['inventories']
                for item in listOfInv:
                    sellingPrice = item['price']
                    if  sellingPrice > origPrc * 0.5:
                        print("%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice))
                        retX.append([yr, numPce, newFlag, origPrc])
                        retY.append(sellingPrice)
            except: print('problem with item %d' % i)

    def setDataCollect(self,retX, retY):
        searchForSet(retX, retY, 8288, 2006, 800, 49.99)
        searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
        searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
        searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
        searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
        searchForSet(retX, retY, 10196, 2009, 3263, 249.99)

    def crossValidation(self,xArr,yArr,numVal=10):
        m = len(yArr)
        indexList = range(m)
        errorMat = np.zeros((numVal,30))#create error mat 30columns numVal rows
        for i in range(numVal):
            trainX=[]; trainY=[]
            testX = []; testY = []
            np.random.shuffle(indexList)
            for j in range(m):#create training set based on first 90% of values in indexList
                if j < m*0.9:
                    trainX.append(xArr[indexList[j]])
                    trainY.append(yArr[indexList[j]])
                else:
                    testX.append(xArr[indexList[j]])
                    testY.append(yArr[indexList[j]])
            wMat = ridgeTest(trainX,trainY)    #get 30 weight vectors from ridge
            for k in range(30):#loop over all of the ridge estimates
                matTestX = np.mat(testX); matTrainX = np.mat(trainX)
                meanTrain = np.mean(matTrainX,0)
                varTrain = np.var(matTrainX,0)
                matTestX = (matTestX-meanTrain)/varTrain #regularize test with training params
                yEst = matTestX * np.mat(wMat[k,:]).T + np.mean(trainY)#test ridge results and store
                errorMat[i,k] = rssError(yEst.T.A, np.array(testY))
                #print errorMat[i,k]
        meanErrors = np.mean(errorMat,0)#calc avg performance of the different ridge weight vectors
        minMean = float(min(meanErrors))
        bestWeights = wMat[nonzero(meanErrors==minMean)]
        #can unregularize to get model
        #when we regularized we wrote Xreg = (x-meanX)/var(x)
        #we can now write in terms of x not Xreg:  x*w/var(x) - meanX/var(x) +meanY
        xMat = np.mat(xArr); yMat = np.mat(yArr).T
        meanX = np.mean(xMat,0); varX = np.var(xMat,0)
        unReg = bestWeights/varX
        print("the best model from Ridge Regression is:\n",unReg)
        print ("with constant term: ",-1*sum(np.multiply(meanX,unReg)) + np.mean(yMat))



if __name__ == '__main__':
    regress_engine = Regression()
    x, y = regress_engine.loadDataSet('ex0.txt')
    print(type(x))

    w = regress_engine.standRegres(x,y)
    py = mat(x) * w
    print('w : ', w)
    # show the fit-curve
    # fig = plt.figure(); ax = fig.add_subplot(111)
    # ax.scatter(array(x)[:,1],y, s = 2, color = 'blue')
    # x_copy = x.copy(); x_copy.sort()
    # pred_y = mat(x_copy) * mat(w)
    # ax.plot(array(x_copy)[:,1], pred_y, linewidth = 1, color = 'k')
    # # plt.show(); plt.close()
    # plt.savefig(os.getcwd() + '/result.png'); plt.close()

    # cal the corrcoef
    print('corrcoef : ', corrcoef(py.T, y))
    # test the corrcoef
    data = []
    predy = py.flatten().A[0].tolist()
    data.append(y-mean(y)); data.append(predy - mean(predy))
    data = mat(array(data))
    Cov =  data * data.T
    print(Cov)

    print(data.shape)
    # cal the lwlr
    pred_y0 = regress_engine.lwlr(x[0], x, y, 1.0)
    pred_y1 = regress_engine.lwlr(x[0], x, y, 0.001)
    print('y : %.3f, pred_y0 : %.3f , pred_y1 : %.3f' % (y[0], pred_y0, pred_y1))

    test_x = []
    for i in range(50):
        test_x.append([1,random.randn()])
    test_y = 3 + 1.7*array(test_x)[:,1] + array([0.1*math.sin(30*i[1]) for i in test_x]) + 0.06*random.randn(50)
    regress_engine.show_dataset(test_x, test_y)
    yHat = regress_engine.lwlrTest(x, x, y, 0.1)
    print(regress_engine.rssError(yHat, y))
