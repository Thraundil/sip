from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial import procrustes
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------
#                          General Purpose
#------------------------------------------------------------------------

XTrain  = np.genfromtxt('images/SIPdiatomsTrain.txt',delimiter=',')
XTest   = np.genfromtxt('images/SIPdiatomsTest.txt',delimiter=',')
XTrainL = np.genfromtxt('images/SIPdiatomsTrain_classes.txt',delimiter=',')
XTestL  = np.genfromtxt('images/SIPdiatomsTest_classes.txt',delimiter=',')

def PrintAccuracy(testSetL, predictions):
  correct = 0
  for x in range(len(testSetL)):
    if (testSetL[x] == predictions[x]):
      correct += 1
  print (str(correct) + " out of " + str(len(XTestL)) + " correct")
  print (accuracy_score(testSetL,predictions))


def procrustes(tDiatom,tSet):

  # Sets X and Y lists, because someone gave us ugly data as input
  tDiatomX = tDiatom[0::2]
  tDiatomY = tDiatom[1::2]

  # Translation, so all objects allign at center 0
  tDiatomX -=  np.mean(tDiatomX)
  tDiatomY -=  np.mean(tDiatomY)

  tDiatomM = np.array([tDiatomX,tDiatomY])
  tDiatomMT = np.transpose(tDiatomM)

  # Return list
  tSetM = []

  # For each diatom in tSet
  for x in range(len(tSet)):
    tSetX = tSet[x][0::2]
    tSetY = tSet[x][1::2]

    # Translate object, allign at center 0
    tSetX -= np.mean(tSetX)
    tSetY -= np.mean(tSetY)
    tOut   = np.array([tSetX,tSetY])

    # Rotate
    rotateYX = np.dot(tOut,tDiatomMT)
    U,S,V = np.linalg.svd(rotateYX)
    UT = np.transpose(U)
    RotationMatrix = np.dot(V,UT)

    Rotated = np.dot(RotationMatrix,tOut)

    tSetX = Rotated[0,:]
    tSetY = Rotated[1,:]

    # Scale
    topSum = 0
    botSum = 0
    for x in range(len(tDiatomX)):
      y_n  = np.array([tSetX[x],tSetY[x]])
      y_nT = np.transpose(y_n)
      x_n  = np.array([tDiatomX[x],tDiatomY[x]])
      topSum += np.dot(y_nT,x_n)
      botSum += np.dot(y_nT,y_n)

    scaleSum = topSum/botSum

    tSetX = tSetX * scaleSum
    tSetY = tSetY * scaleSum    

    # Convert back to a data format to be used in 2.2 + 2.3 KNN
    c = np.empty((tSetX.size + tSetY.size,), dtype=tSetX.dtype)
    c[0::2] = tSetX
    c[1::2] = tSetY
    tSetM.append(c)


  finalTSet = np.array(tSetM)

#  Rotation
#  test = np.linalg.svd(testInput)

  return finalTSet

# -----------------------------------------------------------------------
#                             Task 2.2
#------------------------------------------------------------------------
def task_2_2():
  test = procrustes(XTrain[0],XTrain)




#  mtx1, mtx2, disparity = procrustes(goalMatrix, testMatrix)
#  round(disparity)




# -----------------------------------------------------------------------
#                             Task 2.3
#------------------------------------------------------------------------
def task_2_3():

  knn1 = KNeighborsClassifier()
  knn1.fit(XTrain,XTrainL)
  knn1Predictions = knn1.predict(XTest)
  PrintAccuracy(XTestL,knn1Predictions)

  train2 = procrustes(XTrain[0],XTrain)
  test2  = procrustes(XTrain[0],XTest)

  knn2 = KNeighborsClassifier()
  knn2.fit(train2,XTrainL)
  knn2Predictions = knn2.predict(test2)
  PrintAccuracy(XTestL,knn2Predictions)



# -----------------------------------------------------------------------
#                             MAIN/TESTS
#------------------------------------------------------------------------

#task_2_2()
task_2_3()