import cv2
import math
import numpy
import os
import pandas
import sys

from datetime import datetime
from random import shuffle
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from statistics import stdev

if __name__ == '__main__':
    # Variable Declaration
    start = datetime.now()
    columns = list(range(4096))
    numberOfIterations = 1
    currentIteration = 0
    accList = []
    if len(sys.argv) >= 2:
        try:
            numberOfIterations = int(sys.argv[1])
            if numberOfIterations <= 0:
                print("Parameter Error")
                exit(-1)
        except ValueError:
            print("Parameter Error")
            exit(-1)


    columns.append("celebrityName")

    os.chdir("../../data/Olivetti")

    print("----------------------------------------------------------")
    print("Running script " + str(numberOfIterations) + " time(s)")
    print("----------------------------------------------------------")

    names = os.listdir('.')
    while currentIteration < numberOfIterations:
        XTrain = []
        XTest = []

        for i in range(len(names)):
            currentImages = os.listdir(names[i])
            shuffle(currentImages)
            for imageName in currentImages[:math.floor(len(currentImages) * .80)]:
                images = numpy.asarray(cv2.imread(names[i] + '/' + imageName, 0), dtype="float32").flatten()
                images /= 255
                images = images.tolist()
                images.append(names[i])
                XTrain.append(images)

            for imageName in currentImages[math.floor(len(currentImages) * .80):]:
                images = numpy.asarray(cv2.imread(names[i] + '/' + imageName, 0), dtype="float32").flatten()
                images /= 255
                images = images.tolist()
                images.append(names[i])
                XTest.append(images)
            print("Splitting data into training and testing data for iteration #" + str(currentIteration) + ' ' + str(math.floor((i / len(names)) * 100)) + "% complete")

        print("Splitting complete")

        XTrain = pandas.DataFrame.from_records(XTrain, columns=columns)
        XTest = pandas.DataFrame.from_records(XTest, columns=columns)
        yTrain = XTrain["celebrityName"]
        yTest = XTest["celebrityName"]

        del XTrain["celebrityName"]
        del XTest["celebrityName"]

        print("Generating result")
        pca = PCA(n_components=numpy.where(PCA().fit(XTrain).explained_variance_ratio_.cumsum() > 0.95)[0][0]).fit(XTrain)
        accList.append(accuracy_score(yTest, SVC().fit(pca.transform(XTrain), yTrain).predict(pca.transform(XTest))))

        print("Iteration #" + str(currentIteration + 1) + " complete")
        currentIteration += 1

    print("----------------------------------------------------------")
    for i in range(len(accList)):
        print("Accuracy for Iteration #" + str(i + 1) + " = " + str(accList[i]))
    print("----------------------------------------------------------")
    if numberOfIterations != 1:
        print("Average Accuracy = " + str(sum(accList) / len(accList)) + " (±" + str(stdev(accList)) + ")")
    print("----------------------------------------------------------")
    print("Total time elapsed: " + str(datetime.now() - start))
    print("----------------------------------------------------------")

    print("Total time elapsed: " + str(datetime.now() - start))