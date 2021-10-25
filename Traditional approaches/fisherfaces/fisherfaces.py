import cv2
import math
import numpy
import os
import pandas
import sys

from datetime import datetime
from random import shuffle
from statistics import stdev


if __name__ == '__main__':
    
    # Variable Declaration
    start = datetime.now()
    numberOfIterations = 1
    accuracyList = []
    runCount = 0
    columns = list(range(4096))
    XTrain = []
    XTest = []
    model = cv2.face.FisherFaceRecognizer_create()

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
    while runCount < numberOfIterations:
        blunders = 0
        corrects = 0

        XTrain = []
        XTest = []

        for i in range(len(names)):
            currentImages = os.listdir(names[i])
            shuffle(currentImages)
            for imageName in currentImages[:math.floor(len(currentImages) * .80)]:
                images = numpy.asarray(cv2.imread(names[i] + '/' + imageName, 0), dtype="float32").flatten()
                images /= 255  # Normalize pixel values
                images = images.tolist()
                images.append(names[i])
                XTrain.append(images)

            for imageName in currentImages[math.floor(len(currentImages) * .80):]:
                images = numpy.asarray(cv2.imread(names[i] + '/' + imageName, 0), dtype="float32").flatten()
                images /= 255  # Normalize pixel values
                images = images.tolist()
                images.append(names[i])
                XTest.append(images)
            print("Splitting data into training and testing data for iteration #" + str(runCount) + ' ' + str(
                math.floor((i / len(names)) * 100)) + "% complete")

        print("Splitting complete")

        XTrain = pandas.DataFrame.from_records(XTrain, columns=columns)
        XTest = pandas.DataFrame.from_records(XTest, columns=columns)
        yTrain = XTrain["celebrityName"]
        yTest = XTest["celebrityName"]

        del XTrain["celebrityName"]
        del XTest["celebrityName"]

        print("Generating result")

        XTest = XTest.to_numpy()
        yTest = yTest.to_numpy()

        model.train(XTrain.to_numpy(), yTrain.to_numpy().astype(numpy.int))
        for i in range(len(XTest)):
            if model.predict(XTest[i])[0] == int(yTest[i]):
                corrects += 1
            else:
                blunders += 1

        accuracyList.append(corrects / (corrects + blunders))
        print("Accuracy for Iteration #" + str(runCount + 1) + " = " + str(accuracyList[-1]))
        runCount += 1

    print("----------------------------------------------------------")
    for i in range(len(accuracyList)):
        print("Accuracy for Iteration #" + str(i + 1) + " = " + str(accuracyList[i]))
    print("----------------------------------------------------------")
    if numberOfIterations != 1:
        print("Average Accuracy = " + str(sum(accuracyList) / len(accuracyList)) + " (±" + str(stdev(accuracyList)) + ")")
    print("----------------------------------------------------------")
    print("Total time elapsed: " + str(datetime.now() - start))
    print("----------------------------------------------------------")

    print("Total time elapsed: " + str(datetime.now() - start))