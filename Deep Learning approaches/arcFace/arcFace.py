import math
import os
import random
import sys

from arcface import ArcFace
from datetime import datetime
from statistics import stdev

if __name__ == "__main__":

    # Variable Declaration
    start = datetime.now()
    numberOfIterations = 1
    accuracyList = []
    faceRecognition = ArcFace.ArcFace()
    runCount = 0

    if len(sys.argv) >= 2:
        try:
            numberOfIterations = int(sys.argv[1])
            if numberOfIterations <= 0:
                print("Parameter Error")
                exit(-1)
        except ValueError:
            print("Parameter Error")
            exit(-1)

    print("----------------------------------------------------------")
    print("Running script " + str(numberOfIterations) + " time(s)")
    print("----------------------------------------------------------")

    os.chdir("../../data/Olivetti")

    while runCount < numberOfIterations:
        found = 0
        blunder = 0
        trainingData = []
        testingData = []
        trainingLabels = []
        testingLabels = []
        names = os.listdir(".")
        counter = 0

        for id in range(len(names)):
            current = []
            for i in os.listdir(names[id]):
                current.append(names[id] + "/" + i)
            random.shuffle(current)

            currentTrainingData = current[:math.floor(len(current) * .80)]
            currentTestingData = current[math.floor(len(current) * .80):]

            trainingData.extend(currentTrainingData)
            testingData.extend(currentTestingData)

            trainingLabels.extend(([id] * len(currentTrainingData)))
            testingLabels.extend(([id] * len(currentTestingData)))

        trainImages = faceRecognition.calc_emb(trainingData)
        testImages = faceRecognition.calc_emb(testingData)

        for testImage in testImages:
            current = []
            for trainImage in trainImages:
                current.append(faceRecognition.get_distance_embeddings(testImage, trainImage))
            if testingLabels[counter] == trainingLabels[current.index(min(current))]:
                found += 1
            else:
                blunder += 1
            counter += 1
        accuracyList.append(found / (found + blunder))
        runCount += 1
        print("Iteration " + str(runCount) + " / " + str(numberOfIterations) + " complete")
    print("----------------------------------------------------------")
    print("Total time elapsed: " + str(datetime.now() - start))
    print("----------------------------------------------------------")
    for i in range(len(accuracyList)):
        print("Accuracy for Iteration #" + str(i + 1) + " = " + str(accuracyList[i]))
    print("----------------------------------------------------------")
    if numberOfIterations != 1:
        print("Average Accuracy = " + str(sum(accuracyList) / len(accuracyList)) + " (±" + str(stdev(accuracyList)) + ")")
