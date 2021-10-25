import math
import os
import pickle
import random
import sys

from annoy import AnnoyIndex
from datetime import datetime
from deepface.commons import functions
from deepface.basemodels import FbDeepFace
from statistics import stdev


if __name__ == "__main__":

    # Variable Declaration
    start = datetime.now()
    forcePreProcess = False
    numberOfIterations = 1
    currentIteration = 0
    accuracyList = []

    if len(sys.argv) >= 2:
        try:
            numberOfIterations = int(sys.argv[1])
            if numberOfIterations <= 0:
                print("Parameter Error")
                exit(-1)
        except ValueError:
            print("Parameter Error")
            exit(-1)
        if len(sys.argv) >= 3:
            if sys.argv[2] == "True" or sys.argv[2] == "true":
                forcePreProcess = True

    print("----------------------------------------------------------")
    print("Running script " + str(numberOfIterations) + " time(s)")
    print("----------------------------------------------------------")

    # Handle preprocessing
    if not os.path.exists("deepFaceData.pkl") or forcePreProcess:
        print("Starting preprocessing")
        file = open(r"deepFaceData.pkl", "wb")

        os.chdir("../../data/Olivetti")

        model = FbDeepFace.loadModel()
        print("DeepFace model successfully loaded")
        data = []
        names = os.listdir(".")

        for i in range(len(names)):
            for j in os.listdir(names[i]):
                path = names[i] + "/" + j

                data.append([path, model.predict(functions.preprocess_face(path, (152, 152), False, False))])
            print("(Preprocessing) " + str(round((i / len(names)) * 100)) + "% complete")

        pickle.dump(data, file)
        file.close()

        os.chdir("../../Deep Learning approaches/deepFace")

    # Get Data
    file = open(r"deepFaceData.pkl", "rb")
    data = pickle.load(file)

    while currentIteration < numberOfIterations:

        trainingData = []
        testingData = []
        current = []
        found = 0
        blunder = 0
        currentName = "Aaron_Peirsol"
        # Split data into training and testing
        for i in range(len(data)):
            if currentName != data[i][0].split('/')[0]:
                random.shuffle(current)
                trainingData.extend(current[:math.floor(len(current) * .80)])
                testingData.extend(current[math.floor(len(current) * .80):])
                current = []
                currentName = data[i][0].split('/')[0]
            current.append(data[i])

        index = AnnoyIndex(4096, "euclidean")

        for i in range(len(testingData)):
            index.add_item(i, testingData[i][1][0])

        index.build(3)

        for i in range(len(testingData)):
            index.add_item(i + len(trainingData), testingData[i][1][0])
            ans = index.get_nns_by_item(i + len(trainingData), 10)
            correct = data[ans[0]][0].split('/', 1)[0]
            located = False
            for i in range(1, len(ans)):
                if correct == data[ans[i]][0].split('/', 1)[0]:
                    found += 1
                    located = True
                    break
            if not located:
                blunder += 1
        accuracyList.append(found / (found + blunder))
        currentIteration += 1
        print("Iteration #" + str(currentIteration + 1) + " complete")

    print("----------------------------------------------------------")
    for i in range(len(accuracyList)):
        print("Accuracy for Iteration #" + str(i + 1) + " = " + str(accuracyList[i]))
    print("----------------------------------------------------------")
    if numberOfIterations != 1:
        print("Average Accuracy = " + str(sum(accuracyList) / len(accuracyList)) + " (±" + str(stdev(accuracyList)) + ")")
    print("----------------------------------------------------------")
    print("Total time elapsed: " + str(datetime.now() - start))
    print("----------------------------------------------------------")