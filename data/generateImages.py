"""
This python file is used to generate the Olivetti dataset from the face_data.cvs file
If the Olivetti dataset is present, this does not need to be ran
"""

import cv2
import os
import numpy
import pandas
import shutil

if __name__ == "__main__":
    # Variable Declaration
    currentId = 0
    current = 0
    data = pandas.read_csv("face_data.csv")
    myId = data["target"]

    del data["target"]

    print("Generating Olivetti dataset..")

    if os.path.exists("Olivetti"):
        print("Olivetti folder already found, deleting..")
        shutil.rmtree("Olivetti")

    os.mkdir("Olivetti")

    os.chdir("Olivetti")

    for i in range(40):
        os.mkdir(str(i))
        for j in range(10):
            cv2.imwrite(str(i) + "//" + str(j) + ".jpg", cv2.convertScaleAbs(numpy.array(data)[current].reshape(64, 64), alpha=(255.0)))
            current += 1
    print("Olivetti dataset successfully generated")

