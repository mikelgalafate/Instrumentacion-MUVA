import os
import flirimageextractor
from matplotlib import cm
import numpy as np
import cv2
import math

path = "CODIGOS_PARA_EVALUAR/"
iters = 4
inc = 1
dic = np.array([[1, 4, 7],
                [2, 5, 8],
                [3, 6, 9]])

flir = flirimageextractor.FlirImageExtractor(palettes=[cm.jet, cm.bwr, cm.gist_ncar])
with open("codigos.csv", "w") as file:
    for im in os.listdir(path):
        flir.process_image(path + im)

        infoTerm = flir.extract_thermal_image()
        infoTermNorm = (infoTerm - np.amin(infoTerm)) / (np.amax(infoTerm) - np.amin(infoTerm))
        infoTermNorm_int = infoTermNorm[180:390, 130:340]

        its = 0
        umbral = 255
        lectura = ""

        while len(lectura) < 4 and umbral > 0:
            mask = np.uint8((infoTermNorm_int * 255 > umbral)) * 255
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=iters)
            mask = cv2.dilate(mask, kernel, iterations=iters)

            objetos, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if len(objetos) > 4 or len(lectura) == 4 or its == 60:
                umbral += inc
                break
            if objetos and len(objetos) > len(lectura):
                its = 0
                for objeto in objetos:
                    centro, radio = cv2.minEnclosingCircle(objeto)
                    centro = (int(centro[0]), int(centro[1]))
                    coord1 = math.ceil(centro[0]/(210/3))-1
                    coord2 = math.ceil(centro[1]/(210/3))-1
                    valor = str(dic[coord2, coord1])
                    if valor in lectura:
                        continue
                    elif cv2.contourArea(objeto) < 35**2:
                        lectura += valor

            # restar 5 al umbral
            umbral -= inc
            its += 1
        print(lectura[::-1])
        file.write(im + "," + lectura[::-1] + "\n")
