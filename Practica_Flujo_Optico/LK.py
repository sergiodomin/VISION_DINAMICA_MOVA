# -*- coding: utf-8 -*-
"""
@author: muyma
"""

import cv2
import numpy as np
from scipy.ndimage.filters import convolve as filter2
from scipy import linalg
import datetime as dt
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def LK(k_LK, k_sobel, name_folder):
    #Sacamos los primeros frames del video
    images = []
    #vidcap = cv2.VideoCapture('vtest.avi')
    #vidcap = cv2.VideoCapture('MOT16-04.mp4')
    vidcap = cv2.VideoCapture('pelota.mp4')
    success, image = vidcap.read()
    count = 0
    while count<90:
        images.append(image)
        success,image = vidcap.read()
        count += 1

    #Pasamos dos de la simagenes a gris y las dividimos entre 255 para tener mayor resolucion
    images = np.array(images)
    # img1 = cv2.cvtColor(images[0], cv2.COLOR_RGB2GRAY)/255
    # img2 = cv2.cvtColor(images[1], cv2.COLOR_RGB2GRAY)/255
    # cv2.imwrite(name_folder + '/img1.png', cv2.cvtColor(images[0], cv2.COLOR_RGB2GRAY))
    # cv2.imwrite(name_folder + '/img2.png', cv2.cvtColor(images[1], cv2.COLOR_RGB2GRAY))

    img1 = cv2.cvtColor(images[88], cv2.COLOR_RGB2GRAY)/255
    img2 = cv2.cvtColor(images[89], cv2.COLOR_RGB2GRAY)/255
    cv2.imwrite(name_folder + '/img1.png', cv2.cvtColor(images[88], cv2.COLOR_RGB2GRAY))
    cv2.imwrite(name_folder + '/img2.png', cv2.cvtColor(images[89], cv2.COLOR_RGB2GRAY))

    rows, cols = img1.shape

    #Calculamos las imagenes Ix e Iy
    sobelImg1x = cv2.Sobel(img1,cv2.CV_64F,1,0,ksize=k_sobel)
    sobelImg1y = cv2.Sobel(img1,cv2.CV_64F,0,1,ksize=k_sobel)

    sobelImg2x = cv2.Sobel(img2,cv2.CV_64F,1,0,ksize=k_sobel)
    sobelImg2y = cv2.Sobel(img2,cv2.CV_64F,0,1,ksize=k_sobel)

    Ix = (sobelImg1x+sobelImg2x)
    Iy = (sobelImg1y+sobelImg2y)


    #Calculamos la imagen It
    blur1 = cv2.GaussianBlur(img1,(5,5),0)
    blur2 = cv2.GaussianBlur(img2,(5,5),0)
    It = blur2-blur1



    #Calculamos el resto de imagenes que vamos a necesitar
    I2x = Ix**2
    I2y = Iy**2
    Ixy = Ix*Iy
    Ixt = Ix*It
    Iyt = Iy*It

    #Definimos los rangos e inicializamos las variables que vamos a usar para realizar el algorimto LK
    kmin = int((k_LK-1)/2)
    kmax = int((k_LK+1)/2)

    A = np.zeros_like(img1)
    B = np.zeros_like(img1)
    U = np.zeros_like(img1)
    V = np.zeros_like(img1)

    #Guardamos cuando empezamos el algoritmo para posteriormente calcular cuanto hemos tardado
    init_time = dt.datetime.now()


    for row in range(rows-kmin):
        for col in range(cols-kmin):
            if (kmin < row) and (kmin < col):
                A = np.array([[np.sum(I2x[row-kmin:row+kmax,col-kmin:col+kmax]), 
                            np.sum(Ixy[row-kmin:row+kmax,col-kmin:col+kmax])], 
                            [np.sum(Ixy[row-kmin:row+kmax,col-kmin:col+kmax]), 
                            np.sum(I2y[row-kmin:row+kmax,col-kmin:col+kmax])]
                            ])
                A = np.linalg.pinv(A)
                B = np.array([[-np.sum(Ixt[row-kmin:row+kmax,col-kmin:col+kmax])],
                            [-np.sum(Iyt[row-kmin:row+kmax,col-kmin:col+kmax])]
                            ])
                C = A*B
                U[row,col] = C[0,0]
                V[row,col] = C[1,0]         

    #Finalizamos el algoritmo y volvemos a coger el tiempo            
    end_time = dt.datetime.now()

    #Suavizamos la imagen de salida para evitar outliers, pixeles con valores muy dispares que nos afectan a la representacion
    U = cv2.GaussianBlur(U,(5,5),0)
    V = np.sqrt(U**2+V**2)
    V = cv2.GaussianBlur(V,(5,5),0)

    #Calculamos el tiempo que hemos tardado restando el inicial al final
    time = (end_time - init_time).total_seconds()*1000


    #Representamos lo que hemos obtenido 
    hsv = np.zeros_like(images[0])
    hsv[...,1] = 255
    hsv[...,0] = V*180/np.pi/2
    hsv[...,2] = cv2.normalize(U,None,0,255,cv2.NORM_MINMAX)

    date_img = str(time)  
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    rgb =cv2.putText(img=np.copy(rgb), text="Tiempo de duracion: "+ date_img + " ms" , org=(25,50),fontFace=0, fontScale=0.5, color=(255, 255, 255))
    rgb =cv2.putText(img=np.copy(rgb), text="Numero de K para LK: "+ str(k_LK) , org=(25,75),fontFace=0, fontScale=0.5, color=(255, 255, 255))
    rgb =cv2.putText(img=np.copy(rgb), text="Kernel Sobel : "+ str(k_sobel) , org=(25,100),fontFace=0, fontScale=0.5, color=(255, 255, 255))

    date_file = str(init_time)
    date_file = date_file.replace(":","_").replace(".","_")    
    name_file = name_folder + '/outLK_' + str(k_LK) + '_' + str(k_sobel) + '.png'

    cv2.imwrite(name_file, rgb)

    return time
if __name__ == "__main__":
    name_folder = 'LK_' + str(str(dt.datetime.now()).replace(":","_").replace(".","_"))
    if not os.path.exists(name_folder):
        os.makedirs(name_folder)
    time_arr = []
    k_sobel_arr = []
    k_LK_arr = []
    for k_LK in np.arange(3,30,2):
        time = LK(k_LK,3, name_folder)
        time_arr = np.append(time_arr, time)
        k_sobel_arr = np.append(k_sobel_arr, 3)
        k_LK_arr = np.append(k_LK_arr, k_LK)

    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(k_sobel_arr, k_LK_arr, time_arr, label='LK Duration')
    ax.legend()
    ax.set_xlabel('K Sobel')
    ax.set_ylabel('K LK')
    ax.set_zlabel('Time (ms)')
    plt.savefig(name_folder + '/LK_duration.png')
    #plt.show()        