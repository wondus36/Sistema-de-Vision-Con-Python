from contextlib import contextmanager

@contextmanager
def keep_plots_open(keep_show_open_on_exit=True, even_when_error=True):
    '''
    To continue excecuting code when plt.show() is called
    and keep the plot on displaying before this contex manager exits
    (even if an error caused the exit).
    '''
    import matplotlib.pyplot
    show_original = matplotlib.pyplot.show
    def show_replacement(*args, **kwargs):
        kwargs['block'] = False
        show_original(*args, **kwargs)
    matplotlib.pyplot.show = show_replacement

    pylab_exists = True
    try:
        import pylab
    except ImportError: 
        pylab_exists = False
    if pylab_exists:
        pylab.show = show_replacement

    try:
        yield
    except Exception as err:
        if keep_show_open_on_exit and even_when_error:
            print ("*********************************************")
            print ("Error early edition while waiting for show():" )
            print ("*********************************************")
            import traceback
            print (traceback.format_exc())
            show_original()
            print ("*********************************************")
            raise
    finally:
        matplotlib.pyplot.show = show_original
        if pylab_exists:
            pylab.show = show_original
    if keep_show_open_on_exit:
        show_original()

# ***********************
# Running example
# ***********************
import cv2
import pylab as pl
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import time
from time import sleep
import imutils
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np

if __name__ == '__main__':
    y, x = np.mgrid[0:500:500j, 0:600:600j]
#    with keep_plots_open():
    file="pista-4A.jpg"
    image = cv2.imread(file)
    #cv2.imshow("Foto Pista", image)
    #cv2.waitKey(0)


    hL=49
    sL=50
    vL=50
    hU=100
    sU=255
    vU=210

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    verde_bajos = np.array([hL, sL, vL], dtype=np.uint8)
    verde_altos = np.array([hU, sU, vU], dtype=np.uint8)

    mascara_verde = cv2.inRange(hsv, verde_bajos, verde_altos)

    kernel = np.ones((6,6),np.uint8)
    mascara_verde = cv2.morphologyEx(mascara_verde, cv2.MORPH_CLOSE, kernel)
    mascara_verde = cv2.morphologyEx(mascara_verde, cv2.MORPH_OPEN, kernel)

    res_g = cv2.bitwise_and(image,image, mask= mascara_verde)


    colorCent=res_g
    gray = cv2.cvtColor(colorCent, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

    # find contours in the thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    #xAux=np.zeros((1,1))
    #yAux=np.zeros((1,1))

    MIN_THRESH=0.00000001
    rec=0;
    pos=0;
    xCen=np.zeros((1,1))
    yCen=np.zeros((1,1))
    # loop over the contours
    for c in cnts:
        if cv2.contourArea(c) > MIN_THRESH:
            # process the contour
            # compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            xCen[0][0]=cX
            yCen[0][0]=cY
            #print(xCen,yCen)
            # draw the contour and center of the shape on the image
            cv2.drawContours(colorCent, [c], -1, (0, 230, 25), 2)
            cv2.circle(colorCent, (cX, cY), 7, (255, 255, 255), -1)
            cv2.putText(colorCent, "Centro", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            xVec=cnts[rec]
    
            n=np.shape(xVec)
            xCor=np.zeros((1,n[0]))
            yCor=np.zeros((1,n[0]))
    
            for i in range (0,n[0]):
                xCor[0][i]=xVec[i][0][0]
                yCor[0][i]=xVec[i][0][1]
        
            if rec==0:
                xAux=xCor
                yAux=yCor
                xACen=xCen
                yACen=yCen
                #print(rec)
                #print(xACen,yACen)
            else:
                xAux=np.append(xAux.T,xCor.T)
                yAux=np.append(yAux.T,yCor.T)
                #print(xACen,yACen)
                xACen=np.append(xACen,xCen)
                yACen=np.append(yACen,yCen)
         

    
            rec=rec+1;
    print(xACen)
    print(yACen)
               
    #cv2.imshow('Segmentacion', colorCent)
    #cv2.waitKey(0)

    nombre='Etiqueta Obstaculos'
    resultado=res_g
    shiftedR = cv2.pyrMeanShiftFiltering(resultado, 21, 51)
    grayR = cv2.cvtColor(shiftedR, cv2.COLOR_BGR2GRAY)
    threshR = cv2.threshold(grayR, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    DR = ndimage.distance_transform_edt(threshR)
    localMaxR = peak_local_max(DR, indices=False, min_distance=20, labels=threshR)
    markersR = ndimage.label(localMaxR, structure=np.ones((3, 3)))[0]
    labelsR = watershed(-DR, markersR, mask=threshR)
    if nombre!="Etiquetado Robot":
        print("[INFO] {} unique segments found".format(len(np.unique(labelsR)) - 1))

    for label in np.unique(labelsR):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue

        # otherwise, allocate memory for the label region and draw
        # it on the mask
        maskR = np.zeros(grayR.shape, dtype="uint8")
        maskR[labelsR == label] = 255

        # detect contours in the mask and grab the largest one
        cntsr = cv2.findContours(maskR.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cr = max(cntsr, key=cv2.contourArea)

        #draw a rectangle enclosing the object
        x,y,w,h = cv2.boundingRect(cr)
        rectan=cv2.rectangle(resultado,(x,y),(x+w,y+h),(0,35,255),2)
        #print(rectan)
        rect = cv2.minAreaRect(cr)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        #cv2.drawContours(resultado,[box],0,(0,0,255),2)
        cv2.putText(resultado, "#{}".format(label), (int(x) - 10, int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        #print(box)
        #cv2.imshow(nombre, resultado)
        #cv2.waitKey(0)

    #cv2.destroyAllWindows()


    #Graficar campos potenciales
    print("Voy a pasar a los campos")
    y, x = np.mgrid[0:500:500j, 0:600:600j]


    #Obstaculos
    alpha_obstacle, a_obstacle, b_obstacle = 1.0, 1.9e3, 1.9e3
    pobs=0
    lim=len(xACen)
    yACen2=np.arange(lim)
    for contador in range(lim):
        yACen2[contador]=500-yACen[contador]

        x_obstacle = xACen[contador]
        y_obstacle = yACen2[contador]
        p1= -alpha_obstacle * np.exp(-((x - x_obstacle)**2 / a_obstacle + (y - y_obstacle)**2 / b_obstacle))
        pobs=p1+pobs
    #meta
    x_meta = 50
    y_meta = 140
    alpha_meta, a_meta, b_meta= 1.0, 3e4, 3e4
    pmeta = alpha_meta * np.exp(-((x - x_meta)**2 / a_meta + (y - y_meta)**2 / b_meta))    
    pmeta=pmeta

    #------------------------------Meteer ciclo
    #carro
    print(w)
    x_carro = 550
    y_carro = 140
    alpha_carro, a_carro, b_carro= 1.0, 3.9e3, 3.9e3
    pcarro = -alpha_carro * np.exp(-((x - x_carro)**2 / a_carro + (y - y_carro)**2 / b_carro))    
    #suma de todos los campos
    ptotal=pmeta+pobs+pcarro
    #calcular gradientes
    dy, dx = np.gradient(ptotal, 1,1)
    skip = (slice(None, None, 20), slice(None, None, 20))
    fig,ax = plt.subplots()
    ax.quiver(x[skip], y[skip], dx[skip], dy[skip], ptotal[skip])
    ax.set(aspect=1, title='Quiver Plot2')

    for contador in range(lim):
        rect= patches.Rectangle((xACen[contador]-(w/2),yACen2[contador]-(w/2)),w,w,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)

    rect= patches.Rectangle((x_meta-(w/2),y_meta-(w/2)),w,w,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)

    rect= patches.Rectangle((x_carro -(40/2),y_carro -(40/2)),40,40,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)

    plt.show(block=False)
    plt.pause(2)
    


    print("Necesita moverse")


    
