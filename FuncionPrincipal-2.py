import cv2
import numpy as np

from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import pylab as pl
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

import imutils
from contextlib import contextmanager

import serial,time 

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

def tomar_foto(file):
    #foto = cv2.VideoCapture('http://192.168.43.158:8080/shot.jpg')
    #foto = cv2.VideoCapture('http://192.168.43.25:8080/shot.jpg')
    #foto = cv2.VideoCapture('http://192.168.43.194:8080/shot.jpg')
##    foto = cv2.VideoCapture('http://192.168.227.39:8080/shot.jpg')
  #  foto = cv2.VideoCapture('http://192.168.197.104:8080/shot.jpg')
    foto = cv2.VideoCapture('http://192.168.43.236:8080/shot.jpg')
    #foto = cv2.VideoCapture('http://192.168.5.20:8080/shot.jpg')
    #foto = cv2.VideoCapture('http://192.168.227.13:8080/shot.jpg')
#    foto = cv2.VideoCapture(0)
    retval, im=foto.read()
    cv2.imwrite(file, im)
    image = cv2.imread(file)
    cv2.imshow("Foto Pista", image)
    #del(foto)
    return image

    

def scroll_bar(hsv,image,nameWin):
    res_c=0
    def nothing(x):
        pass
    img = np.zeros((300,512,3), np.uint8)
    cv2.namedWindow('Scroll bar color')

    # create trackbars for color change
    cv2.createTrackbar('R','Scroll bar color',0,255,nothing)
    cv2.createTrackbar('G','Scroll bar color',0,255,nothing)
    cv2.createTrackbar('B','Scroll bar color',0,255,nothing)

    # create switch for ON/OFF functionality
    switch = '0 : OFF \n1 : ON'
    cv2.createTrackbar(switch, 'Scroll bar color',1,0,nothing)

    while(1):
        cv2.imshow('Scroll bar color',img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        # get current positions of four trackbars
        r = cv2.getTrackbarPos('R','Scroll bar color')
        g = cv2.getTrackbarPos('G','Scroll bar color')
        b = cv2.getTrackbarPos('B','Scroll bar color')
        sw = cv2.getTrackbarPos(switch,'Scroll bar color')

        if sw == 1:
            img[:] = [b,g,r]
            #print(b,g,r)
            bgr = np.uint8([[[b,g,r ]]])
            bgr_hsv = cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
            #print(bgr_hsv)
            h=bgr_hsv[0][0][0]
            s=bgr_hsv[0][0][1]
            v=bgr_hsv[0][0][2]
            #print(h,s,v)
            if h<235 and h>20:
                hU=h+20
                hL=h-20
            else:
                hU=255
                hL=0
                
            if s<225 and s>30:
                sU=s+30
                sL=s-30
            else:
                sU=255
                sL=0
                
            if v<215 and v>40:
                vU=v+40
                vL=v-40
            else:
                vU=255
                vL=0
            #print(hL,sL,vL)
            #print(hU,sU,vU)

            res_c=seg_foto(hsv,hL,sL,vL,hU,sU,vU,image)
            cv2.imshow(nameWin, res_c)
            
        else:
            img[:] = 0
            break
    return res_c,hL,sL,vL,hU,sU,vU
    

def seg_foto(hsv,hL,sL,vL,hU,sU,vU,image):

    #Rango de colores detectados:
    #Verdes:
    color_bajos = np.array([hL, sL, vL], dtype=np.uint8)
    color_altos = np.array([hU, sU, vU], dtype=np.uint8)

    #Crear las mascaras
    mascara_color = cv2.inRange(hsv, color_bajos, color_altos)

    #Filtrar el ruido aplicando un OPEN seguido de un CLOSE
    kernel = np.ones((6,6),np.uint8)
    mascara_color = cv2.morphologyEx(mascara_color, cv2.MORPH_CLOSE, kernel)
    mascara_color = cv2.morphologyEx(mascara_color, cv2.MORPH_OPEN, kernel)
        
    # Bitwise-AND mask and original image
    res_Seg = cv2.bitwise_and(image,image, mask= mascara_color)
    #cv2.imshow("Segmentacion", res_Seg)
    return res_Seg


def centro(nombre,colorCent):
    gray = cv2.cvtColor(colorCent, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

    # find contours in the thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    
    MIN_THRESH=0.00000001
    #############
    ###Para Unir los vectores
    rec=0;
    pos=0;
    xCen=0
    yCen=0
    xACen=0
    yACen=0
    # loop over the contours
    for c in cnts:
        if cv2.contourArea(c) > MIN_THRESH:
            # process the contour
            # compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            xCen=cX
            yCen=cY
           # print(xCen,yCen)
        
            # draw the contour and center of the shape on the image
            cv2.drawContours(colorCent, [c], -1, (0, 255, 0), 2)
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
            else:
                xAux=np.append(xAux.T,xCor.T)
                yAux=np.append(yAux.T,yCor.T)
                #print(xACen,yACen)
                xACen=np.append(xACen,xCen)
                yACen=np.append(yACen,yCen)
                             
            rec=rec+1;
        
    cv2.imshow(nombre, colorCent)
    return xACen,yACen

def etiquetado(nombre,resultado):
    w=0
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
        rectan=cv2.rectangle(resultado,(x,y),(x+w,y+h),(0,255,0),2)
        #print(rectan)
        rect = cv2.minAreaRect(cr)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(resultado,[box],0,(0,0,255),2)
        cv2.putText(resultado, "#{}".format(label), (int(x) - 10, int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        #print(box)
        cv2.imshow(nombre, resultado)
    return w

    
def main():
    arduino=serial.Serial("COM3",9600)
    y, x = np.mgrid[0:480:480j, 0:640:640j]
    
    
    nombre_foto="pista-3B.jpg"
    image=tomar_foto(nombre_foto)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    nombre1="Segmentacion-1"
    res_O,hL,sL,vL,hU,sU,vU=scroll_bar(hsv,image,nombre1)
##    print("Limites",hL,sL,vL,hU,sU,vU)
    cv2.imshow('Obstaculos', res_O)
    nombre=("Centros Obstaculos")
    xCenO,yCenO=centro(nombre,res_O)
    nombre="Etiquetas Obstaculos"
    wO=etiquetado(nombre,res_O)
    cv2.destroyWindow(nombre1)
    print(xCenO)
    print(yCenO)
    xCenO=np.append(xCenO,0)
    yCenO=np.append(yCenO,0)
    print(xCenO)
    print(yCenO)

    #----------------------Campo potencia de lo obstaculos
    alpha_obstacle, a_obstacle, b_obstacle = 1.0, 1.9e3, 1.9e3
    pobs=0
    lim=np.size(xCenO)
    print('Longitud del vector ',lim)
    yACen2=np.arange(lim)
    lim2=np.size(yACen2)
    print('Longitud del vector ',lim2)
    for contador in range(lim-1):
        yACen2[contador]=480-yCenO[contador]
        x_obstacle = xCenO[contador-1]
        y_obstacle = yACen2[contador-1]
        p1= -alpha_obstacle * np.exp(-((x - x_obstacle)**2 / a_obstacle + (y - y_obstacle)**2 / b_obstacle))
        pobs=p1+pobs

    nombre1="Segmentacion-2"
    res_M,hL,sL,vL,hU,sU,vU=scroll_bar(hsv,image,nombre1)
    cv2.imshow('Meta', res_M)
    nombre=("Centros Meta")
    xCenM,yCenM=centro(nombre,res_M)
    nombreM="Etiqueta Meta"
    wM=etiquetado(nombreM,res_M)
    cv2.destroyWindow(nombre1)
    print(xCenM,480-yCenM)

    nombre1="Segmentacion-3"
    res_R1,hL1,sL1,vL1,hU1,sU1,vU1=scroll_bar(hsv,image,nombre1)
##    print("Limites",hL1,sL1,vL1,hU1,sU1,vU1)
    cv2.destroyWindow(nombre1)
    nombre1="Segmentacion-4"
    res_R2,hL2,sL2,vL2,hU2,sU2,vU2=scroll_bar(hsv,image,nombre1)
##    print("Limites",hL2,sL2,vL2,hU2,sU2,vU2)
    cv2.destroyWindow(nombre1)

##    captura = cv2.VideoCapture('http://192.168.5.20:8080/video')
##    captura = cv2.VideoCapture('http://192.168.227.39:8080/video')
    captura = cv2.VideoCapture('http://192.168.43.236:8080/video')
    #captura =cv2.VideoCapture('http://192.168.43.158:8080/video')
    #captura =cv2.VideoCapture('http://192.168.43.194:8080/video')
    #captura =cv2.VideoCapture('http://192.168.43.25:8080/video')
    #captura = cv2.VideoCapture('http://192.168.227.13:8080/video')
#    captura = cv2.VideoCapture(0)
    #Rango de colores detectados:
    #Azules para el robot: 
    azul_bajos = np.array([hL1,sL1,vL1], dtype=np.uint8)
    azul_altos = np.array([hU1,sU1,vU1], dtype=np.uint8)
    #Azules para la marca de orientación: 
    rosa_bajos = np.array([hL2,sL2,vL2], dtype=np.uint8)
    rosa_altos = np.array([hU2,sU2,vU2], dtype=np.uint8)
    
    nombre="Centro del Robot"
    nombre2="Centro dos"


    
    #----------------------Campo potencial de la meta
    x_meta = xCenM
    y_meta = 480-yCenM
    alpha_meta, a_meta, b_meta= 1.0, 3e4, 3e4
    pmeta = alpha_meta * np.exp(-((x - x_meta)**2 / a_meta + (y - y_meta)**2 / b_meta))
    contador2=0

    while(1):
        #Capturamos una imagen y la convertimos de RGB -> HSV
        ret, imagen = captura.read()
        cv2.imshow('Imagen Original', imagen)
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
     
        #Crear las mascaras
        mascara_azul = cv2.inRange(hsv, azul_bajos, azul_altos)
        mascara_rosa = cv2.inRange(hsv, rosa_bajos, rosa_altos)
        
        #Filtrar el ruido aplicando un OPEN seguido de un CLOSE
        kernel = np.ones((6,6),np.uint8)
        mascara_azul = cv2.morphologyEx(mascara_azul, cv2.MORPH_CLOSE, kernel)
        mascara_azul = cv2.morphologyEx(mascara_azul, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((6,6),np.uint8)
        mascara_rosa = cv2.morphologyEx(mascara_rosa, cv2.MORPH_CLOSE, kernel)
        mascara_rosa = cv2.morphologyEx(mascara_rosa, cv2.MORPH_OPEN, kernel)

        # Bitwise-AND mask and original image
        res_b = cv2.bitwise_and(imagen,imagen, mask= mascara_azul)
        res_r = cv2.bitwise_and(imagen,imagen, mask= mascara_rosa)
        
        # show the image
        #cv2.imshow("Imagen Azul", res_b)
        
        #Ubicar el centro del robot
        xcenR,ycenR=centro(nombre,res_b)
        xcenR1,ycenR1=centro(nombre2,res_r)
        #print("Centro Robot",xcenR,480-ycenR)
        #print("Centro Marca",xcenR1,480-ycenR1)

        print(wM)

        if contador2>=100:
            if np.size(xcenR)==1 and np.size(xcenR1)==1:
                #Campo potencial del carro
                x_carro = xcenR
                y_carro = 480-ycenR
                alpha_carro, a_carro, b_carro= 1.0, 3.9e3, 3.9e3
                pcarro = -alpha_carro * np.exp(-((x - x_carro)**2 / a_carro + (y - y_carro)**2 / b_carro))    

            #----------------------Campo potencia de lo obstaculos
                alpha_obstacle, a_obstacle, b_obstacle = 1.0, 1.9e3, 1.9e3
                pobs=0
                lim=np.size(xCenO)
                yACen2=np.arange(lim)
                lim2=np.size(yACen2)
                for contador in range(lim-1):
                    yACen2[contador]=480-yCenO[contador]
                    x_obstacle = xCenO[contador-1]
                    y_obstacle = yACen2[contador-1]
                    p1= -alpha_obstacle * np.exp(-((x - x_obstacle)**2 / a_obstacle + (y - y_obstacle)**2 / b_obstacle))
                    pobs=p1+pobs

                #----------------------Campo potencial de la meta
                x_meta = xCenM
                y_meta = 480-yCenM
                alpha_meta, a_meta, b_meta= 1.0, 3e4, 3e4
                pmeta = alpha_meta * np.exp(-((x - x_meta)**2 / a_meta + (y - y_meta)**2 / b_meta))
                
                ptotal=pmeta+pobs+pcarro       #Suma de todos los campos

                print("Centro Robot",xcenR,480-ycenR)
                print("Centro Marca",xcenR1,480-ycenR1)
        
                #Genera los gradientes
                dy, dx = np.gradient(ptotal, 1,1)
                skip = (slice(None, None, 3), slice(None, None, 3))
                fig,ax = plt.subplots()
                ax.quiver(x[skip], y[skip], dx[skip], dy[skip], ptotal[skip])
                ax.set(aspect=1, title='Quiver Plot2')
                print (dx[480-ycenR1][xcenR1], dy[480-ycenR1][xcenR1])

                #Calcula el angulo anterior
                Angulo_Anterior=angulo=math.atan(((480-ycenR1)-(480-ycenR))/(xcenR1-xcenR))*180/3.1416
                print ('Angulo_Anterior', Angulo_Anterior)

                #Calcula el angulo deseado
                Angulo_Deseado=angulo=math.atan((dy[480-ycenR1][xcenR1])/(dx[480-ycenR1][xcenR1]))*180/3.1416
                print ('Angulo Deseado', Angulo_Deseado)
                
                #Angulo a mover
                Angulo_Mover=Angulo_Deseado-Angulo_Anterior
                print('Angulo a mover', Angulo_Mover)
                time.sleep(1)
                print("limite x-",(x_meta)-(wM*.5))
                print("limite x+",(x_meta)+(wM*.5))
                print("limite y-",(y_meta)-(wM*.5))
                print("limite y+",(y_meta)+(wM*.5))
                
                if ((480-ycenR)>((y_meta)-(wM*.5)) and (480-ycenR)<((y_meta)+(wM*.5))):
                    if (xcenR>((x_meta)-(wM*.5)) and xcenR<((x_meta)+(wM*.5))):
                        arduino.write(b'3')
                    
                if Angulo_Mover <0:
                    arduino.write(b'5')
                    
                if Angulo_Mover >0:
                    arduino.write(b'4')
                    
                #if Angulo_Mover >= 0 and Angulo_Mover<15:
                #    arduino.write(b'1')
                
                contador2=0
            ######
        contador2=contador2+1
        tecla = cv2.waitKey(5) &     0xFF
        if tecla == 27:
            break
            
if __name__ == "__main__":
    main()
