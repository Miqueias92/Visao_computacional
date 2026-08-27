import cv2 # OpenCV
import numpy as np

## versão do opencv
print(cv2.__version__)

## carregando imagem
imagem = cv2.imread('person.jpg')

# tamanho da imagem
print(imagem.shape)

# Esse resultado, 840000, é o número total de valores individuais que compõem a imagem. Ou seja:

#500 × 560 = 280.000 pixels no total
#Cada pixel tem 3 componentes de cor (B, G, R)
# 280.000 × 3 = 840.000 valores guardados na memória
print(500 * 560 * 3)

#cv2.imshow('janela', imagem)
#cv2.waitKey(0)  
#cv2.destroyAllWindows()  

# imagem em cinza
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

print(imagem_cinza.shape)

#cv2.imshow('janela', imagem_cinza)
#cv2.waitKey(0)  
#cv2.destroyAllWindows() 

# criando o detector de faces
detector_facial = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

deteccoes = detector_facial.detectMultiScale(imagem_cinza)

print(deteccoes)
    # X  Y  dimensoes
##[[177 123 197 197]]

## mostra quantas faces foram detectadas na imagem
print(len(deteccoes))

for (x, y, w, h) in deteccoes:
    #print(x, y, w, h)
    # (imagem, (posicaoXY), (posicaofinal), cor(BGR), borda)
    cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 255), 4)
cv2.imshow('janela', imagem)
cv2.waitKey(0)  
cv2.destroyAllWindows() 