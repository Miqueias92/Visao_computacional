import cv2 # OpenCV
import numpy as np

## versão do opencv
print(cv2.__version__)

## carregando imagem
imagem = cv2.imread('person.jpg')

# tamanho da imagem
print(imagem.shape)