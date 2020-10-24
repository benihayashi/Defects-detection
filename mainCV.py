import cv2
import numpy as np

kernel = np.ones((5,5),np.uint8)
kernel2 = np.ones((3,3),np.uint8)
img = cv2.imread('./Decks/Non-Cracked/7001-234.jpg', cv2.COLOR_BGR2GRAY)
#imagem = cv2.bitwise_not(img)
#imagem = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
#imagem = cv2.resize(img,(64,64))
#cv2.imshow("image2",imagem)
imagem = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
imagem =  cv2.dilate(imagem,kernel2,iterations = 1)
cv2.imshow('image',imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()

