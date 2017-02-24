import cv2
import numpy as np

image = cv2.imread('/home/agneetc/Documents/WordSpot/BTECH/01/word0001.bmp')
gray_image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(gray_image)
print(gray_image.shape)
print("\n")
count=0
for num in range(1,1193):
    if(num%149==0):
        B=gray_image[:,count:num].copy()
        print(B)
        print(B.shape)
        count=num
        print("\n")
        s=np.zeros(147)
        for i in range(0,147):
            s[i]=(B[i].sum())
        print (s)
        print("\n \n \n")
        
            

    

