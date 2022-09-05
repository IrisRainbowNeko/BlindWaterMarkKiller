import cv2
import numpy as np


img = cv2.imread('proc.png')#[500:800,800:,:]

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

laplacian = cv2.Laplacian(gray, cv2.CV_64F)

cv2.imshow('test', laplacian)
cv2.waitKey()