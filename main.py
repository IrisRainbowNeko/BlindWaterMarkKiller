import cv2
import numpy as np
import argparse

def open_demo(image,size=(5,5)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    binary = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return binary

def close_demo(image,size=(5,5), iterations=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    binary = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel,anchor=(-1, -1), iterations=iterations)
    return binary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--img_path", default='imgs/test.jpg', type=str)
    parser.add_argument("--mark_low", default=1.0, type=float)
    parser.add_argument("--mark_high", default=8.0, type=float)
    parser.add_argument("--intensity", default=3, type=float)
    args = parser.parse_args()

    img = cv2.imread(args.img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    #print(np.max(laplacian))

    laplacian[args.mark_low>laplacian]=0
    laplacian[args.mark_high<laplacian]=0
    laplacian[laplacian>0]=255
    laplacian=laplacian.astype(np.uint8)

    laplacian=cv2.medianBlur(laplacian,3)
    laplacian = close_demo(laplacian,size=(7,7),iterations=5)
    laplacian = open_demo(laplacian, size=(3,3))
    laplacian = close_demo(laplacian,iterations=3)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(laplacian, connectivity=8)

    output=cv2.dilate(laplacian, (9,9))

    #cv2.imshow('test', laplacian)
    #cv2.waitKey()

    img_proc=img.astype(int)

    img_proc[output>0,:]=img_proc[output>0,:]+np.random.randint(-args.intensity,args.intensity, size=(np.sum(output>0),3))
    img_proc=np.clip(img_proc, 0, 255).astype(np.uint8)

    cv2.imwrite('proc.png', img_proc)