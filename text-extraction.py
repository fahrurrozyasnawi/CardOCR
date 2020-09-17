import imutils
import cv2
import pytesseract
import numpy as np

custom_config = r'--oem 3 --psm 6'
img = cv2.imread("16.jpg")
img = imutils.resize(img, height=400)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('grayscale', gray)

# ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (36, 4))
sq_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
# cv2.imshow('Threshold', thresh1)

#smooth img
gray = cv2.GaussianBlur(gray, (5,5), 0)
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rect_kernel)
# cv2.imshow('Blur', gray)
cv2.imshow('Blackhat', blackhat)

#Compute Scharr Gradient
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
print("Min : ", minVal)
print("Max : ", maxVal)
gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
# cv2.imshow('gradX', gradX)
print('GradX : ', gradX)

#apply closing operation
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rect_kernel)
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow('Thresh', thresh)

#perform another closing operation
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, sq_kernel)
thresh2 = cv2.erode(thresh, None, iterations=1)
thresh3 = cv2.dilate(thresh, None, iterations=11)
cv2.imshow('Thresh Erosion', thresh2)
cv2.imshow('Thresh Dilate', thresh3)

#set borders
p = int(img.shape[1] * 0.001)
thresh3[:, 0:p] = 0
thresh3[:, img.shape[1] - p:] = 0
cv2.imshow('Thresh with set borders', thresh3)

#find contours
cnts = cv2.findContours(thresh3.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
# cv2.imshow('cnts', cnts)
# print("Contours : ", cnts)

#loop over contours
for c in cnts:
    #compute bbox
    (x, y, w, h) = cv2.boundingRect(c)
    # print("x : ", x)
    # print("y : ", y)
    # print("w : ", w)
    # print("h : ", h)
    aspect_ratio = w / float(h)
    crWidth = w / float(gray.shape[1])
    maxAR = np.max(aspect_ratio)
    maxCrWidth = np.max(crWidth)
    # print("Aspect Ratio", maxAR)
    # print("crWidth : ", maxCrWidth)

    #check aspect ratio
    if aspect_ratio > 3 and crWidth > 0.75:
        pX = int((x + w) * 0.03)
        pY = int((y + h) * 0.03)
        (x, y) = (x - pX, y - pY)
        (w, h) = (w + (pX * 2), h + (pY * 2))

    #extract ROI
    roi = img[y:y + h, x:x + w].copy()
    rs = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    result = pytesseract.image_to_string(roi, config=custom_config)
    print("Result : ", result)
    # print("ROI", roi)

cv2.imshow("Image", img)
cv2.imshow("ROI", roi)




# opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, rect_kernel)
# dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)
# erosion = cv2.erode(thresh1, rect_kernel, iterations=1)
# cv2.imshow('opening', opening)
# cv2.imshow('erosion', erosion)
# cv2.imshow('dilation', dilation)
cv2.waitKey(0)

# contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
#                                        cv2.CHAIN_APPROX_NONE)

# im2 = img.copy()
#
# file = open("recognized.txt", "w+")
# file.write("")
# file.close()

# for cnt in contours:
#     x, y, w, h = cv2.boundingRect(cnt)
#
#     rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#     cropped = im2[y:y + h, x:x + w]
#
#     file = open("recognized.txt", "a")
#     print(cropped)
#
#     text = pytesseract.image_to_string(cropped)
#
#     file.write(text)
#     file.write("\n")
#     file.close
