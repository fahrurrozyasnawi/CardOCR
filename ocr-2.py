import numpy as np
import cv2
import imutils
import pytesseract

img = cv2.imread('14.jpg')  # Read input image

# Convert to hsv
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
s = hsv[:, :, 1]

ret, thresh = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Find contours
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnts = imutils.grab_contours(cnts)

c = max(cnts, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(c)
thresh_card = thresh[y:y+h, x:x+w].copy()

# OCR
result = pytesseract.image_to_string(thresh_card)
print(f"OCR Results:\n {result}")

# Show image
cv2.imshow('s', s)
cv2.imshow('thresh', thresh)
cv2.imshow('thresh_card', thresh_card)
cv2.waitKey(0)
cv2.destroyAllWindows()