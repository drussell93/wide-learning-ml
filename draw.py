import numpy as np
import cv2

canvas = np.zeros((300, 300, 3), dtype = "uint8")
green = (0, 255, 0)
cv2.line(canvas, (10,10), (290, 290), green)
cv2.imshow("Canvas", canvas)
# cv2.waitKey(0)

red = (0, 0, 255)
cv2.line(canvas, (290, 10), (10, 290), red, 3)
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

cv2.rectangle(canvas, (10, 10), (60, 60), green, -1)
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

cv2.rectangle(canvas, (50, 200), (200, 225), red, 5)
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

blue = (255, 0, 0)
cv2.rectangle(canvas, (200, 50), (225, 125), blue)
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

canvas = np.zeros((300, 300, 3), dtype = "uint8")
for i in xrange(0, 5):
	radius = np.random.randint(5, high = 50)
	color = np.random.randint(50, high = 256, size = (3,)).tolist()
	
	pt = np.random.randint(50, high = 250, size = (2,))
	cv2.circle(canvas, tuple(pt), radius, color, 2)
	
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)
cv2.imwrite("circles.jpg", canvas)