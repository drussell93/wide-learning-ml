import numpy as np
import cv2

canvas = np.zeros((300, 300, 3), dtype="uint8")

green = (0, 255, 0)
cv2.line(canvas, (10, 10), (290, 290), green)
cv2.line(canvas, (290, 10), (10, 290), green)
cv2.imshow("Line on Canvas", canvas)
cv2.waitKey(0)