import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
print args

image = cv2.imread(args["image"])
print "height: %d" % (image.shape[0])
print "width: %d" % (image.shape[1])
print "depth: %d" % (image.shape[2])

cv2.imshow("T-Rex", image)
cv2.waitKey(0)

cv2.imwrite("T-Rex-copy.png", image)