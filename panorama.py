# By Olivier Racette, 40017231
# COMP 425 Term project: panorama image stitching

import cv2 as cv
import numpy as np

IMG_1_PATH = "project_images/Rainier1.png"
IMG_2_PATH = "project_images/Rainier2.png"

#Uses openCV's SIFT implementation to get keypoints in two images and matches them using ratio test
def findMatches(img1, img2):
	return 0


#From pdf:
# Projects point (x1, y1) using the homography H. 
# Returns the projected point (x2, y2)
def project(x1, y1, H, x2, y2):
	return 0


#From pdf:
# computes the number of inlying points given a homography "H". 
# That is, project the first point in each match using the function "project". 
# If the projected point is less than the distance "inlierThreshold" from the second point, it is an inlier. 
# Returns the total number of inliers.
def computeInlierCount(H, matches, numMatches, inlierThreshold):
	return 0


# takes a list of potentially matching points between two images 
# returns the homography transformation that relates them. 
def RANSAC (matches , numMatches, numIterations, inlierThreshold, hom, homInv, image1Display, image2Display):
	return 0


# Stitches the images together to produce the result panorama
def stitch(image1, image2, hom, homInv, stitchedImage):
	return 0


def main():
	img1 = cv.imread(IMG_1_PATH)
	img2 = cv.imread(IMG_2_PATH)

	cv.waitKey(0)
	cv.destroyAllWindows()

if __name__ == '__main__':
	main()