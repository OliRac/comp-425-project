# By Olivier Racette, 40017231
# COMP 425 Term project: panorama image stitching

import cv2 as cv
import numpy as np

IMG_1_PATH = "project_images/Rainier1.png"
IMG_2_PATH = "project_images/Rainier2.png"

#utility function to show an image and wait for key press
def showImg(title, img):
	cv.imshow(title, img)
	cv.waitKey(0)

#Uses openCV's ORB implementation to get keypoints and make their descriptors
def findFeatures(img, debug = False):
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

	#NOTE: I compiled openCV with contrib and the ENABLE_NON_FREE tag
	sift = cv.xfeatures2d.SIFT_create()

	#orb = cv.ORB_create()
	#keypoints = orb.detect(gray, None)
	keypoints, descriptors = sift.detectAndCompute(gray, None)

	if debug:
		result = cv.drawKeypoints(img, keypoints, None, color=(0,255,0), flags=0)
		showImg("Keypoints", result)

	return keypoints, descriptors


#Uses ratio test to match images
def findMatches(img1, img2, debug = False):
	kp1, desc1 = findFeatures(img1, debug)
	kp2, desc2 = findFeatures(img2, debug)

	matcher = cv.BFMatcher()

	#to apply ratio test, k = 2
	matches = matcher.knnMatch(desc1, desc2, k=2)

	#This is taken directly from openCV documentation. try to make it your own!
	good = []

	for m, n in matches:
		if m.distance < 0.75*n.distance:
			good.append([m])

	if debug:
		matchImg = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
		showImg("Matches", matchImg)

	return good


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

	matches = findMatches(img1, img2, True)

	cv.waitKey(0)
	cv.destroyAllWindows()

if __name__ == '__main__':
	main()