# By Olivier Racette, 40017231
# COMP 425 Term project: panorama image stitching

import cv2 as cv
import numpy as np
import random

IMG_1_PATH = "project_images/Rainier1.png"
IMG_2_PATH = "project_images/Rainier2.png"
RESULTS_DIR = "results/"

#utility function to show an image and wait for key press
def showImg(title, img):
	cv.imshow(title, img)
	cv.waitKey(0)


#Builds a list of keypoint objects from a list of dMatch objects
#dMatch only contains keypoint indices; they do not contain information like x, y, z position
def buildMatchList(dMatchList, kpList1, kpList2):
	matches = []

	for dm in dMatchList:
		matches.append((kpList1[dm.queryIdx],kpList2[dm.trainIdx]))

	return matches

#Uses openCV's ORB implementation to get keypoints and make their descriptors
def findFeatures(img, save, debug = False):
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

	#NOTE: I compiled openCV with contrib and the ENABLE_NON_FREE tag
	sift = cv.xfeatures2d.SIFT_create()

	keypoints, descriptors = sift.detectAndCompute(gray, None)

	result = cv.drawKeypoints(img, keypoints, None, color=(0,255,0), flags=0)

	if debug:
		showImg("Keypoints", result)

	if save is not None:
		cv.imwrite(save, result) 

	return keypoints, descriptors


#Uses ratio test to match images
#Returns a list of DMatch objects
def findMatches(img1, img2, debug = False):
	kp1, desc1 = findFeatures(img1, RESULTS_DIR + "1b.png", debug)
	kp2, desc2 = findFeatures(img2, RESULTS_DIR + "1c.png", debug)

	matcher = cv.BFMatcher()
	matches = matcher.knnMatch(desc1, desc2, k=2)	#to apply ratio test, we need k = 2

	ratio = 0.8 	#proposed by David Lowe in his paper

	goodMatches = []

	for i in range(len(matches)):
		if matches[i][0].distance < ratio * matches[i][1].distance:
			goodMatches.append(matches[i][0])


	resultImg = cv.drawMatches(img1, kp1, img2, kp2, goodMatches, None)

	if debug:
		showImg("Matches", resultImg)

	cv.imwrite(RESULTS_DIR + "2.png", resultImg)

	return buildMatchList(goodMatches, kp1, kp2)


#From pdf:
# Projects point (x1, y1) using the homography H. 
# Returns the projected point newP (x2, y2)
def project(x1, y1, H):
	#h = [ a b c 
	#	   d e f
	#	   g h 1 ]
	#
	# p = [x y (1)]
	#
	# newP = h * p = [u v w]
	# x2 = u / w
	# y2 = v / w

	newP = np.matmul(H, (x1, y1, 1))


	return (newP[0] / newP[2]), (newP[1] / newP[2])


#From pdf:
# computes the number of inlying points given a homography "H". 
# That is, project the first point in each match using the function "project". 
# If the projected point is less than the distance "inlierThreshold" from the second point, it is an inlier. 
# Returns the total number of inliers.
def computeInlierCount(H, matches, inlierThreshold):
	inliers = 0

	for m in matches:
		projection = project(m[0].pt[0], m[0].pt[1], H)
		distSq = (projection[0] - m[1].pt[0])**2 + (projection[1] - m[1].pt[1])**2 

		if distSq < inlierThreshold:
			inliers += 1

	return inliers


def showInliers(img1, img2, H, matches, inlierThreshold):
	return 0;

#Nearly identical to computeInliercount, but returns a list of inliers instead of a count
def findInliers(matches, H, inlierThreshold):
	inliers = []

	for m in matches:
		projection = project(m[0].pt[0], m[0].pt[1], H)
		distSq = (projection[0] - m[1].pt[0])**2 + (projection[1] - m[1].pt[1])**2 

		if distSq < inlierThreshold:
			inliers.append((m[0], m[1]))

	return inliers

# takes a list of potentially matching points between two images 
# returns the homography transformation and its inverse. 
def RANSAC (matches, numIterations, inlierThreshold, img1, img2):
	bestH = np.zeros((3,3))

	numMatches = len(matches)

	inliersMax = 0

	#for each iteration
	#	select 4 random matches
	#	compute homography using opencv method
	#	compute the number of inliers from this homography
	#	get the maximum number of inliers

	for i in range(numIterations):
		src = np.zeros((4, 2))
		dst = np.zeros((4, 2))

		for i in range(4):
			idx = random.randint(0, numMatches-1)
			src[i] = matches[idx][0].pt
			dst[i] = matches[idx][1].pt

		currH = cv.findHomography(src, dst, 0)[0]
		currInliers = computeInlierCount(currH, matches, inlierThreshold)

		if currInliers > inliersMax:
			bestH = currH
			inliersMax = currInliers

	#After iterations:
	#find matches that are inliers using the "best" homography and the specified threshold
	#compute another homography with all of the inliers (not just 4 points)

	inliers = findInliers(matches, bestH, inlierThreshold)

	src = np.array([i[0].pt for i in inliers])
	dst = np.array([i[1].pt for i in inliers])

	bestH = cv.findHomography(src, dst, 0)[0]

	#Finally, displaying the inlier matches

	return bestH, np.linalg.inv(bestH)


# Stitches the images together to produce the result panorama
def stitch(img1, img2, hom, homInv):
	sitchImg = 0



	return sitchImg


def main():
	img1 = cv.imread(IMG_1_PATH)
	img2 = cv.imread(IMG_2_PATH)

	debug = False

	iterations = 100
	threshold = 1

	#Mandatory step 1: find features of boxes, save in 1a.png
	findFeatures(cv.imread("project_images/Boxes.png"), RESULTS_DIR + "1a.png", debug)

	#Mandatory step 2: find features of mount rainier 1, save in 1b.png
	#Mandatory step 3: find features of mount rainier 2, save in 1c.png
	matches = findMatches(img1, img2, debug)

	h, hInv = RANSAC(matches, iterations, threshold, img1, img2)

	stitch(img1, img2, h, hInv)

	cv.destroyAllWindows()

if __name__ == '__main__':
	main()