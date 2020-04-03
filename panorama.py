# By Olivier Racette, 40017231
# COMP 425 Term project: panorama img stitching

import cv2 as cv
import numpy as np
import random
import math

IMG_1_PATH = "project_images/Rainier1.png"
IMG_2_PATH = "project_images/Rainier2.png"
RESULTS_DIR = "results/"

ITERATIONS = 1000
THRESHOLD = 0.995

DEBUG_MODE = True


#Utility function to show an img and wait for key press
def showImg(title, img):
	cv.imshow(title, img)
	cv.waitKey(0)


#c(H)
#Where H is the Harris matrix
def cornerStrength(H):
	#det 	[a b 
	#		c d] 
	# 	= ad-bc

	#trace	[a b
	# 		c d] 
	#	= a+d

	det = H[0,0] * H[1,1] - H[0,1] * H[1,0]
	trace = H[0,0] + H[1,1]

	#I get trace = 0 for some pixels...nobody likes to divide by 0
	if trace == 0:
		trace = 0.0000001

	return det / trace


#Computes the Harrix matrix for each pixel
#Note that in the pdf, 'I' means img derivative!
#Returns KeyPoints found
#code from A2
def harrisDetector(img, windowSize, threshold):
	#print("Detecting features...")

	properties = img.shape
	height = properties[0]
	width = properties[1]

	#Converting the img to floating point grayscale...(from 0 to 1) --> not needed anymore, its done outside the function
	gray = np.float32(cv.cvtColor(img,cv.COLOR_BGR2GRAY)) / 255

	#computing derivatives
	dx = cv.Sobel(gray, -1, 1, 0)
	dy = cv.Sobel(gray, -1, 0, 1)
	dxdy = dx*dy

	dx2 = dx**2
	dy2 = dy**2

	#smoothing the derivatives with GaussianBlur of size 3*3, std dev 1.5
	dx2 = cv.GaussianBlur(dx2, (3, 3), 1.5)
	dy2 = cv.GaussianBlur(dy2, (3, 3), 1.5)
	dxdy = cv.GaussianBlur(dxdy, (3, 3), 1.5)

	#list to hold all of the results
	keypoints = []

	#temporary img to hold all the corner vales for each pixel
	cValues = np.zeros((height, width))

	windowOffset = (int)(windowSize/2)

	#Here we need to pass through the img with a "window" of x,y pixels
	for y in range(height):
		for x in range(width):
			xStart = x - windowOffset
			xEnd = x + windowOffset

			yStart = y - windowOffset
			yEnd = y + windowOffset

			#Taking advantage of numpy array slicing
			#Having for loops here drastically impacts performance!!
			dx2Vals = dx2[yStart:yEnd, xStart:xEnd]
			dy2Vals = dy2[yStart:yEnd, xStart:xEnd]
			dxdyVals = dxdy[yStart:yEnd, xStart:xEnd]

			dx2Sum = dx2Vals.sum()
			dy2Sum = dy2Vals.sum()
			dxdySum = dxdyVals.sum()

			H = np.array([[dx2Sum, dxdySum],[dxdySum, dy2Sum]], dtype=np.float32)

			cValues[y][x] = cornerStrength(H)

	smoothed = cv.GaussianBlur(gray, (3, 3), 1.5)

	#Now looping over the corner strengths to check which are valid
	for y in range(height):
		for x in range(width):
			#Need to check if c is over the user-defined threshold
			#Then check if its a local maximum in at least a 3x3 neighborhood
			if cValues[y][x] > threshold:
				if cValues[y][x] >= np.amax(cValues[y-2:y+2, x-2:x+2]):
					xDiff = smoothed[y][x+1] - smoothed[y][x-1]
					yDiff = smoothed[y+1][x] - smoothed[y-1][x]
					#angle = calcOrientation(xDiff, yDiff)

					#Keypoint constructor:
					#float x, float y, float _size, float _angle=-1, float _response=0, int _octave=0, int _class_id=-1
					keypoints.append(cv.KeyPoint(x, y, windowSize, 0, cValues[y][x], 0, -1))

	return np.asarray(keypoints)


#Uses openCV's SIFT implementation to get keypoints and make their descriptors
def findFeatures(img, save, debug = False):
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	
	keypoints = harrisDetector(img, 9, 10)

	#NOTE: I compiled openCV with contrib and the ENABLE_NON_FREE tag
	sift = cv.xfeatures2d.SIFT_create()

	#keypoints, descriptors = sift.detectAndCompute(gray, None) 	#Changed this to only use compute and use my own corner detection
	#My corner detection is much less efficient but it still works
	keypoints, descriptors = sift.compute(gray, keypoints)

	result = cv.drawKeypoints(img, keypoints, None, color=(0,255,0), flags = 0)

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
	matches = matcher.knnMatch(desc1, desc2, 2)	#to apply ratio test, we need k = 2

	ratio = 0.8 	#proposed by David Lowe in his paper

	goodMatches = []

	#ratio test: 
	# d1 / d2 < ratio --> d1 < ratio * d2 (to prevent division by 0)
	for i in range(len(matches)):
		if matches[i][0].distance < ratio * matches[i][1].distance:
			goodMatches.append(matches[i][0])


	resultImg = cv.drawMatches(img1, kp1, img2, kp2, goodMatches, None, flags = 2)

	if debug:
		showImg("Matches", resultImg)

	cv.imwrite(RESULTS_DIR + "2.png", resultImg)

	return goodMatches, kp1, kp2


#Projects point (x1, y1) using the homography H. 
#Returns the projected point newP (x2, y2)
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

	newP = np.matmul(H, (x1, y1, 1))	#Oddly, some times I get a runtime warning here (program still completes). Cause: H is filled of NaN / None values.

	#In the case of H being filled with None, w = 0....
	if newP[2] == 0:
		return newP[0], newP[1]

	return (newP[0] / newP[2]), (newP[1] / newP[2])


#From pdf:
#	computes the number of inlying points given a homography "H". 
#	That is, project the first point in each match using the function "project". 
#	If the projected point is less than the distance "inlierThreshold" from the second point, it is an inlier. 
#	Returns the total number of inliers.
#
# NOTE: I modified this so it always returns a list of inliers. I just use len() afterwards to get the total.
def computeInliers(H, matches, inlierThreshold, keypoints1, keypoints2):
	inliers = []

	for m in matches:
		projection = project(keypoints1[m.queryIdx].pt[0], keypoints1[m.queryIdx].pt[1], H)
		distSq = (projection[0] - keypoints2[m.trainIdx].pt[0])**2 + (projection[1] - keypoints2[m.trainIdx].pt[1])**2 

		if distSq < inlierThreshold:
			inliers.append(m)

	return inliers


#Takes a list of potentially matching points between two images.
#Returns the homography transformation and its inverse. 
def RANSAC (matches, numIterations, inlierThreshold, img1, img2, keypoints1, keypoints2, debug = False):
	bestH = np.zeros((3,3))

	numMatches = len(matches)

	maxInliers = 0

	#for each iteration
	#	select 4 random matches
	#	compute homography using opencv method
	#	compute the number of inliers from this homography
	#	get the maximum number of inliers

	sampleSize = 4

	for i in range(numIterations):
		src = np.zeros((sampleSize, 2))
		dst = np.zeros((sampleSize, 2))

		for i in range(sampleSize):
			idx = random.randint(0, numMatches-1)
			src[i] = keypoints1[matches[idx].queryIdx].pt  	#matches act as a list of pointers to keypoints
			dst[i] = keypoints2[matches[idx].trainIdx].pt

		currH = cv.findHomography(src, dst, 0)[0]
		currInliers = len(computeInliers(currH, matches, inlierThreshold, keypoints1, keypoints2))

		if currInliers > maxInliers:
			bestH = currH
			maxInliers = currInliers

	#After iterations:
	#find matches that are inliers using the "best" homography and the specified threshold
	#compute another homography with all of the inliers (not just 4 points)
	inliers = computeInliers(bestH, matches, inlierThreshold, keypoints1, keypoints2)

	src = np.array([keypoints1[i.queryIdx].pt for i in inliers])	#is this what people call "pythonic" code?
	dst = np.array([keypoints2[i.trainIdx].pt for i in inliers])

	bestH = cv.findHomography(src, dst, 0)[0]

	#Finally, displaying the inlier matches
	inlierImg = cv.drawMatches(img1, keypoints1, img2, keypoints2, inliers, None, flags = 2)

	if debug:
		showImg("Inliers", inlierImg)

	cv.imwrite(RESULTS_DIR + "3.png", inlierImg)

	return bestH, np.linalg.inv(bestH)


#Stitches the images together using H and its inverse to produce the panorama
def stitch(img1, img2, hom, homInv):
	sitchImg = 0

	#First, compute the size of the panorama. Project the corners of img2 on img1 with inverse of H. Allocate memory for the panorama.
	img1Height = img1.shape[0]
	img1Width = img1.shape[1]

	img2Height = img2.shape[0]
	img2Width = img2.shape[1]

	#Getting the four corners of the projection of img 2 onto img 1
	topLeft = project(0, 0, homInv)
	topRight = project(img2Width, 0, homInv)
	bottomLeft = project(0, img2Height, homInv)
	bottomRight = project(img2Width, img2Height, homInv)

	#Panorama sides relative to img 1
	panoTop = min(topLeft[1], topRight[1], 0)
	panoBottom = max(bottomLeft[1], bottomRight[1], img1Height)
	panoLeft = min(topLeft[0], bottomLeft[0], 0)
	panoRight = max(topRight[0], bottomRight[0], img1Width)

	panoWidth = math.ceil(abs(panoLeft) + abs(panoRight))
	panoHeight = math.ceil(abs(panoTop) + abs(panoBottom))

	stitchImg = np.zeros((panoHeight, panoWidth, 3), np.uint8)	#always mind the type!

	#Second, copy img1 onto the panorama
	xOffset = math.ceil(0 - panoLeft)
	yOffset = math.ceil(0 - panoTop)

	#Perhaps theres a better way to copy an img onto another...These are just numpy arrays after all. For now this will do fine.
	for y in range(img1Height):
		for x in range(img1Width):
			stitchImg[y+yOffset][x+xOffset] = img1[y][x]


	#Third, for every pixel p in the panorama, project p onto img2. If it's correctly inside img2, blend the pixel values of img2 and panorama.
	#	protip: use bilinear interpolation to get pixel values of img2 -> cv.getRectSubPix(img, patchSize, center[, patch[, patchType]]
	for y in range(panoHeight):
		for x in range(panoWidth):
			xp, yp = project(x-xOffset, y-yOffset, hom)

			if xp < img2Width and xp >= 0 and yp < img2Height and yp >= 0:
				stitchImg[y][x] = cv.getRectSubPix(img2, (1,1), (xp, yp))		#making use of getRectSubPix's interpolation to fetch single pixels at floating point values

	cv.imwrite(RESULTS_DIR + "4.png", stitchImg)

	return stitchImg


def main():
	img1 = cv.imread(IMG_1_PATH)
	img2 = cv.imread(IMG_2_PATH)

	#Mandatory step 1: find features of boxes, save in 1a.png
	findFeatures(cv.imread("project_images/Boxes.png"), RESULTS_DIR + "1a.png", DEBUG_MODE)

	#Mandatory step 2: find features of mount rainier 1, save in 1b.png. Find features of mount rainier 2, save in 1c.png
	matches, kp1, kp2 = findMatches(img1, img2, DEBUG_MODE)

	#Mandatory step 3: compute the homography using RANSAC
	h, hInv = RANSAC(matches, ITERATIONS, THRESHOLD, img1, img2, kp1, kp2, DEBUG_MODE)

	#Mandatory step 4: stitch the images together
	panorama = stitch(img1, img2, h, hInv)

	showImg("Panorama", panorama)

	cv.destroyAllWindows()

if __name__ == '__main__':
	main()