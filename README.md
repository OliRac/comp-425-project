# comp-425-project

Python Image Stitching

Made by Olivier Racette, id 40017231

## OpenCV

This program uses OpenCV & OpenCV_Contrib compiled with the ENABLE_NON_FREE tag.

## Description

Produces a panorama from a series of (2) images.

Uses RANSAC to find the appropriate homography matrix between the two.

Saves all progress in the "results" directory under images:

 - 1a.png (corner matching)
 - 1b.png (features of image 1)
 - 1c.png (features of image 2)
 - 2.png (matching of features)
 - 3.png (removal of outliers)
 - 4.png (stitching)
 
The program is configured to present progress at each step; press any key for the program to continue when it does.
 
## Variables
 
The number of iterations for RANSAC can be modified at line 13 with the ITERATIONS variable. (default = 1000)
 
The threshold for RANSAC can be modified at line 14 with the THRESHOLD variable. (default = 0.995)
 
To prevent the program from showing progress at each step, DEBUG_MODE can be set to False at line 16.


## Method Details

### harrisDetector(img, windowSize, thresHold)

Computes the Harris matrix H for each pixel in the image.
Does so by first converting the image to a floating point grayscale, where pixel values are from 0 to 1.
Then the required derivaties are obtain throuth the use of OpenCV's Sobel() method. These are then blurred with a Gaussian.
The grayscale is parsed with a window of a user defined size; H matrices are computed and the corner strength indicators are put into a temporary image of the same size. The image is once again parsed to determine which pixels with appropriate corner values could potentially be keypoints. Keypoints are added to a list and returned.

### findFeatures(img, save, debug = False)

First, this uses my own harrisDetector to find keypoints in the passed image.
Then, OpenCV's SIFT is used to make the descriptors of each generated keypoint.
A new image with keypoints drawn on it is made and saved in the path "save".
If debug is true, it will show the image to the user and wait for a key press.

### findMatches(img1, img2, debug = False)

Uses findFeatures() to first find keypoints in img1 and img2.
OpenCV's BFMatcher() is used to get potential matche with k = 2, to get the two best matches.
The ratio test is then performed on each match to shave off bad matches.
Good matches are added to a list and returned.
Keypoints for img1 and img2 are also returned.
Creates a new image with img1, img2 and all good matches shown.
If debug is true, it will show the image to the user and wait for a key press.

### project(x1, y1, H)

Projects a point by the homography H
Makes use of numpy's matrix multiplication method matmul().
Returns the pair u/w and v/w.

### computeInliers(H, matches, inlierThreshold, keypoints1, keypoints2)

Given a homography H, this will compute the list of inlying points by the inlierThreshold.
Each match is projected using project() and if the distance squared is smaller than the threshold, it is added to a list.
The list of inliers is returned. (To get the number of inliers, simply call len()).

### RANSAC(matches, numIterations, inlierThreshold, img1, img2, keypoints1, keypoints2, debug = false)

Tries to find the homography that produces the best number of inliers between matches in img1 and img2.
To do this, this algorithm runs numIterations of times (this value can be changed at line 13). 

At each iteration, 4 random matches are chosen and a homography H is computed.
The number of inliers produced by H is calculated and compared to the best one found yet.
After all iterations, another homography is computed using all inliers of the best H found previously (not just 4 points).

Inlier matches are then drawn onto a new image and saved. 
If debug is true, it will show the image to the user and wait for a key press.
The final homography and its inverse are returned.

### stitch(img1, img2, hom, homInv)

Takes two images, hopefully of the same thing but at different perspectives, and builds a panorama.
First, the size of the panorama is calculated. To do this, each corner of img2 is projected to img1 space using hom. Then, the corners of img1 and projected corners img2 are compared to see which are the outlying ones. The outlying corners are the panorama's corners. From here, width & height can easily be calculated.

Second, img1 is copied onto the panorama. For each pixel of img1, it is copied onto the same location in the panorama, with an offset (-panorama left side for x axis and -panorama top side for y axis).

Third, img2 is copied onto the panorama. For each pixel p in the panorama, p is projected onto img2. The color of the projection is then copied in the panorama at p, using OpenCV's getRedSubPix() method (this method can use interpolation to get color at floating point coordinates), only if p's projection is actually in the bounds of img2.

The stitch is saved and returned.



## Other Notes

My corner detection isn't as precise as I would want it to be (observable with the first image; the keypoints aren't aligned perfectly with the cube corners) but it still works well enough to build a panorama.
