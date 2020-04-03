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


## Other Notes

My corner detection isn't as precise as I would want it to be (observable with the first image; the keypoints aren't aligned perfectly with the cube corners) but it still works well enough to build a panorama.