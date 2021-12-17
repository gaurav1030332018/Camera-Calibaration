# import numpy as np
# import cv2 as cv
# import glob
# chessboardSize = (24,17)
# frameSize = (1440,1080)
# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
# objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)
# # size_of_chessboard_squares_mm = 20
# # objp = objp * size_of_chessboard_squares_mm
# objpoints = [] # 3d point in real world space
# imgpoints = [] # 2d points in image plane.
# #imgpointsR = [] # 2d points in image plane.
#
#
# # imagesLeft = sorted(glob.glob('images/stereoLeft/*.png'))
# # imagesRight = sorted(glob.glob('images/stereoRight/*.png'))
# images=glob.glob('*.png')
# for image in images:
#     print(image)
#     img = cv.imread(image)
#     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # convert to gray scale
#     # Find the chess board corners
#     ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
#     # If found, add object points, image point (after refining them)
#     if ret == True:
#         objpoints.append(objp)
#         corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
#         imgpoints.append(corners)
#
#         # Draw and display the Corners
#         cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
#         #cv.imshow("img", img)
#         cv.waitKey(1000)
#
# cv.destroyAllWindows()
# ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
#
# print("Camera Calibration:", ret)
# print("\nCamera Matrix\n", cameraMatrix)
# print("\nDistortion Parameters:\n ", dist)
# print("\nRotation Vectors: \n", rvecs)
# print("\n Translation Vectors: \n", tvecs)
# img= cv.imread("11.png")
# h, w = img.shape[:2]
# newCameraMatrix, roi= cv.getOptimalNewCameraMatrix(cameraMatrix,dist, (w,h), 1, (w,h) )
# #undistort
# dst= cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)
# #crop the image
# x,y,w,h= roi
# dst= dst[ y:y+h, x:x+w]
# cv.imwrite("picResult9.png", dst)
# #undistort with remapping
# mapx, mapy= cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
# dst= cv.remap(img,mapx,mapy,cv.INTER_LINEAR)
#
# #crop the image
# x,y,w,h =roi
# dst= dst[y:y+ h, x:x+w]
# cv.imwrite("picResult2.png", dst)
# # Reprojection Error
# mean_error = 0
#
# for i in range(len(objpoints)):
#     imgPoint2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
#     error= cv.norm(imgpoints[i], imgPoint2, cv.NORM_L2)/len(imgPoint2)
#     mean_error += error
#
# print("\ntotal error: {}" .format(mean_error/len(objpoints)))
# print("\n\n\n")
import glob
import pickle
import cv2 as cv
import numpy as np

## find chessboard corners

chessboardsize = (24, 17) # coners i want to find
frameSize = (1440, 1080) # pixs of camera

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, .0001) ## finding subpixs

# prepare object point, line (0,0,0), (1,1,1), (2,0,0) .....,(6,5,0)
objp = np.zeros((chessboardsize[0] * chessboardsize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardsize[0], 0:chessboardsize[1]].T.reshape(-1, 2)

# arrays to store object point and image point from all the images.
objPoints = [] # 3d point in real world space
imgPoint = [] # 2d point in image plane

images = glob.glob("*.png")## all my images

for image in images:
    print(image)
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)# convert to gray scale
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardsize, None)

    # If found, add object points, image point (after refining them)
    if ret == True:

        objPoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgPoint.append(corners)

    # Draw and display the Corners
    cv. drawChessboardCorners(img, chessboardsize, corners2, ret)
    cv.imshow("img", img)
    cv.waitKey(500)

    cv.destroyAllWindows()

# print( "object Point : ", objpoint)
# print(image points: ", imgpoint)

##### Calibration#####

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objPoints, imgPoint, frameSize, None, None)

print("Camera Calibration:", ret)
print("\nCamera Matrix\n", cameraMatrix)
print("\nDistortion Parameters:\n ", dist)
print("\nRotation Vectors: \n", rvecs)
print("\n Translation Vectors: \n", tvecs)

#### undistortion #####

img= cv.imread("11.png")
h, w = img.shape[:2]
newCameraMatrix, roi= cv. getOptimalNewCameraMatrix(cameraMatrix,dist, (w,h), 1, (w,h) )

#undistort
dst= cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)
#crop the image
x,y,w,h= roi
dst= dst[ y:y+h, x:x+w]
cv.imwrite("picResult9.png", dst)

#undistort with remapping
mapx, mapy= cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
dst= cv.remap(img,mapx,mapy,cv.INTER_LINEAR)

#crop the image
x,y,w,h =roi
dst= dst[y:y+ h, x:x+w]
cv.imwrite("picResult2.png", dst)

# Reprojection Error
mean_error = 0

for i in range(len(objPoints)):
    imgPoint2, _ = cv.projectPoints(objPoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error= cv.norm(imgPoint[i], imgPoint2, cv.NORM_L2)/len(imgPoint2)
    mean_error += error

print("\ntotal error: {}" .format(mean_error/len(objPoints)))
print("\n\n\n")