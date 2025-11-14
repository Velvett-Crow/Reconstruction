import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt
from photo_error import pixel_error

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

checkerboard_size = (6, 7) #
square_size = 2.0
images_path = '/home/jovyan/videos_renders/frames/calibration/EndoSLAM/*.jpg'

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((checkerboard_size[0] * checkerboard_size[1],3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)

objp *= square_size  # scale by size of the squares

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob(images_path)
#print(f'{len(images)}')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (6, 7), None) #

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        # cv.drawChessboardCorners(img, (4, 4), corners2, ret) #
        # cv.imshow('img', img)
        # cv.waitKey(500)

cv.destroyAllWindows()

print(len(objpoints))

# Calibrating the camera
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

calibrated_data = {
    'camera_matrix': mtx,
    'dist_coeffs': dist,
    # 'rvecs': rvecs,
    # 'tvecs': tvecs,
    'reprojection_error': ret,
    # 'objpoints': objpoints,
    # 'imgpoints': imgpoints
}
# print(calibrated_data)

# print(f'{len(objpoints[1])}')
# print(f'{len(imgpoints[1])}')
# print(f'{len(rvecs[1])}')
# print(f'{len(tvecs[1])}')
# print(objpoints[1])
# print(rvecs[1])

# the intrinsic matrix
fx = mtx[0, 0]
fy = mtx[1, 1]
cx = mtx[0, 2]
cy = mtx[1, 2]

print(f"\nFocal Length (fx, fy): ({fx:.2f}, {fy:.2f}) pixels")
print(f"Principal Point (cx, cy): ({cx:.2f}, {cy:.2f}) pixels")
print(f"RMS Reprojection Error: ({ret:.2f}) pixels")
print(f"Distortion Coefficients: {dist}")

'''
# Undistorting one of the images
img = cv.imread(images[1])
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))  # New optimal camera matrix

dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst)
'''

'''
# Display results
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Original Image (Distorted)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB))
plt.title('Undistorted Image')
plt.axis('off')

plt.tight_layout()
plt.show()
'''

# Calculating mean reprojection error and error per image
accumulated_error, image_error = pixel_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist)

print(accumulated_error)

plt.plot(image_error)
plt.xlabel('Image Index')
plt.ylabel('Error per Image [pixels]')
plt.title('Reprojection Error of the Calibration Images')
plt.savefig("Reprojection Error.png")
