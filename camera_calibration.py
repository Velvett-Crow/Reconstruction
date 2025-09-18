import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

checkerboard_size = (9, 9) #
square_size = 15.0
images_path = 'calibration_images/*.jpg'

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
    ret, corners = cv.findChessboardCorners(gray, (9, 9), None) #

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        # cv.drawChessboardCorners(img, (9, 9), corners2, ret) #
        # cv.imshow('img', img)
        # cv.waitKey(500)

cv.destroyAllWindows()

# Calibrating the camera
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

calibrated_data = {
    'camera_matrix': mtx,
    'dist_coeffs': dist,
    'rvecs': rvecs,
    'tvecs': tvecs,
    'reprojection_error': ret,
    # 'objpoints': objpoints,
    # 'imgpoints': imgpoints
}
print(calibrated_data)

# the intrinsic matrix
# fx = mtx[0, 0]
# fy = mtx[1, 1]
# cx = mtx[0, 2]
# cy = mtx[1, 2]

# print(f"\nFocal Length (fx, fy): ({fx:.2f}, {fy:.2f}) pixels")
# print(f"Principal Point (cx, cy): ({cx:.2f}, {cy:.2f}) pixels")

# Undistorting one of the images
img = cv.imread(images[7])
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))  # New optimal camera matrix

dst = cv.undistort(img, mtx, dist, None, newcameramtx)

'''
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst)
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