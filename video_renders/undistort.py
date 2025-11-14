import cv2 as cv
import os
import glob
import numpy as np

def rectify(mtx, dist, input_dir, output_dir):
    
    images = glob.glob(os.path.join(input_dir, '*.jpg'))
    os.makedirs(output_dir, exist_ok=True)
    
    for k in range(len(images)):
        img = cv.imread(images[k])
        h,  w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h)) # new optimal camera matrix
        dst = cv.undistort(img, mtx, dist, None, newcameramtx)
        filename = os.path.basename(images[k])
        output = os.path.join(output_dir, filename)
        cv.imwrite(output, dst)
        
cam_mtx = np.array([[183.83, 0, 157.13], [0, 184.42, 168.92], [0, 0, 1]])

dist_coeffs = np.array([[-0.50391067, 0.36213787, -0.01563724, 0.00532104, 0.00579802]])

rectify(cam_mtx, dist_coeffs, '/home/jovyan/videos_renders/frames/trial/originals', '/home/jovyan/videos_renders/frames/trial/rectified')