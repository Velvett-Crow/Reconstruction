import numpy as np
import cv2

def pixel_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    """
    Calculates the mean reprojection error across all images.
    
    Args:
        objpoints: List of 3D object points for each image.
        imgpoints: List of 2D detected image points for each image.
        rvecs: List of rotation vectors from calibration.
        tvecs: List of translation vectors from calibration.
        mtx: The 3x3 camera intrinsic matrix.
        dist: The lens distortion coefficients.
        
    Returns:
        image_error: The total mean reprojection error in pixels.
        errors: A list of errors for each individual image.
    """
    
    cumulative_error = 0
    num_of_points = 0
    image_error = []
    
    for i in range (len(objpoints)):
        projpoints, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist) # reproject back to 2D image plane
        error_per_image = cv2.norm(imgpoints[i], projpoints, cv2.NORM_L2) # calculate Eulidean distance per point between detected points and reprojected ones
        
        cumulative_error += (error_per_image) ** 2
        num_of_points += len(projpoints)
        image_error.append(error_per_image)
        
    print(cumulative_error)
    mean_error = cumulative_error / num_of_points  # mean error across all points
    return cumulative_error, image_error
