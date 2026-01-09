"""
Calculates the mean reprojection error across all images.
"""

import numpy as np
import cv2

def pixel_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    
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
