'''Used to crop the rectified images to remove the circular boundary'''

import cv2
import os

input_folder = "/home/jovyan/videos_renders/frames/trial/rectified"
output_folder = "/home/jovyan/videos_renders/frames/trial/resized/rectified"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Cropping coordinates (y1:y2, x1:x2)
y1, y2 = 45, 275
x1, x2 = 45, 275

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):  # supported formats
        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path)

        # Crop
        cropped_image = image[y1:y2, x1:x2]

        # Save
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, cropped_image)

print(f"Cropping complete.")
