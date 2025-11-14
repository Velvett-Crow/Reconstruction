import cv2
import os

def extract_frames(video_path, output_dir):
    """
    Extract frames from video at specified time intervals using OpenCV
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        interval_seconds: Time interval between frames (in seconds)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration:.2f} seconds")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        filename = f"frame_{frame_count:04d}.jpg"
        filepath = os.path.join(output_dir, filename)
            
        # Save the frame
        cv2.imwrite(filepath, frame)
        # print(f"Saved: {filename}")
        
        frame_count += 1
    
    cap.release()
    print(f"Extraction complete. Saved {frame_count} frames.")

extract_frames("/home/jovyan/videos_renders/videos/dx_polyp_observation.avi", "/home/jovyan/videos_renders/frames/dx/originals")
