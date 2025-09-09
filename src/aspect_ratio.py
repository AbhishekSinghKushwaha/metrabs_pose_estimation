import cv2
import os
from utils.file_utils import ensure_directory

def check_and_adjust_aspect_ratio(input_path, temp_output_dir):
    """
    Check if video has 1920x1080 resolution and rotate it to 1080x1920 if needed.
    
    Args:
        input_path (str): Path to the input video file.
        temp_output_dir (str): Directory to store the rotated video.
    
    Returns:
        str: Path to the processed (or original) video file.
    """
    print(f"[Aspect Ratio] Checking video: {input_path}")
    
    # Ensure temp directory exists
    ensure_directory(temp_output_dir)
    print(f"[Aspect Ratio] Temporary directory ensured: {temp_output_dir}")
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[Aspect Ratio] Error: Cannot open video file {input_path}")
        return None
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[Aspect Ratio] Video dimensions: {frame_width}x{frame_height}, FPS: {fps}")
    
    # Check if video needs rotation (1920x1080)
    if frame_width == 1920 and frame_height == 1080:
        print("[Aspect Ratio] Video is 1920x1080, rotating to 1080x1920")
        # Define output path for rotated video
        video_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(temp_output_dir, f"{video_name}_rotated.mp4")
        
        # Define output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_height, frame_width))
        
        print("[Aspect Ratio] Processing frames for rotation...")
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Rotate frame 90 degrees clockwise
            rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            out.write(rotated_frame)
            frame_count += 1
        
        # Release resources
        cap.release()
        out.release()
        print(f"[Aspect Ratio] Saved rotated video as: {output_path}")
        return output_path
    else:
        # If no rotation needed, return original path
        cap.release()
        print(f"[Aspect Ratio] No adjustment needed for {input_path}")
        return input_path