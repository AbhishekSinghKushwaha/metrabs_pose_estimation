import cv2
import tensorflow as tf
import pandas as pd
import os
import shutil
from tqdm import tqdm
from src.visualization import visualize_frame
from utils.file_utils import ensure_directory
from utils.config import SMPL24_JOINT_NAMES

def create_video_from_frames(image_dir, output_video_path, fps, original_width, original_height):
    """
    Create a video from saved images in a directory.
    
    Args:
        image_dir (str): Directory containing PNG images.
        output_video_path (str): Path to save the output video.
        fps (float): Frames per second for the output video.
        original_width (int): Width of the output video.
        original_height (int): Height of the output video.
    """
    print(f"[Video Processor] Creating video from: {image_dir}")
    
    # Get list of image files sorted by name
    images = sorted([img for img in os.listdir(image_dir) if img.endswith('.png')])
    if not images:
        print(f"[Video Processor] No images found in {image_dir}")
        return
    
    # Define codec and VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (original_width, original_height))
    
    # Write images to video
    print(f"[Video Processor] Writing {len(images)} images to: {output_video_path}")
    for image in images:
        frame = cv2.imread(os.path.join(image_dir, image))
        resized_frame = cv2.resize(frame, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
        video_writer.write(resized_frame)
    
    video_writer.release()
    print(f"[Video Processor] Video saved to: {output_video_path}")

def process_single_video(video_path, output_excel_path, output_2d_dir, output_3d_dir, output_comparison_dir, output_videos_dir, output_csv_dir, model):
    """
    Process a video to extract 3D poses, generate visualizations, and save results.
    
    Args:
        video_path (str): Path to the input video.
        output_excel_path (str): Path to save Excel file with 3D coordinates.
        output_2d_dir (str): Directory for 2D visualization images.
        output_3d_dir (str): Directory for 3D visualization images.
        output_comparison_dir (str): Directory for comparison images.
        output_videos_dir (str): Directory for output videos.
        output_csv_dir (str): Directory for Azure Kinect CSV file.
        model: Loaded Metrabs model.
    
    Returns:
        float: Frame rate of the video.
    """
    print(f"[Video Processor] Processing video: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"[Video Processor] Error: Video file {video_path} does not exist")
        return None
    
    # Ensure output directories exist
    for directory in [output_2d_dir, output_3d_dir, output_comparison_dir, output_videos_dir, output_csv_dir]:
        ensure_directory(directory)
    
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Video Processor] Video FPS: {fps}, Width: {original_width}, Height: {original_height}, Total Frames: {total_frames}")
    
    # Process frames with progress bar
    data = []
    frame_number = 0
    print("[Video Processor] Starting frame processing...")
    
    with tqdm(total=total_frames, desc="Processing Frames", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(f"[Video Processor] Processed {frame_number} frames")
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_number += 1
            image = tf.convert_to_tensor(rgb_frame, dtype=tf.uint8)
            pred = model.detect_poses(image, max_detections=1, skeleton='smpl_24')
            
            poses3d = pred['poses3d'].numpy()
            poses2d = pred['poses2d'].numpy()
            edges = model.per_skeleton_joint_edges['smpl_24'].numpy()
            
            visualize_frame(frame, poses3d, poses2d, edges, frame_number, output_2d_dir, output_3d_dir, output_comparison_dir, original_width, original_height)
            
            for pose_idx, pose3d in enumerate(poses3d):
                pose_data = pose3d.flatten()
                row = [frame_number, pose_idx] + pose_data.tolist()
                data.append(row)
            
            pbar.update(1)
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Create videos
    video_base_name = os.path.splitext(video_name)[0]
    video_2d_path = os.path.join(output_videos_dir, f'{video_base_name}_2D.mp4')
    video_3d_path = os.path.join(output_videos_dir, f'{video_base_name}_3D.mp4')
    create_video_from_frames(output_2d_dir, video_2d_path, fps, original_width, original_height)
    create_video_from_frames(output_3d_dir, video_3d_path, fps, original_width, original_height)
    
    # Copy processed video
    original_video_copy_path = os.path.join(output_videos_dir, video_name)
    shutil.copy2(video_path, original_video_copy_path)
    print(f"[Video Processor] Copied video to: {original_video_copy_path}")
    
    # Create column names using SMPL24 joint names
    columns = ['Frame', 'PersonID'] + [f'{joint}_{coord}' for joint in SMPL24_JOINT_NAMES for coord in ['X', 'Y', 'Z']]
    
    # Save to Excel
    print(f"[Video Processor] Saving Excel with {len(data)} rows")
    df = pd.DataFrame(data, columns=columns)
    ensure_directory(os.path.dirname(output_excel_path))
    df.to_excel(output_excel_path, index=False)
    print(f"[Video Processor] Saved Excel to: {output_excel_path}")
    
    print(f"[Video Processor] Outputs saved: 2D Images ({output_2d_dir}), 3D Images ({output_3d_dir}), Comparison ({output_comparison_dir}), Videos ({output_videos_dir})")
    return fps