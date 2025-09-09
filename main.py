import os
import tensorflow as tf
from src.aspect_ratio import check_and_adjust_aspect_ratio
from src.model_loader import download_model
from src.video_processor import process_single_video
from src.csv_converter import convert_excel_to_kinect_csv
from src.joint_visualizer import visualize_joints_z
from utils.file_utils import ensure_directory
from utils.config import MODEL_TYPE

def main():
    """
    Main function to run the Metrabs pose estimation.
    """
    print("[Main] Starting Metrabs Pose Estimation")
    
    # Define paths
    video_path = 'data/videos/iphone16_30fps_short.mov'
    parent_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    excel_output_dir = os.path.join(parent_dir, '../output/Joints', video_name)
    output_2d_dir = os.path.join(parent_dir, '../output/2D_Images', video_name)
    output_3d_dir = os.path.join(parent_dir, '../output/3D_Images', video_name)
    output_comparison_dir = os.path.join(parent_dir, '../output/Comparison_Images', video_name)
    output_videos_dir = os.path.join(parent_dir, '../output/Videos', video_name)
    output_csv_dir = os.path.join(parent_dir, '../output/formatted_csv_files', video_name)
    excel_path = os.path.join(excel_output_dir, f'{video_name}_3D_coordinates.xlsx')
    temp_output_dir = os.path.join(parent_dir, 'temp_videos')
    
    print(f"[Main] Video path: {video_path}")
    
    # Adjust aspect ratio
    processed_video_path = check_and_adjust_aspect_ratio(video_path, temp_output_dir)
    if not processed_video_path:
        print("[Main] Failed to process video aspect ratio")
        return
    
    # Load model
    print(f"[Main] Loading Metrabs model: {MODEL_TYPE}")
    model_path = download_model()
    model = tf.saved_model.load(model_path)
    print("[Main] Model loaded successfully")
    
    # Process video
    fps = process_single_video(
        processed_video_path, excel_path, output_2d_dir, output_3d_dir,
        output_comparison_dir, output_videos_dir, output_csv_dir, model
    )
    
    if fps is not None:
        # Visualize Z joints
        print("[Main] Visualizing Z joints for ankle and foot")
        visualize_joints_z(excel_path, fps)
        
        # Convert to CSV
        convert_excel_to_kinect_csv(excel_path, output_csv_dir, fps)
        print("[Main] Pipeline completed successfully")
    else:
        print("[Main] Pipeline failed due to video processing error")

if __name__ == "__main__":
    main()