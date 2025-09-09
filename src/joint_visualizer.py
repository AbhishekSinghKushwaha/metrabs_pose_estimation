import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from utils.file_utils import ensure_directory
import os

def visualize_unfiltered_joints_z(excel_path):
    """
    Visualize unfiltered Z coordinates of left and right ankle and foot joints
    from an Excel file containing 3D joint coordinates.

    Parameters:
    - excel_path: Path to the Excel file containing 3D joint coordinates.
    """
    print(f"[Joint Visualizer] Loading Excel file: {excel_path}")
    try:
        df = pd.read_excel(excel_path)
        print(f"[Joint Visualizer] Loaded Excel file with {len(df)} rows")
    except FileNotFoundError:
        print(f"[Joint Visualizer] Excel file {excel_path} not found")
        return
    
    # Filter data for PersonID == 0 (assuming main person)
    df = df[df['PersonID'] == 0]
    print(f"[Joint Visualizer] Filtered data for PersonID == 0, rows: {len(df)}")
    if df.empty:
        print("[Joint Visualizer] No data found for PersonID == 0")
        return
    
    # Define joints to visualize
    joints = ['Left_Ankle_7', 'Right_Ankle_8', 'Left_Foot_10', 'Right_Foot_11']
    print(f"[Joint Visualizer] Selected joints for visualization: {joints}")
    
    # Extract Z coordinates for relevant joints
    joint_data = {joint: df[f'{joint}_Z'].values for joint in joints}
    print("[Joint Visualizer] Extracted Z coordinates for joints")
    
    # Create subplots for unfiltered data
    print("[Joint Visualizer] Creating subplots for unfiltered Z coordinates")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Unfiltered Left and Right Ankle Z
    frames = df['Frame'].values
    print(f"[Joint Visualizer] Plotting unfiltered Z coordinates for ankle joints, frames: {len(frames)}")
    ax1.plot(frames, joint_data['Left_Ankle_7'], label='Left Ankle Z', color='b', linestyle='-')
    ax1.plot(frames, joint_data['Right_Ankle_8'], label='Right Ankle Z', color='r', linestyle='--')
    ax1.set_title('Unfiltered Left and Right Ankle Z Coordinates')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Z Position (mm)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Unfiltered Left and Right Foot Z
    print("[Joint Visualizer] Plotting unfiltered Z coordinates for foot joints")
    ax2.plot(frames, joint_data['Left_Foot_10'], label='Left Foot Z', color='b', linestyle='-')
    ax2.plot(frames, joint_data['Right_Foot_11'], label='Right Foot Z', color='r', linestyle='--')
    ax2.set_title('Unfiltered Left and Right Foot Z Coordinates')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Z Position (mm)')
    ax2.legend()
    ax2.grid(True)
    
    # Adjust layout and save unfiltered plot
    plt.tight_layout()
    output_dir = os.path.dirname(excel_path)
    ensure_directory(output_dir)
    unfiltered_plot_path = os.path.join(output_dir, os.path.splitext(os.path.basename(excel_path))[0] + '_unfiltered_joints_z_plot.png')
    plt.savefig(unfiltered_plot_path, dpi=300, bbox_inches='tight')
    print(f"[Joint Visualizer] Saved unfiltered Z joints plot to {unfiltered_plot_path}")
    plt.close()

def visualize_filtered_joints_z(excel_path, frame_rate):
    """
    Visualize filtered Z coordinates of left and right ankle and foot joints
    from an Excel file, applying a Butterworth low-pass filter.

    Parameters:
    - excel_path: Path to the Excel file containing 3D joint coordinates.
    - frame_rate: Frame rate of the video (used for filtering).
    """
    print(f"[Joint Visualizer] Loading Excel file: {excel_path}")
    try:
        df = pd.read_excel(excel_path)
        print(f"[Joint Visualizer] Loaded Excel file with {len(df)} rows")
    except FileNotFoundError:
        print(f"[Joint Visualizer] Excel file {excel_path} not found")
        return
    
    # Filter data for PersonID == 0 (assuming main person)
    df = df[df['PersonID'] == 0]
    print(f"[Joint Visualizer] Filtered data for PersonID == 0, rows: {len(df)}")
    if df.empty:
        print("[Joint Visualizer] No data found for PersonID == 0")
        return
    
    # Define joints to visualize
    joints = ['Left_Ankle_7', 'Right_Ankle_8', 'Left_Foot_10', 'Right_Foot_11']
    print(f"[Joint Visualizer] Selected joints for visualization: {joints}")
    
    # Extract Z coordinates for relevant joints
    joint_data = {joint: df[f'{joint}_Z'].values for joint in joints}
    print("[Joint Visualizer] Extracted Z coordinates for joints")
    
    # Design Butterworth low-pass filter
    print("[Joint Visualizer] Designing Butterworth low-pass filter")
    cutoff_freq = 2.0  # Default cutoff frequency
    filter_order = 5   # Default filter order
    nyquist_freq = frame_rate / 2.0
    normalized_cutoff = cutoff_freq / nyquist_freq
    print(f"[Joint Visualizer] Filter parameters: cutoff_freq={cutoff_freq}, filter_order={filter_order}, frame_rate={frame_rate}")
    b, a = butter(filter_order, normalized_cutoff, btype='low', analog=False)
    print("[Joint Visualizer] Created Butterworth filter coefficients")
    
    # Apply filter to each joint's Z coordinates
    filtered_data = {joint: filtfilt(b, a, joint_data[joint]) for joint in joints}
    
    # Create subplots for filtered data
    print("[Joint Visualizer] Creating subplots for filtered Z coordinates")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Filtered Left and Right Ankle Z
    frames = df['Frame'].values
    print(f"[Joint Visualizer] Plotting filtered Z coordinates for ankle joints, frames: {len(frames)}")
    ax1.plot(frames, filtered_data['Left_Ankle_7'], label='Left Ankle Z', color='b', linestyle='-')
    ax1.plot(frames, filtered_data['Right_Ankle_8'], label='Right Ankle Z', color='r', linestyle='--')
    ax1.set_title('Filtered Left and Right Ankle Z Coordinates')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Z Position (mm)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Filtered Left and Right Foot Z
    print("[Joint Visualizer] Plotting filtered Z coordinates for foot joints")
    ax2.plot(frames, filtered_data['Left_Foot_10'], label='Left Foot Z', color='b', linestyle='-')
    ax2.plot(frames, filtered_data['Right_Foot_11'], label='Right Foot Z', color='r', linestyle='--')
    ax2.set_title('Filtered Left and Right Foot Z Coordinates')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Z Position (mm)')
    ax2.legend()
    ax2.grid(True)
    
    # Adjust layout and save filtered plot
    plt.tight_layout()
    output_dir = os.path.dirname(excel_path)
    ensure_directory(output_dir)
    filtered_plot_path = os.path.join(output_dir, os.path.splitext(os.path.basename(excel_path))[0] + '_filtered_joints_z_plot.png')
    plt.savefig(filtered_plot_path, dpi=300, bbox_inches='tight')
    print(f"[Joint Visualizer] Saved filtered Z joints plot to {filtered_plot_path}")
    plt.close()

def visualize_joints_z(excel_path, frame_rate):
    """
    Wrapper function to visualize both unfiltered and filtered Z coordinates
    of left and right ankle and foot joints.

    Parameters:
    - excel_path: Path to the Excel file containing 3D joint coordinates.
    - frame_rate: Frame rate of the video (used for filtered visualization).
    """
    print("[Joint Visualizer] Starting Z joints visualization")
    # Visualize unfiltered Z joints
    visualize_unfiltered_joints_z(excel_path)
    # Visualize filtered Z joints
    visualize_filtered_joints_z(excel_path, frame_rate)
    print("[Joint Visualizer] Completed ankle and foot Z joints visualization")