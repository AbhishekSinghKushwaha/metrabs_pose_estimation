import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import numpy as np
from utils.file_utils import ensure_directory

def visualize_frame(im, poses3d, poses2d, edges, frame_number, output_2d_dir, output_3d_dir, output_comparison_dir, original_width, original_height):
    """
    Visualize 2D and 3D poses on the frame and save images to specified directories.
    
    Args:
        im (numpy.ndarray): Input frame in BGR format.
        poses3d (numpy.ndarray): 3D pose coordinates.
        poses2d (numpy.ndarray): 2D pose coordinates.
        edges (numpy.ndarray): Joint edges for skeleton visualization.
        frame_number (int): Frame number for naming output files.
        output_2d_dir (str): Directory to save 2D visualization images.
        output_3d_dir (str): Directory to save 3D visualization images.
        output_comparison_dir (str): Directory to save comparison images.
        original_width (int): Original video width.
        original_height (int): Original video height.
    """    
    # Ensure output directories exist
    ensure_directory(output_2d_dir)
    ensure_directory(output_3d_dir)
    ensure_directory(output_comparison_dir)
    
    # Calculate figure size in inches to match original video resolution at 100 DPI
    dpi = 100
    fig_width = original_width / dpi
    fig_height = original_height / dpi
    
    # Create figures for 2D, 3D, and comparison visualizations
    fig_2d = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    fig_3d = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    fig_comparison = plt.figure(figsize=(fig_width * 2, fig_height), dpi=dpi)
    
    # 2D visualization
    image_ax = fig_2d.add_subplot(1, 1, 1)
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct colors
    image_ax.imshow(im_rgb)
    image_ax.axis('off')  # Hide axes
    
    # 3D visualization
    pose_ax = fig_3d.add_subplot(1, 1, 1, projection='3d')
    pose_ax.view_init(5, -85)
    pose_ax.set_xlim3d(-1500, 1500)
    pose_ax.set_zlim3d(-1500, 1500)
    pose_ax.set_ylim3d(0, 3000)
    pose_ax.set_xlabel('X')
    pose_ax.set_ylabel('Z')
    pose_ax.set_zlabel('Y')
    
    # Comparison visualization: 2D on left, 3D on right
    comp_ax_2d = fig_comparison.add_subplot(1, 2, 1)
    comp_ax_2d.imshow(im_rgb)
    comp_ax_2d.axis('off')
    comp_ax_3d = fig_comparison.add_subplot(1, 2, 2, projection='3d')
    comp_ax_3d.view_init(5, -85)
    comp_ax_3d.set_xlim3d(-1500, 1500)
    comp_ax_3d.set_zlim3d(-1500, 1500)
    comp_ax_3d.set_ylim3d(0, 3000)
    comp_ax_3d.set_xlabel('X')
    comp_ax_3d.set_ylabel('Z')
    comp_ax_3d.set_zlabel('Y')
    comp_ax_2d.set_title('2D Pose')
    comp_ax_3d.set_title('3D Pose')
    
    # Adjust 3D poses for visualization
    poses3d_vis = poses3d.copy()
    poses3d_vis[..., 1], poses3d_vis[..., 2] = poses3d_vis[..., 2], -poses3d_vis[..., 1]
    
    # Generate colors for edges and joints
    colors = cm.tab20(np.linspace(0, 1, len(edges)))
    
    # Plot 2D and 3D joints and edges
    for pose3d, pose2d in zip(poses3d_vis, poses2d):
        for idx, (i_start, i_end) in enumerate(edges):
            color = colors[idx]
            image_ax.plot(*zip(pose2d[i_start], pose2d[i_end]), marker='o', markersize=6, color=color, linewidth=3)
            comp_ax_2d.plot(*zip(pose2d[i_start], pose2d[i_end]), marker='o', markersize=6, color=color, linewidth=3)
            pose_ax.plot(*zip(pose3d[i_start], pose3d[i_end]), marker='o', markersize=6, color=color, linewidth=3)
            comp_ax_3d.plot(*zip(pose3d[i_start], pose3d[i_end]), marker='o', markersize=6, color=color, linewidth=3)
        for joint_idx in range(pose2d.shape[0]):
            joint_color = next((colors[idx] for idx, (i_start, i_end) in enumerate(edges) 
                                if i_start == joint_idx or i_end == joint_idx), colors[0])
            image_ax.scatter(pose2d[joint_idx, 0], pose2d[joint_idx, 1], s=5, color=joint_color)
            comp_ax_2d.scatter(pose2d[joint_idx, 0], pose2d[joint_idx, 1], s=5, color=joint_color)
            pose_ax.scatter(pose3d[joint_idx, 0], pose3d[joint_idx, 1], pose3d[joint_idx, 2], s=5, color=joint_color)
            comp_ax_3d.scatter(pose3d[joint_idx, 0], pose3d[joint_idx, 1], pose3d[joint_idx, 2], s=5, color=joint_color)
    
    # Save 2D image
    output_2d_path = os.path.join(output_2d_dir, f'frame_{frame_number:06d}.png')
    fig_2d.tight_layout()
    fig_2d.savefig(output_2d_path, bbox_inches='tight', dpi=dpi, facecolor='white', edgecolor='none')
    plt.close(fig_2d)
    
    # Save 3D image
    output_3d_path = os.path.join(output_3d_dir, f'frame_{frame_number:06d}.png')
    fig_3d.tight_layout()
    fig_3d.savefig(output_3d_path, bbox_inches='tight', dpi=dpi, facecolor='white', edgecolor='none')
    plt.close(fig_3d)
    
    # Save comparison image
    output_comp_path = os.path.join(output_comparison_dir, f'frame_{frame_number:06d}.png')
    fig_comparison.tight_layout()
    fig_comparison.savefig(output_comp_path, bbox_inches='tight', dpi=dpi, facecolor='white', edgecolor='none')
    plt.close(fig_comparison)