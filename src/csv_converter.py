import pandas as pd
import numpy as np
import os
from utils.file_utils import ensure_directory
from utils.config import JOINT_MAPPING

def convert_excel_to_kinect_csv(excel_path, output_csv_dir, frame_rate):
    """
    Convert Excel file with 3D coordinates to Azure Kinect formatted CSV.
    
    Args:
        excel_path (str): Path to the input Excel file.
        output_csv_dir (str): Directory to save the CSV file.
        frame_rate (float): Frame rate for timestamp calculation.
    """
    print(f"[CSV Converter] Converting Excel: {excel_path}")
    
    ensure_directory(output_csv_dir)
    output_file = os.path.join(output_csv_dir, os.path.splitext(os.path.basename(excel_path))[0] + "_kinect.csv")
    print(f"[CSV Converter] Output CSV path: {output_file}")
    
    # Load Excel
    print(f"[CSV Converter] Loading Excel file")
    df = pd.read_excel(excel_path)
    print(f"[CSV Converter] Excel contains {len(df)} rows")
    
    # Calculate timestamps
    num_frames = len(df)
    timestamps = np.linspace(0, num_frames / frame_rate, num_frames)
    print(f"[CSV Converter] Calculated timestamps with FPS: {frame_rate}")
    
    # Convert to Azure Kinect format
    output_data = []
    print("[CSV Converter] Processing rows...")
    for frame_idx, row in df.iterrows():
        timestamp = timestamps[frame_idx]
        person_id = row['PersonID']
        for joint, azure_idx in JOINT_MAPPING.items():
            pos_x = row[f'{joint}_X']
            pos_y = row[f'{joint}_Y']
            pos_z = row[f'{joint}_Z']
            output_data.append({
                'Timestamp': timestamp,
                'BodyID': person_id,
                'Joint_': azure_idx,
                'Position_x_': pos_x,
                'Position_y_': pos_y,
                'Position_z_': pos_z
            })
    
    # Save to CSV
    output_df = pd.DataFrame(output_data)
    print(f"[CSV Converter] Saving CSV with {len(output_data)} rows")
    output_df.to_csv(output_file, index=False)
    print(f"[CSV Converter] Saved CSV to: {output_file}")
    return output_file