# Metrabs 3D Pose Estimation

## Overview

This project processes videos to extract 3D human poses using the Metrabs model, adjusts aspect ratios (1920x1080 to 1080x1920 if needed), generates 2D/3D visualizations, visualizes unfiltered and filtered Z coordinates of ankle and foot joints, and converts results to an Azure Kinect formatted CSV.

## What You Need

- Python 3.10.16

- Enough disk space for the Metrabs model (~1.5 GB) and output files

## Installation

1. Clone or download the repository.

2. Navigate to the project directory: 
`cd metrabs_pose_estimation`

3. Create a virtual environment (recommended):
`python -m venv venv`
`source venv/bin/activate  # On Windows: venv\Scripts\activate`

4. Install dependencies:
`pip install -r requirements.txt`

## Usage

1. Place your input video in data/videos/.

2. Update main.py with the correct video filename and output paths if needed.

3. Optionally, modify utils/config.py to change the model type or joint mappings.

4. Run the pipeline:
`main.py`

5. Outputs will be saved in the `output/` subdirectories.


## Troubleshooting

- Video not found: Verify the video path in main.py.

- Model download failure: Check internet connection and disk space.

- Missing Excel file for visualization: Ensure the video is processed successfully to generate the Excel file.