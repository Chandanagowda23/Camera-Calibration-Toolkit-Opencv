# Camera Calibration Toolkit

This Python project provides functions for camera calibration, including finding checkerboard corners, calculating intrinsic parameters, and determining extrinsic parameters.

## Files

- **Geometry.py**: Python script containing camera calibration functions.
- **checkboard.png**: Sample image of a checkerboard used for calibration.
- **Chessboard23.png**: Output image with detected corners.
- **result_task.json**: Result file for camera calibration.
- **README.md**: Documentation for the project.

## Overview

This toolkit is designed for camera calibration, a crucial step in computer vision applications. The calibration process involves detecting checkerboard corners in images, finding the intrinsic parameters of the camera, and determining its extrinsic parameters.

## Usage

1. **Run Camera Calibration:**
   - Execute `Geometry.py` to run camera calibration functions.
   - Ensure the required dependencies (NumPy, OpenCV, SciPy) are installed.

2. **Checkerboard Image:**
   - Use the provided `checkboard.png` as a sample image for calibration.

3. **Detected Corners:**
   - The script detects 32 checkerboard corners and generates an output image (`Chessboard23.png`) with drawn corners.

4. **Intrinsic Parameters:**
   - Intrinsic parameters, including focal length and principal point, are calculated and saved in `result_task.json`.

5. **Extrinsic Parameters:**
   - Extrinsic parameters, such as rotation matrix and translation vector, are also calculated and included in `result_task.json`.

## Functionality

### 1. Find Checkerboard Corners

Use the following function to find the pixel coordinates of 32 checkerboard corners:

```python
from Geometry import find_corner_img_coord

# Example Usage
image = cv2.imread('checkboard.png')
corners = find_corner_img_coord(image)

world_coordinates = find_corner_world_coord(corners)
fx, fy, cx, cy = find_intrinsic(corners, world_coordinates)
R, T = find_extrinsic(corners, world_coordinates)

