# KIBO Robot Programming Challenge 5th - Training Image Generator 📸

This module contains Python scripts for the KRPC Computer Vision project.

## Files

- [`__init__.py`](krpc_cv_py/__init__.py): This is the initialization file for the `krpc_cv_py` module.
- [`createTrainingData.py`](krpc_cv_py/createTrainingData.py): This script is used to create training data.
- [`undistortNavCamImage.py`](krpc_cv_py/undistortNavCamImage.py): This script is used to undistort images from the Navigation Camera.

## Data Outputs

The `data_outputs` directory contains labeled data for different levels. Each level has its own directory and a corresponding `labels.json` file.\
It is generated upon running the script.

## Images

The `images` directory contains the images from NavCam.

## Item Inputs

The `item_inputs` directory contains the images of lost item candidates.

## Running the Scripts

To run the scripts, use the following command inside a Poetry shell:

```sh
python <script_name.py>
```

Replace `<script_name.py>` with the name of the script you want to run.

## Testing

Tests for the module are located in the `tests` directory.\
No tests have been implemented

## Dependencies

The project dependencies are listed in the [`pyproject.toml`](pyproject.toml) file.

## License

This project is licensed under the terms of the GNU GPLv3