This folders contains **Python scripts** for solving project-related tasks.

### - *map.py*

Create a **bird's-eye view map** showing the position of detected cones in the camera reference system.

### - *build_trt.py*

Build serialized engine files for **TensorRT** inference from **onnx** models, using **FP32**, **FP16**, or **INT8** precision. For INT8 quantization, this script performs **calibration** using either a previously generated calibration **cache** file or new **images**.

Launch `python3 build_trt.py --help` to check required arguments.

*NOTE: Calibration is tailored to each model, so re-use a calibration cache file only when building from the same onnx model!*

### - *calibrate_camera.py*

Calibrates a camera using the **OpenCV** [approach](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html). The script expects the required images (**the more the better**) to be at `../media/calib/*`.

*NOTE: Set `IMGSZ` and `CHECKERBOARD` to the correct values before launching the script.*

### - *check_distortion.py*

Helps **assessing the distortion** by showing both the original and undistorted images. Undistortion is performed using the **OpenCV** `cv::undistort()` method.

By default, this script expects a valid calibration file at `../config/cameraCalib.yaml`.

### - *process_log_yolo.py* & *process_log_depth.py*

Process logs generated in detailed benchmarking mode to display **average pre-processing**, **inference**, **post-processing**, as well as **depth estimation latency**.

By default, logs are written in `../build/log_{}.txt`, so there is no need to modify the input paths. Remember, however, to **delete these file before each subsequent benchmark** because they are opened in **append** mode.

*NOTE: Set `numIts` to the correct value to skip inferences performed during warm-up.* 

### - *get_zscore_values.py*

Calculate **mean** and **standard deviation** for each channel (B,G,R) of the dataset to enable **Z-Score Normalization** during inference. 

By default, this pipeline uses **IMAGENET** values because this is what is done in [ultralytics/yolov5](https://github.com/ultralytics/yolov5/blob/master/utils/augmentations.py#L58). Currently, [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/augment.py) (YOLOv8 and YOLOv11) and [yolov10/ultralytics](https://github.com/THU-MIG/yolov10/blob/main/ultralytics/data/augment.py) do not apply this normalization step.

*NOTE: Set `normalize:True` in `../config/yolo.yaml` to enable this normalization.*