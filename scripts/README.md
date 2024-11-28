This folders contains **Python scripts** for solving project-related tasks.

### *build_trt.py*

Build serialized engine files for **TensorRT** inference from **onnx** models, using **FP32**, **FP16**, or **INT8** precision. For INT8 quantization, this script performs **calibration** using either a previously generated calibration **cache** file or new **images**.

Launch `python3 build_trt.py --help` to check required arguments.

*NOTE: Calibration is tailored to each model, so re-use a calibration cache file only when building from the same onnx model!*

### *get_zscore_values.py*

Calculate **mean** and **standard deviation** for each channel (B,G,R) of the dataset to enable **Z-Score Normalization** during inference. 

By default this pipeline uses **IMAGENET** values because this is what is done in [ultralytics/yolov5](https://github.com/ultralytics/yolov5/blob/master/utils/augmentations.py#L58). Currently, [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/augment.py) (YOLOv8 and YOLOv11) and [yolov10/ultralytics](https://github.com/THU-MIG/yolov10/blob/main/ultralytics/data/augment.py) do not apply this normalization step.

*NOTE: Set `normalize:True` in `../config/yolo.yaml` to enable Z-Score Normalization.*

### *process_log.py*

Process logs generated in detailed benchmarking mode to display **average pre-processing**, **inference**, and **post-processing latency**.

By default logs are written in `../build/log.txt`, so there is no need to modify the input path. Remember, however, to **delete this file before each subsequent benchmark** because it is opened in **append** mode.

*NOTE: Set `numIts` to the correct value to skip inferences performed during warm-up.*