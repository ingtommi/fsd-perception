<div align="center">

# FSD Perception

## YOLO-based End-to-End Monocular Perception Pipeline for the Formula Student Driverless (FSD) Competition :racing_car:

</div>

This work is the result of my master's thesis in Electronics Engineering at UnivPM. You can read it [here]()!

It is a gift to Polimarche Racing Team :racing_car:. I would love to see further developments!

## Idea

To estimate depth with a **single monocular camera** (a choice made to reduce cost and complexity...), the idea is to leverage a priori information about cone geometry and use it to invert the pinhole camera equations, as shown in the figure below. Object detection is done before and provides the height of the cones in the image plane.

NOTE: This is only possible with a calibrated camera!

![monocular](https://github.com/user-attachments/assets/af48dc6d-e10a-44ed-b9c3-18b2d8377628)

## Components

- IDS uEye UI-3160CP-C-HQ Rev. 2.1
- NVIDIA Jetson Orin Nano Developer Kit

## Setup and Usage

First, download the code by typing:

```
git clone https://github.com/ingtommi/fsd-perception
cd fsd-perception
```

At this point, make sure you have installed all the required libraries and then configure the application by writing in the YAML config files under `config/`. 

NOTE: You have to generate model *engine* (TensorRT) files by converting *onnx* files under `models/` with `scripts/build_trt.py`. Also, check that `cameraCalib.yaml` contains the parameters corresponding to the current camera. If you need to perfom camera calibration, take a look at `scripts/calibrate_camera.py`!

Then, build the system with:

```
mkdir build
cd build
```

For **normal** operation (i.e., running perception), continue with:

```
cmake ..
make
./perception
```

If instead the **benchmarking** mode is required, compile with `cmake .. -DBENCHMARK=ON` for **overall** benchmarking (i.e., end-to-end latency and fps) and `cmake .. -DBENCHMARK=ON -DDETAIL=ON` for **detailed** benchmarking (i.e., latency of single sub-modules).