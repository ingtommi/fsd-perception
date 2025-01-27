<div align="center">

# FSD Perception

## YOLO-based End-to-End Monocular Perception Pipeline for the Formula Student Driverless (FSD) Competition :racing_car:

</div>

This work is the result of my master's thesis in Electronics Engineering at UnivPM. You can read it [here]()!

It is a gift to Polimarche Racing Team :racing_car:. I would love to see further developments!

## Components

- NVIDIA Jetson Orin Nano Developer Kit
- IDS uEye UI-3160CP-C-HQ Rev. 2.1

## Usage

```
git clone https://github.com/ingtommi/fsd-perception
cd fsd-perception
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
