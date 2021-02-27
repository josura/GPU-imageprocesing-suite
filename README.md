# Image Processing suite in GPU
This project was built to create a suite of techniques in image processing exploiting the embarassingly parallel procedures used in image processing.

## Requirements (OpenCL)
To build this project for opencl the following dependencies must be satisfied:
> - OpenCL (platform included)
> - Make
> - gcc 

## Requirements (CUDA, not implemented yet)
To build this project for opencl the following dependencied must be satisfied:
> - CUDA(with a supporting GPU of course)
> - OpenCV (for image loading and manipulation)
> - Cmake
> - Make

## Requirements (GUI)
To build the GUI QT5 needs to be installed and working.

# Usage
To download the project and run GUI application do the following steps

1. git clone https://github.com/josura/GPU-imageprocesing-suite.git
2. chmod u+x buildall.sh run.sh
3. ./buildall.sh
4. ./run.sh
