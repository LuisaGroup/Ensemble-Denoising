# Ensemble Denoising
Source Code for SIGGRAPH Asia 2021 Paper ***Ensemble Denoising for Monte Carlo Renderings***.

[Paper](https://github.com/Mike-Leo-Smith/EnsembleDenoising/tree/master/paper) | [Code](https://github.com/Mike-Leo-Smith/EnsembleDenoising) | Talk Slides (Comming Soon)

## Dependencies
- A C++17 conformant compiler for building the code
- [`CMake`](http://cmake.org) for configuring the project
- [`Eigen3`](https://eigen.tuxfamily.org) for linear algebra data structures and operations
- [`OpenCV`](https://opencv.org) for image IO and basic operators
- [`cxxopts`](https://github.com/jarro2783/cxxopts) (included as a git submodule) for parsing CLI options
- [`fbstab`](https://github.com/dliaomcp/fbstab) (optional, included as a git submodule) for linearly constrained quadratic programming as the baseline solver

## Compiling the Project
First, please make sure that Eigen3 and OpenCV are properly install in your system and can be found by CMake.

To clone the repository:
```bash
git clone --recursive https://github.com/Mike-Leo-Smith/EnsembleDenoising.git
```

Then, use CMake to configure the build:
```bash
cmake -S <project-dir> -B <build-dir> -D CMAKE_BUILD_TYPE=Release
```
Note: to help CMake find Eigen3 and OpenCV, you may need to provide `-D Eigen3_DIR=...` and `-D OpenCV_DIR=...` as well.
Also, you can use the `-G <generator>` option to specify your favorite generator, and `-D CMAKE_CXX_COMPILER=<...>` to choose the C++ compiler.

After the project is properly configured, use the following command to build the program:
```bash
cmake --build <build-dir>
```
And the executable `EnsembleDenosing` should appear in `<build-dir>` if successfully built.

## Preparing Input Data
The input data should be organized in the following structure:
```
<input-dir>
  │
  ├─ color.exr, colorA.exr, colorB.exr, albedo.exr, normal.exr
  │
  ├─ <base-denoiser-1>
  │    │
  │    └─ <base-denoiser-1>.exr <base-denoiser-1>A.exr <base-denoiser-1>B.exr
  │
  ├─ <base-denoiser-2>
  │    │
  │    └─ <base-denoiser-2>.exr <base-denoiser-2>A.exr <base-denoiser-2>B.exr
  ...
```
For example, if `denoising-data` is the input directory and two denoisers
[`oidn`](https://www.openimagedenoise.org) and
[`nfor`](https://cs.dartmouth.edu/wjarosz/publications/bitterli16nonlinearly.html) are used,
the directory structure should be
```
denoising-data
  │
  ├─ color.exr, colorA.exr, colorB.exr, albedo.exr, normal.exr
  │
  ├─ oidn
  │    │
  │    └─ oidn.exr oidnA.exr oidnB.exr
  │
  └─ nfor
       │
       └─ nfor.exr nforA.exr nforB.exr
```
You can use any scene, renderer, and denoiser to generate these data. In the paper, we use the
[`TungstenRenderer`](https://github.com/tunabrain/tungsten) for rendering.

## Running the Program
To print the help message:
```bash
./EnsembleDenoising -h
```
To perform the ensemble denoising:
```
./EnsembleDenoising -i <input-dir> -o <output-dir> -m <base-denoiser-1>[,<base-denoiser-2>[...]]
```
