# Compilation

### Linux:

```bash
mkdir build && cd build
cmake  ..
make
cd ..
```

### Windows:

Generate the project using CMake (Tested on vs2019)

# Execution

### Linux:

##### Reconstruction using calibration

```bash
build/3D_Reconstruct
```
##### Reconstruction using already rectified stereo images

```bash
build/NoCalibReconstruct
```

### Windows:

Execute either executable using your IDE.

