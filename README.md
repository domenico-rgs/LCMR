# LCMR
Porting an algorithm from MATLAB to C for feature extraction from hyperspectral images using local covariance matrix representation with consequent parallelization using [Cuda 10.2](https://developer.nvidia.com/cuda-10.2-download-archive) and OpenMP.

__Hyperspectral Remote Sensing Scenes (sample datasets)__: http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes

### Used libraries

* [LibSVM 3.25](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)
* [Eigen 3.4](https://eigen.tuxfamily.org/index.php?title=Main_Page)
* [cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html)

## Parameters and data from Matlab
_param.txt_
```
n°_classes
windows_size
spectral_reflectance_bands
width height excepted_output_dimensionality final_no_bands
color_map [india, uni, center, dc]
```

_HSI_
```Matlab
hsi=reshape(permute(indian_pines_corrected, [1,2,3]),[],size(indian_pines_corrected,3))';
dlmwrite('hsi.txt',hsi,'delimiter',' ');
```

_Groundtruth_
```Matlab
labels=reshape(permute(labels, [1,3,2]),[],size(labels,2));
dlmwrite('labels.txt', labels,'delimiter',' ');
```

## Serial execution
```console
cd C
g++ -O2 -I ../eigen-lib/ demo.cpp demoFunctions.cpp fileHandler.cpp funLCMR.cpp funMyMNF.cpp BitmapWriter.cpp svm.cpp -lm
./a.out param.txt hsi.txt labels.txt
```

## Parallel execution
### Windows
Compile the source code in LCMR Parallel through Visual Studio (release mode) and then run as follow:
```console
cd "\LCMR Parallel\x64\Release"
"LCMR Parallel.exe" param.txt hsi.txt labels.txt
```
