# LCMR
Porting an algorithm from MATLAB to C for feature extraction from hyperspectral images using local covariance matrix representation with consequent parallelization using CUDA/OpenMP.

### Used libraries

* [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)
* [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) (logm implementation)

## Serial execution
```console
cd C
g++ -O2 -I ./eigen/ demo.c demoFunctions.c fileHandler.c funLCMR.cpp BitmapWriter.c svm.cpp -lm
./a.out param.txt MNF.txt labels.txt india lcmrfea_all.txt
```
