# LCMR
Porting an algorithm from MATLAB to C for feature extraction from hyperspectral images using local covariance matrix representation with consequent parallelization using CUDA/OpenMP.

## Execution
```console
g++ -O2 demo.c demoFunctions.c fileHandler.c funLCMR.c BitmapWriter.c svm.cpp -lm
./a.out param.txt MNF.txt labels.txt lcmrfea_all.txt
```
