#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

void fun_LCMR_all(double* RD_hsi, int wnd_sz, int K, int* sz, double* lcmrfea_all);
void corCalc(int* sz, int scale, double* tt_RD_DAT, double* cor, double* sli_id, int id, double* norm_temp, double* norm_block_2d);
void centeredMat(int* sz, int K, int scale, double* tmp_mat, double* tt_RD_DAT, double* sli_id, double* mean_mat, double* min, double* max);
void allSamplesGeneration(int* sz, int K, double* tmp_mat, double* lcmrfea_all, int i, int j);

//EXTRA FUNCTIONS
void padArray(int* sz, int scale, double* RD_ex, double* RD_hsi);
void scale_func(double *data, int *sz, int K, double* min, double* max);
double trace(double* squaredMatrix, int dim);
void quickSort(double* sli_id, double* arr, int low, int high);
int partition (double* sli_id, double* arr, int low, int high);
void swap(double* a, double* b);
