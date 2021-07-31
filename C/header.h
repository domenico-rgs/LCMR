#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include "BitmapWriter.h"
//#include "svm.h"

#define TRAIN_NUMBER 5
#define N_IT 1

void readHSI(FILE* f1, double* RD_HSI, int* sz);
void readLabels(FILE* f2, int* labels, int* sz);
void fun_LCMR_all(FILE *file,double* RD_hsi, int wnd_sz, int K, int* sz, double* lcmrfea_all);
void readlcmrFEA(FILE* f3, double* lcmrfea_all, int* sz);
void savelcmrFEA(FILE* file, double* lcmrfea_all, int* sz);
void logmkernel(const double* array1, const double* array2, double* result, int m, int n, int p);
void generateSample(const int* labels, const int* train_number, int no_classes, const int* sz, int* train_id, int* train_label, int* test_id, int* test_label, int* test_size);
void shuffle(int* array, int n);
double mean(const double* OA);
void calcError(double* OA, double* class_accuracy, const int* test_label, const double* predicted_label, const int* test_id, int n_it, int size, int no_classes, const int* sz, double* kappa);
void scale_func(double *data, int *sz, int K);

void quickSort(double *sli_id, double *arr, int low, int high);
int partition (double *sli_id, double *arr, int low, int high);
void swap(double* a, double* b);
