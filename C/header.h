#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include "BitmapWriter.h"
#include "svm.h"

#define TRAIN_NUMBER 5
#define N_IT 1
#define EPS 2.2204e-16

//FILE 
void readHSI(FILE* f1, double* RD_HSI, const int* sz);
void readLabels(FILE* f2, int* labels, const int* sz);
void readlcmrFEA(FILE* f3, double* lcmrfea_all, const int* sz);
void savelcmrFEA(FILE* file, double* lcmrfea_all, const int* sz);

//DATA
void fun_LCMR_all(double* RD_hsi, int wnd_sz, int K, int* sz, double* lcmrfea_all);
void generateSample(int* labels, int no_classes, int* sz, int* train_id, double* train_label, int* test_id, int* test_label, int* test_size);

//COMPUTATION
void logmTrain(struct svm_node **nod, const double* array1, const double* array2, int m, int n, int p);
void logmTest(struct svm_node **nod, const double* array1, const double* array2, int m, int n, int p);
void calcError(double *OA, double *class_accuracy, const int *test_label, const double *predicted_label, const int* test_id, int size, int no_classes, double* kappa);
void errorMatrixGeneration(int no_classes, const int *test_label, const double* predicted_label, int *nrPixelsPerClass, int* errorMatrix, const int* test_id, int size);
void KappaClassAccuracy(int no_classes, int *errorMatrix, double *class_accuracy, double *kappa, int *nrPixelsPerClass);
void overallAccuracy(int size, const int* test_label, const double *predicted_label, const int *test_id,  double *OA);
void scale_func(double *data, int *sz, int K);

//SVM
void svmSetParameter(struct svm_parameter *param, int no_fea);
void svmSetProblem(struct svm_problem *prob, double *labels, int no_labels);

//EXTRA FUNCTIONS
void shuffle(int* array, int n);
int intersection(int *array1, int* array2, int len1, int len2, int size);
double mean(const double* array, int length);
void quickSort(double *sli_id, double *arr, int low, int high);
int partition (double *sli_id, double *arr, int low, int high);
void swap(double* a, double* b);
