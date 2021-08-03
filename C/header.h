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

//FILE 
void readHSI(FILE *test,FILE* f1, double* RD_HSI, const int* sz);
void readLabels(FILE *test, FILE* f2, int* labels, const int* sz);
void readlcmrFEA(FILE* test,FILE* f3, double* lcmrfea_all, const int* sz);
void savelcmrFEA(FILE* file, double* lcmrfea_all, const int* sz);

//DATA
void fun_LCMR_all(double* RD_hsi, int wnd_sz, int K, int* sz, double* lcmrfea_all);
void generateSample(FILE *test,int* labels, int no_classes, int* sz, int* train_id, double* train_label, int* test_id, int* test_label, int* test_size);

//COMPUTATION
void logmTrain(FILE *test, struct svm_node **nod, const double* array1, const double* array2, int m, int n, int p);
void logmTest(FILE *test, struct svm_node **nod, const double* array1, const double* array2, int m, int n, int p);
void calcError(double* OA, double* class_accuracy, const int* test_label, const double* predicted_label, const int* test_id, int n_it, int size, int no_classes, const int* sz, double* kappa);
void scale_func(double *data, int *sz, int K);

//SVM
void svmSetParameter(struct svm_parameter *param, int no_fea);
void svmSetProblem(struct svm_problem *prob, double *labels, int no_labels);

//EXTRA FUNCTIONS
void shuffle(int* array, int n);
double mean(const double* array, int length);
void quickSort(double *sli_id, double *arr, int low, int high);
int partition (double *sli_id, double *arr, int low, int high);
void swap(double* a, double* b);
