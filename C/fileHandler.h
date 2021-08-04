#pragma once

#include <stdio.h>
#include <stdlib.h>

void readHSI(FILE* f1, double* RD_HSI, const int* sz);
void readLabels(FILE* f2, int* labels, const int* sz);
void readlcmrFEA(FILE* f3, double* lcmrfea_all, const int* sz);
void savelcmrFEA(FILE* file, double* lcmrfea_all, const int* sz);
