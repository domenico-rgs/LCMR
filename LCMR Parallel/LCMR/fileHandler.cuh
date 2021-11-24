#pragma once

#include <stdio.h>
#include <stdlib.h>

void readHSI(FILE* f1, double* img, const int* sz);
void readLabels(FILE* f2, int* labels, const int* sz);