#include "fileHandler.h"

void readHSI(FILE* f1, double* img, const int *sz) {
	int i, j;

	for (i = 0; i < sz[3]; i++) {
		for (j = 0; j < sz[0] * sz[1]; j++) {
			fscanf(f1, "%lf", &img[i*sz[0]*sz[1]+j]);
		}
	}
}

void readLabels(FILE* f2, int* labels, const int* sz) {
	int i, j;

	for (i = 0; i < sz[0]; i++) {
		for (j = 0; j < sz[1]; j++) {
			fscanf(f2, "%d", &labels[j * sz[0] + i]); //read "transpose" because matlab read by columns while in the following labels are read by rows
		}
	}
}
