#include "fileHandler.h"

void readHSI(FILE* f1, double* RD_HSI, const int *sz) {
	int i, j;

	for (i = 0; i < sz[2]; i++) {
		for (j = 0; j < sz[0]*sz[1]; j++) {
			fscanf(f1, "%lf", &RD_HSI[i*sz[0]*sz[1]+j]);
		}
	}
}

void readlcmrFEA(FILE* f3, double* lcmrfea_all, const int* sz) {
	int i, j;

	for (i = 0; i < sz[0] * sz[1]; i++) {
		for (j = 0; j < sz[2] * sz[2]; j++) {
			fscanf(f3, "%lf", &lcmrfea_all[i * sz[2] * sz[2] + j]);
		}
	}
}

void savelcmrFEA(FILE* file, double* lcmrfea_all, const int* sz) {
	int i, j;
	
	for (i = 0; i < sz[0] * sz[1]; i++) {
		for (j = 0; j < sz[2]*sz[2]; j++) {
			fprintf(file, "%lf ", lcmrfea_all[i * sz[2]*sz[2] + j]);
		}
		fprintf(file, "\n");
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
