#include "demo.h"

void fun_myMNF(FILE *test, int* img, double *RD_hsi, int d, int *sz){
    int i, j, k;

    double *center_Z = (double *)malloc(sizeof(double) * sz[0] * sz[1] * 200);
    double *mean_mat = (double *)malloc(sizeof(double) * 200);
    double *Sigma_X = (double *)malloc(sizeof(double) * 200 * 200);

	memset(mean_mat, 0, sizeof(double) * 200);
	memset(Sigma_X, 0, sizeof(double) * 200 * 200);

    for (i = 0; i < 200; i++) {
        for (j = 0; j < sz[0]*sz[1]; j++) {
            mean_mat[i] += img[i*sz[0]*sz[1]+j];
        }
        mean_mat[i] /= sz[0]*sz[1];
    }

    for (i = 0; i < 200; i++) {
	    for (j = 0; j < sz[0]*sz[1]; j++) {
            center_Z[i*sz[0]*sz[1]+j] = img[i*sz[0]*sz[1]+j]-mean_mat[i];
        }
    }

    //S.1 Calculate the covariance matrix of whole data
    for (i = 0; i < 200; i++) {
		for (j = 0; j < 200; j++) {
			for (k = 0; k < sz[0] * sz[1]; k++) {
                Sigma_X[i*200+j] += center_Z[i * sz[0] * sz[1] + k] * center_Z[j * sz[0]*sz[1] + k]; //center_Z*center_Z'
			}
            Sigma_X[i*200+j] *= (1/(double)((sz[0] * sz[1])-1));
        }
    }

    //S.2 estimate the covariance matrix of noise (with MAF)

    free(Sigma_X);
    free(mean_mat);
    free(center_Z);
}