#include "../eigen-lib/Eigen/Dense"
#include "../eigen-lib/Eigen/Eigenvalues"

using namespace Eigen;

void cov_fun(double* mean_mat, int* sz, double* center, double* signal, Ref<MatrixXd> Sigma);

void fun_myMNF(double* img, double* RD_hsi, int* sz){
    int i, j, k;

    double* center = (double*)malloc(sizeof(double) * sz[0] * sz[1] * sz[3]);
    double* mean_mat = (double*)malloc(sizeof(double) * sz[3]);
    double* D_above = (double*)malloc(sizeof(double) * sz[0] * sz[1] * sz[3]);
    double* D_right = (double*)malloc(sizeof(double) * sz[0] * sz[1] * sz[3]);
    double* D_mat = (double*)malloc(sizeof(double) * sz[0] * sz[1] * sz[3]);

    MatrixXd Sigma_X(sz[3],sz[3]);
	MatrixXd Sigma_N(sz[3],sz[3]);
    MatrixXd eig(sz[3],sz[2]);

    memset(D_above, 0, sizeof(double) * sz[0] * sz[1] * sz[3]);
    memset(D_right, 0, sizeof(double) * sz[0] * sz[1] * sz[3]);

    cov_fun(mean_mat, sz, center, img, Sigma_X);

    //S.2 estimate the covariance matrix of noise (with MAF)
    for(k=0; k<sz[3]; k++){ 
        for(i=1; i<sz[0]; i++){ 
            for(j=0; j<sz[1]; j++){
                D_above[(k * sz[0] * sz[1]) + i * sz[1] + j] = img[(k * sz[0] * sz[1]) + j * sz[0] + i] - img[(k * sz[0] * sz[1]) + j * sz[0] + (i-1)];
            }
        }
    }

    for(k=0; k<sz[3]; k++){
        for(i=0; i<sz[0]; i++){
            for(j=0; j<(sz[1] - 1); j++){
                D_right[(k * sz[0] * sz[1]) + i * sz[1] + j] = img[(k * sz[0] * sz[1]) + j * sz[0] + i] - img[(k * sz[0] * sz[1]) + (j+1) * sz[0] + i];
            }
        }
    }

    for(k=0; k<sz[3]; k++){
        for(i=0; i<sz[0]; i++){
            for(j=0; j<sz[1]; j++){
                D_mat[(k * sz[0] * sz[1]) + j * sz[0] + i] = (D_right[(k * sz[0] * sz[1])+i * sz[1] + j] + D_above[(k * sz[0] * sz[1]) + i * sz[1] + j]) / 2;
            }
        }
    }

    cov_fun(mean_mat, sz, center, D_mat, Sigma_N);

    EigenSolver<MatrixXd> eigensolver((Sigma_N.fullPivLu().inverse()) * Sigma_X);
    //if (eigensolver.info() != Success) abort();
    
    eig=((eigensolver.eigenvectors()).real()).transpose();

    for (i = 0; i < sz[2]; i++) {
		for (j = 0; j < sz[0] * sz[1]; j++) {
			for (k = 0; k < sz[3]; k++) {
                RD_hsi[i * sz[0] * sz[1] + j] += eig(i, k) * img[k * sz[0] * sz[1] + j];
			}
        }
    }

    free(D_mat);
    free(center);
    free(mean_mat);
    free(D_above);
    free(D_right);
}

void cov_fun(double* mean_mat, int* sz, double* center, double* signal, Ref<MatrixXd> Sigma){
    int i, j, k;

    Sigma.setZero();
    memset(mean_mat, 0, sizeof(double) * sz[3]);

    for (i = 0; i < sz[3]; i++) {
        for (j = 0; j < sz[0]*sz[1]; j++) {
            mean_mat[i] += signal[i * sz[0] * sz[1] + j];
        }
        mean_mat[i] /= sz[0] * sz[1];
    }

    for (i = 0; i < sz[3]; i++) {
	    for (j = 0; j < sz[0] * sz[1]; j++) {
            center[i * sz[0] * sz[1] + j] = signal[i * sz[0] * sz[1] + j] - mean_mat[i];
        }
    }

    for (i = 0; i < sz[3]; i++) {
		for (j = 0; j < sz[3]; j++) {
			for (k = 0; k < sz[0] * sz[1]; k++) {
                Sigma(i,j) += center[i * sz[0] * sz[1] + k] * center[j * sz[0]*sz[1] + k]; //center*center'
			}
            Sigma(i,j) *= (1/(double)((sz[0] * sz[1]) - 1));
        }
    }
}