#include <Eigen/Dense>

using namespace Eigen;

void cov_fun(double *mean_mat, int *sz, double *center, double* signal, Ref<MatrixXd> Sigma);

void fun_myMNF(FILE *test, double* img, double *RD_hsi, int *sz){
    int i, j, k;

    double *center_Z = (double *)malloc(sizeof(double) * sz[0] * sz[1] * sz[3]);
    double *mean_mat = (double *)malloc(sizeof(double) * sz[3]);
    double *D_above = (double *)malloc(sizeof(double) * sz[0] * sz[1] * sz[3]);
    double *D_right = (double *)malloc(sizeof(double) * sz[0] * sz[1] * sz[3]);
    double *D_mat = (double *)malloc(sizeof(double) * sz[0] * sz[1] * sz[3]);
    double *center_D_mat = (double *)malloc(sizeof(double) * sz[0] * sz[1] * sz[3]);
    double *project_H = (double *)malloc(sizeof(double) * sz[2] * sz[3]);

    MatrixXd Sigma_X(sz[3],sz[3]);
	MatrixXd Sigma_N(sz[3],sz[3]);
    MatrixXd eig(sz[3],sz[2]);

    memset(D_above, 0, sizeof(double) * sz[0] * sz[1] * sz[3]);
    memset(D_right, 0, sizeof(double) * sz[0] * sz[1] * sz[3]);

    cov_fun(mean_mat, sz, center_Z, img, Sigma_X);

    //S.2 estimate the covariance matrix of noise (with MAF)
    //TO-BE-ADDED

    cov_fun(mean_mat, sz, center_D_mat, D_mat, Sigma_N);

    SelfAdjointEigenSolver<MatrixXd> eigensolver((Sigma_N.inverse())*Sigma_X);
    if (eigensolver.info() != Success) abort();
    eig=(eigensolver.eigenvectors())(Eigen::all, sz[2]);

    std::copy(eig.data(), eig.data() + eig.size(), project_H);

    for (i = 0; i < sz[2]; i++) {
		for (j = 0; j < sz[0] * sz[1]; j++) {
			for (k = 0; k < sz[3]; k++) {
                 RD_hsi[i*sz[0] * sz[1]+j] += project_H[i * sz[0] * sz[1] + k] * img[j * sz[0]*sz[1] + k];
			}
        }
    }

    free(center_Z);
    free(mean_mat);
    free(D_above);
    free(D_right);
    free(D_mat);
    free(center_D_mat);
    free(project_H);
}

void cov_fun(double *mean_mat, int *sz, double *center, double* signal, Ref<MatrixXd> Sigma){
    int i, j, k;

    Sigma.setZero();
    memset(mean_mat, 0, sizeof(double) * sz[3]);

    for (i = 0; i < sz[3]; i++) {
        for (j = 0; j < sz[0]*sz[1]; j++) {
            mean_mat[i] += signal[i*sz[0]*sz[1]+j];
        }
        mean_mat[i] /= sz[0]*sz[1];
    }

    for (i = 0; i < sz[3]; i++) {
	    for (j = 0; j < sz[0]*sz[1]; j++) {
            center[i*sz[0]*sz[1]+j] = signal[i*sz[0]*sz[1]+j]-mean_mat[i];
        }
    }

    for (i = 0; i < sz[3]; i++) {
		for (j = 0; j < sz[3]; j++) {
			for (k = 0; k < sz[0] * sz[1]; k++) {
                Sigma(i,j) += center[i * sz[0] * sz[1] + k] * center[j * sz[0]*sz[1] + k]; //center*center'
			}
            Sigma(i,j) *= (1/(double)((sz[0] * sz[1])-1));
        }
    }
}