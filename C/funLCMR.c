#include "funLCMR.h"

void fun_LCMR_all(double *RD_hsi, int wnd_sz, int K, int* sz, double* lcmrfea_all) {
	int scale = (int)floor(wnd_sz / 2);
	int id = (int)ceil(wnd_sz * wnd_sz / 2);
	int i, j, k, ii, jj;

	double* RD_ex = (double*)malloc(sizeof(double) * (sz[1]+(2 * scale))*(sz[0]+(2 * scale)) * sz[2]);	
	double* tt_RD_DAT = (double*)malloc(sizeof(double) * (2 * scale +1) * (2 * scale +1) * sz[2]);
	double* cor = (double*)malloc(sizeof(double) * (2 * scale + 1) * (2 * scale + 1));
	double* sli_id = (double*)malloc(sizeof(double) * (2 * scale + 1) * (2 * scale + 1));
	double* tmp_mat = (double*)malloc(sizeof(double) * sz[2] * K);
	double* mean_mat = (double*)malloc(sizeof(double) * sz[2]);
	double* subFeatures = (double*)malloc(sizeof(double) * sz[2] * sz[2]);

	padArray(sz, scale, RD_ex, RD_hsi);

	for (i = 0; i < sz[0]; i++) { 
		for (j = 0; j < sz[1]; j++) { 

			for (k = 0; k < sz[2]; k++) {
				for (ii = i; ii <= i + 2*scale; ii++) {
					for (jj = j; jj <= j + 2*scale; jj++) {
						tt_RD_DAT[k * (2 * scale + 1) * (2 * scale + 1) + (jj-j)*(2*scale + 1)+(ii-i)] = RD_ex[k* (sz[0] + (2 * scale)) * (sz[1] + (2 * scale)) + ii* (sz[0] + (2 * scale)) + jj];
					}
				}
			}
			
			corCalc(sz, scale, tt_RD_DAT, cor, sli_id, id);
			quickSort(sli_id, cor, 0, (2 * scale + 1) * (2 * scale + 1)-1);
		 	centeredMat(sz, K, scale, tmp_mat, tt_RD_DAT, sli_id, mean_mat);
		
			allSamplesGeneration(sz, K, tmp_mat, subFeatures);
			
			//LOGM TO BE ADDED
			
			memcpy(&lcmrfea_all[(i*sz[1]+j)*sz[2]*sz[2]+j], subFeatures, sz[2] * sz[2]*sizeof(double));
		}
	}

	free(tt_RD_DAT);
	free(cor);
	free(sli_id);
	free(tmp_mat);
	free(mean_mat);
	free(subFeatures);
	free(RD_ex);
}

void padArray(int *sz, int scale, double *RD_ex, double *RD_hsi){
	int i, j, k;
	int row, col;
	int inc_row, inc_col;
	int val_row, val_col;
	
	for (k = 0; k < sz[2]; k++) {
		row = scale, col = scale, inc_row = -1, inc_col = -1;
		
		for (i = 0; i < (sz[0] + (2 * scale)); i++) {
			val_row = row + (1 == inc_row) - 1;
			row += inc_row;
			if (0 == row % sz[0]) {
				inc_row *= -1;
			}

			for (j = 0; j < (sz[1] + (2 * scale)); j++) {
				val_col = col + (1 == inc_col) - 1;
				col += inc_col;
				if (0 == col % sz[1]) {
					inc_col *= -1;
				}
				RD_ex[k * (sz[0] + (2 * scale))* (sz[1] + (2 * scale)) + i* (sz[0] + (2 * scale)) + j] = RD_hsi[k*sz[0]*sz[1]+val_row *sz[0] + val_col];
			}
			col = scale;
		}
	}
}

void corCalc(int *sz, int scale, double *tt_RD_DAT, double *cor, double *sli_id, int id){
	int ii, jj, k;
	double* norm_temp = (double*)malloc(sizeof(double) * (2 * scale + 1)* (2 * scale + 1));
	double* norm_block_2d = (double*)malloc(sizeof(double) * (2 * scale + 1) * (2 * scale + 1) * sz[2]);
	
	memset(norm_temp, 0, sizeof(double) * (2 * scale + 1) * (2 * scale + 1));
			
	for (ii = 0; ii < sz[2]; ii++) {
		for (jj = 0; jj < (2 * scale + 1) * (2 * scale + 1); jj++) {
			norm_temp[jj] += pow(tt_RD_DAT[ii * (2 * scale + 1) * (2 * scale + 1) + jj], 2);
		}
	}

	for (ii = 0; ii < sz[2]; ii++) {
		for (jj = 0; jj < (2 * scale + 1) * (2 * scale + 1); jj++) {
			norm_block_2d[ii*(2 * scale + 1) * (2 * scale + 1)+jj] = tt_RD_DAT[ii * (2 * scale + 1) * (2 * scale + 1) + jj]/sqrt(norm_temp[jj]);
		}
	}		

	memset(cor, 0, sizeof(double) * (2 * scale + 1) * (2 * scale + 1));

	for (jj = 0; jj < (2 * scale + 1) * (2 * scale + 1); jj++) {
		for (k = 0; k < sz[2]; k++) {
			cor[jj] += norm_block_2d[k * (2 * scale + 1) * (2 * scale + 1)+id] * norm_block_2d[k * (2 * scale + 1) * (2 * scale + 1) + jj];
			sli_id[jj]=jj;
		}
	}
	
	free(norm_temp);
	free(norm_block_2d);
}

void centeredMat(int *sz, int K, int scale, double* tmp_mat, double *tt_RD_DAT, double *sli_id, double *mean_mat){
	int k, jj;
	
	for (k = 0; k < sz[2]; k++) {
		for (jj = 0; jj < K; jj++) {
			tmp_mat[k*K+jj] = tt_RD_DAT[k * (2 * scale + 1) * (2 * scale + 1)+(int)sli_id[jj]];
		}
	}
	
	scale_func(tmp_mat, sz, K);
			
	memset(mean_mat, 0, sizeof(double) * sz[2]);

	for(k=0; k<sz[2]; k++){
		for (jj = 0; jj < K; jj++) {
			mean_mat[k] += tmp_mat[k*K+jj];
		}
		mean_mat[k] /= K;
				
		for (jj = 0; jj < K; jj++) {
			tmp_mat[k*K+jj]= tmp_mat[k*K+jj]-mean_mat[k];
		}
	}
}

void allSamplesGeneration(int *sz, int K, double *tmp_mat, double *subFeatures){
	const double tol = 1e-3;
	int ii, jj, k;
	
	memset(subFeatures, 0, sizeof(double) * sz[2] * sz[2]);
						
	for (ii = 0; ii < sz[2]; ii++) {
		for (jj = 0; jj < sz[2]; jj++) {
			for (k = 0; k < K; k++) {
				subFeatures[ii * sz[2] + jj] += tmp_mat[ii * K + k] * tmp_mat[jj * K + k];
			}
			subFeatures[ii * sz[2] + jj] /= (K-1);
		}
	}

	double matTrace = trace(subFeatures, sz[2]);
		
	for(k=0; k<sz[2]; k++){
		for (jj = 0; jj < sz[2]; jj++) {
			if(k==jj){
				subFeatures[k*sz[2]+jj] += tol*matTrace;
			}
		}
	}

}

void scale_func(double *data, int *sz, int K){
	int i, j=0;
	
	double* min = (double*)malloc(sizeof(double) * K);
	double* max = (double*)malloc(sizeof(double) * K);
	
	for(i=0; i<K; i++){
		min[i]=data[j*K+i];
		max[i]=data[j*K+i];
		
		for(j=0; j<sz[2]; j++){
			if(data[j*K+i]>max[i]){
				max[i]=data[j*K+i];
			}
			
			if(data[j*K+i]<min[i]){
				min[i]=data[j*K+i];
			}
		}
	}
	
	for(i=0; i<sz[2]; i++){
		for(j=0; j<K; j++){
			data[i*K+j] = (data[i*K+j]-min[j])/(max[j]-min[j]);
		}
	}
	
	free(min);
	free(max);
}

double trace(double *squaredMatrix, int dim){
	int k;
	double trace = 0;
			
	for(k=0; k<dim; k++){
		trace += squaredMatrix[k*dim+k];
	}
	return trace;
}

void quickSort(double *sli_id, double *arr, int low, int high){
	if (low < high){
		int pi = partition(sli_id, arr, low, high);
		quickSort(sli_id, arr, low, pi - 1);
		quickSort(sli_id, arr, pi + 1, high);
	}
}

int partition (double *sli_id, double *arr, int low, int high){
	double pivot = arr[high];
	int i = (low - 1);
	
	for (int j = low; j <= high- 1; j++){
		if (arr[j] >= pivot){	
			i++;
			swap(&arr[i], &arr[j]);
			swap(&sli_id[i], &sli_id[j]);
		}
	}
	swap(&arr[i + 1], &arr[high]);
	swap(&sli_id[i + 1], &sli_id[high]);
	
	return (i + 1);
}

void swap(double* a, double* b){
	double t = *a;
	*a = *b;
	*b = t;
}
