#include "header.h"

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
			fscanf(f2, "%d", &labels[j * sz[1] + i]);
		}
	}
}

void fun_LCMR_all(double *RD_hsi, int wnd_sz, int K, int* sz, double* lcmrfea_all) {
	double tol = 1e-3;
	int scale = (int)floor(wnd_sz / 2);
	int id = (int)ceil(wnd_sz * wnd_sz / 2);
	int i, j, k, ii, jj;

	double* RD_ex = (double*)malloc(sizeof(double) * (sz[1]+(2 * scale))*(sz[0]+(2 * scale)) * sz[2]);

	//PADARRAY
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
	//END PADARRAY
	
	double* tt_RD_DAT = (double*)malloc(sizeof(double) * (2 * scale +1) * (2 * scale +1) * sz[2]);
	double* norm_temp = (double*)malloc(sizeof(double) * (2 * scale + 1)* (2 * scale + 1));
	double* norm_block_2d = (double*)malloc(sizeof(double) * (2 * scale + 1) * (2 * scale + 1) * sz[2]);
	double* cor = (double*)malloc(sizeof(double) * (2 * scale + 1) * (2 * scale + 1));
	double* sli_id = (double*)malloc(sizeof(double) * (2 * scale + 1) * (2 * scale + 1));
	double* tmp_mat = (double*)malloc(sizeof(double) * sz[2] * K);
	double* mean_mat = (double*)malloc(sizeof(double) * sz[2]);
	double* tmp = (double*)malloc(sizeof(double) * sz[2] * sz[2]);


	for (i = 0; i < sz[0]; i++) { 
		for (j = 0; j < sz[1]; j++) { 

			for (k = 0; k < sz[2]; k++) {
				for (ii = i; ii <= i + 2*scale; ii++) {
					for (jj = j; jj <= j + 2*scale; jj++) {
						tt_RD_DAT[k * (2 * scale + 1) * (2 * scale + 1) + (jj-j)*(2*scale + 1)+(ii-i)] = RD_ex[k* (sz[0] + (2 * scale)) * (sz[1] + (2 * scale)) + ii* (sz[0] + (2 * scale)) + jj];
					}
				}
			}
			
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

			quickSort(sli_id, cor, 0, (2 * scale + 1) * (2 * scale + 1)-1);
			
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
			}
			
			for(k=0; k<sz[2]; k++){
				for (jj = 0; jj < K; jj++) {
					tmp_mat[k*K+jj]= tmp_mat[k*K+jj]-mean_mat[k];
				}
			}
		
			memset(tmp, 0, sizeof(double) * sz[2] * sz[2]);
						
			for (ii = 0; ii < sz[2]; ii++) {
				for (jj = 0; jj < sz[2]; jj++) {
					for (k = 0; k < K; k++) {
						tmp[ii * sz[2] + jj] += tmp_mat[ii * K + k] * tmp_mat[jj * K + k];
					}
					tmp[ii * sz[2] + jj] /= (K-1);
				}
			}

			double trace = 0;
			
			for(k=0; k<sz[2]; k++){
				trace += tmp[k*sz[2]+k];
			}
			
			for(k=0; k<sz[2]; k++){
				for (jj = 0; jj < sz[2]; jj++) {
					if(k==jj){
						tmp[k*sz[2]+jj] += tol*trace;
					}
				}
			}
			//LOGM TO BE ADDED
			memcpy(&lcmrfea_all[(i*sz[1]+j)*sz[2]*sz[2]+j], tmp, sz[2] * sz[2]*sizeof(double));
		}
	}

	free(tt_RD_DAT);
	free(norm_temp);
	free(norm_block_2d);
	free(cor);
	free(sli_id);
	free(tmp_mat);
	free(mean_mat);
	free(tmp);
	free(RD_ex);
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
void logmTrain(struct svm_node **nod, const double* array1, const double* array2, int m, int n, int p) {
	int i, j, k;
	double sum;

	for (i = 0; i < m; i++) {
		for (j = 0; j < p; j++) {
			sum=0;
			for (k = 0; k < n; k++) {
				sum += array1[i * n + k] * array2[j * n + k];
			}
			
			nod[i][j+1].index=j+1;	
			nod[i][j+1].value = sum;					
		}
		
		nod[i][0].index=0;
		nod[i][0].value=i+1;
		
		nod[i][m+1].index=-1;
		nod[i][m+1].value=0;
	}
}

void logmTest(struct svm_node **nod, const double* array1, const double* array2, int m, int n, int p) {
	int i, j, k;
	double sum;

	for (i = 0; i < m; i++) {
		for (j = 0; j < p; j++) {
			sum=0;
			for (k = 0; k < n; k++) {
				sum += array1[i * n + k] * array2[j * n + k];
			}
			
			nod[j][i+1].index=i+1;	
			nod[j][i+1].value = sum;
			
			nod[j][0].index=0;
			nod[j][0].value=j+1;
		
			nod[j][m+1].index=-1;
			nod[j][m+1].value=0;
		}
	}
}

void generateSample(int* labels, int no_classes, int* sz, int* train_id, double*train_label, int* test_id, int* test_label, int* test_size){
	int ii, i, size, len=0;

	double* tmp_label = (double*)malloc(sizeof(double) * no_classes*sz[0]*sz[1]);
	int* tmp_id = (int*)malloc(sizeof(int) * no_classes*sz[0]*sz[1]);


	for (ii = 1; ii <= no_classes; ii++) {
		for (i = 0; i < (sz[0] * sz[1]); i++) {
			if (labels[i] == ii) {
				tmp_id[test_size[0]] = i;
				tmp_label[test_size[0]] = ii;
				test_size[0]++;
			}
		}
	}

	int* W_class_index = (int*)malloc(sizeof(int) * test_size[0]);

	for (ii = 1; ii <= no_classes; ii++) {
		size = 0;

		for (i = 0; i < test_size[0]; i++) {
			if (tmp_label[i] == ii) {
				W_class_index[size] = i;
				size++;
			}
		}

		shuffle(W_class_index, size);

		for (i = 0; i < TRAIN_NUMBER; i++) {
			train_id[(ii-1)*TRAIN_NUMBER+i] = tmp_id[W_class_index[i]];
			train_label[(ii-1)*TRAIN_NUMBER+i] = tmp_label[W_class_index[i]];
			tmp_label[W_class_index[i]]=HUGE_VAL;
		}
	}
	
	for(ii=0; ii<test_size[0]; ii++){
		if(tmp_label[ii] != HUGE_VAL){
			test_id[len]=tmp_id[ii];
			test_label[len]=tmp_label[ii];
			len++;
		}
	}
	test_size[0]=len;

	free(tmp_id);
	free(tmp_label);
	free(W_class_index);
}

void shuffle(int* array, int n){
	srand((unsigned)time(NULL));
	if (n > 1){
		int i;
		for (i = 0; i < n - 1; i++){
			int j = i + rand() / (RAND_MAX / (n - i) + 1);
			int t = array[j];
			array[j] = array[i];
			array[i] = t;
		}
	}
}

double mean(const double* array, int length) {
	int i;
	double sum = 0;
	
	for (i = 0; i < length; i++) {
		sum += array[i];
	}

	return sum / length;
}

void calcError(double *OA, double *class_accuracy, const int *test_label, const double *predicted_label, const int* test_id, int size, int no_classes, double* kappa){
	int i, j;

	int* nrPixelsPerClass = (int*)malloc(sizeof(int) * no_classes);
	int* errorMatrix = (int*)malloc(sizeof(int) * no_classes*no_classes);

	memset(nrPixelsPerClass, 0, sizeof(int) * no_classes);
	memset(errorMatrix, 0, sizeof(int) * no_classes*no_classes);

	errorMatrixGeneration(no_classes, test_label, predicted_label, nrPixelsPerClass, errorMatrix, test_id, size);
	
	KappaClassAccuracy(no_classes, errorMatrix, class_accuracy, kappa, nrPixelsPerClass);
	overallAccuracy(size, test_label, predicted_label, test_id, OA);

	free(nrPixelsPerClass);
	free(errorMatrix);
}

void KappaClassAccuracy(int no_classes, int *errorMatrix, double *class_accuracy, double *kappa, int *nrPixelsPerClass){
	int i, j;
	int col_val, row_val, tot_sum = 0, diag_sum = 0, prod_mat = 0;

	for (i = 0; i < no_classes; i++) {
		col_val = 0; row_val = 0;
		for (j = 0; j < no_classes; j++) {
			tot_sum += errorMatrix[i * no_classes + j];
			row_val += errorMatrix[i * no_classes + j];
			col_val += errorMatrix[j * no_classes + i];
		}
		prod_mat += col_val * row_val;

		diag_sum += errorMatrix[i * no_classes + i];
		class_accuracy[i] = errorMatrix[i * no_classes + i] / (nrPixelsPerClass[i] + EPS);
	}

	kappa[0] = (double)((tot_sum * diag_sum) - prod_mat)/(pow(tot_sum,2) - prod_mat);
}

void overallAccuracy(int size, const int* test_label, const double *predicted_label, const int *test_id,  double *OA){
	int i;
	
	for(i=0; i<size; i++){
		if ((test_label[i]-1) == (int)(predicted_label[test_id[i]]-1)) {
			OA[0]++;
		}
	}

	OA[0] /= (size+EPS);
}

void errorMatrixGeneration(int no_classes, const int *test_label, const double* predicted_label, int *nrPixelsPerClass, int* errorMatrix, const int* test_id, int size){
	int ii, i, j, len_seg, len_true;
	int* tmp_true = (int*)malloc(sizeof(int) * size);
	int* tmp_seg = (int*)malloc(sizeof(int) * size);
	
	for (ii = 0; ii < no_classes; ii++) {
		len_true=0;
		for (i = 0; i < size; i++) {
			if (test_label[i]-1 == ii) {
				tmp_true[len_true] = i;
				len_true++;
			}
		}
		nrPixelsPerClass[ii] = len_true;

		for (i = 0; i < no_classes; i++) {
			len_seg = 0;
			
			for (j = 0; j < size; j++) {
				if (predicted_label[test_id[j]]-1 == i) {
					tmp_seg[len_seg] = j;
					len_seg++;
				}
			}
			errorMatrix[ii*no_classes+i]=intersection(tmp_true, tmp_seg, len_true, len_seg, size);
		}
	}

	free(tmp_true);
	free(tmp_seg);
}

int intersection(int *array1, int* array2, int len1, int len2, int size){
	int j, k, t, len=0, flag;
	int* tmp = (int*)malloc(sizeof(int) * size);
	
	for (j = 0; j < len1; j++) {
		for (k = 0; k < len2; k++) {
			if (array1[j] == array2[k]) {
				flag=0;
				for(t=0; t<len; t++){
					if(tmp[t]==array1[j]){
						flag=1;
					}
				}
				if(flag!=1){
					tmp[t]=array1[j];
					len++;
				}
			}
		}
	}
	free(tmp);
	return len;
}

void svmSetParameter(struct svm_parameter *param, int no_fea){
	param->svm_type = C_SVC;
	param->kernel_type = PRECOMPUTED;
	param->degree = 3;
	param->coef0 = 0;
	param->gamma = 1/(double)(no_fea+1);
	
	param->eps = 0.01;
	param->C = 1;
	param->cache_size = 100;
	param->shrinking = 1;
	param->probability = 0;
	
	param->nr_weight = 0;
	param->weight = NULL;
	param->weight_label = NULL;
}

void svmSetProblem(struct svm_problem *prob, double *labels, int no_labels){
	int i;
	
	prob->l = no_labels;
	prob->y= labels;
	
	prob->x = (struct svm_node**)malloc((no_labels)* sizeof(struct svm_node*));
	for(i=0; i<no_labels; i++){
		prob->x[i]=(struct svm_node*)malloc((sizeof(struct svm_node)*(no_labels+2)));
	}
}

