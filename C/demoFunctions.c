#include "demo.h"

void generateSample(int* labels, int no_classes, int* sz, int* train_id, double* train_label, int* test_id, int* test_label, int* test_size){
	int ii, i, size, len=0;

	double* tmp_label = (double*)malloc(sizeof(double) * no_classes * sz[0] * sz[1]);
	int* tmp_id = (int*)malloc(sizeof(int) * no_classes * sz[0] * sz[1]);
	
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
	int* indices = (int*)malloc(sizeof(int) * no_classes * TRAIN_NUMBER);
	
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
			train_id[(ii-1) * TRAIN_NUMBER + i] = tmp_id[W_class_index[i]];
			train_label[(ii-1) * TRAIN_NUMBER + i] = tmp_label[W_class_index[i]];
			indices[(ii-1) * TRAIN_NUMBER + i]=W_class_index[i];
		}
	}
	
	int flag=0;
	
	for(ii=0; ii<test_size[0]; ii++){ //rimuove dai dati di test i dati da usare per il train
		for(i=0; i<no_classes * TRAIN_NUMBER; i++){
			if(ii==indices[i]){
				flag=1;
			}
		}
		
		if(flag!=1){
			test_id[len]=tmp_id[ii];
			test_label[len]=tmp_label[ii];
			len++;
		}
		flag=0;
	}
	
	test_size[0]=len;

	free(tmp_id);
	free(tmp_label);
	free(W_class_index);
	free(indices);
}

void logmTrain(struct svm_node** nod, const double* array1, const double* array2, int m, int n, int p) {
	int i, j, k;
	double sum;

	for (i = 0; i < m; i++) {
		nod[i][0].index=0;
		nod[i][0].value=i+1;
		
		for (j = 0; j < p; j++) {
			sum=0;
			for (k = 0; k < n; k++) {
				sum += array1[i * n + k] * array2[j * n + k];
			}
			
			nod[i][j+1].index=j+1;	
			nod[i][j+1].value = sum;
		}
					
		nod[i][m+1].index=-1;
		nod[i][m+1].value=0;			
	}
}

void logmTest(struct svm_node** nod, const double* array1, const double* array2, int m, int n, int p) {
	int i, j, k;
	double sum;

	for (i = 0; i < m; i++) {
		for (j = 0; j < p; j++) {
			nod[j][0].index=0;
			nod[j][0].value=j+1;
			
			sum=0;
			for (k = 0; k < n; k++) {
				sum += array1[i * n + k] * array2[j * n + k];
			}
			
			nod[j][i+1].index=i+1;	
			nod[j][i+1].value = sum;
			
			nod[j][m+1].index=-1;
			nod[j][m+1].value=0;
		}
	}
}

void calcError(double* OA, double* class_accuracy, const int* test_label, const double* predicted_label, const int* test_id, int size, int no_classes, double* kappa){
	int i, j;

	int* nrPixelsPerClass = (int*)malloc(sizeof(int) * no_classes);
	int* errorMatrix = (int*)malloc(sizeof(int) * no_classes * no_classes);

	memset(nrPixelsPerClass, 0, sizeof(int) * no_classes);
	memset(errorMatrix, 0, sizeof(int) * no_classes * no_classes);

	errorMatrixGeneration(no_classes, test_label, predicted_label, nrPixelsPerClass, errorMatrix, test_id, size);
	
	KappaClassAccuracy(no_classes, errorMatrix, class_accuracy, kappa, nrPixelsPerClass);
	overallAccuracy(size, test_label, predicted_label, test_id, OA);

	free(nrPixelsPerClass);
	free(errorMatrix);
}

void errorMatrixGeneration(int no_classes, const int* test_label, const double* predicted_label, int* nrPixelsPerClass, int* errorMatrix, const int* test_id, int size){
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
			errorMatrix[ii * no_classes + i]=intersection(tmp_true, tmp_seg, len_true, len_seg, size);
		}
	}

	free(tmp_true);
	free(tmp_seg);
}

void KappaClassAccuracy(int no_classes, int* errorMatrix, double* class_accuracy, double* kappa, int* nrPixelsPerClass){
	int i, j;
	double col_val, row_val, tot_sum = 0, diag_sum = 0, prod_mat = 0;

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

	kappa[0] = ((tot_sum * diag_sum) - prod_mat)/(pow(tot_sum,2) - prod_mat);
}

void overallAccuracy(int size, const int* test_label, const double* predicted_label, const int* test_id,  double* OA){
	int i;
	
	for(i=0; i<size; i++){
		if ((test_label[i]-1) == (int)(predicted_label[test_id[i]]-1)) {
			OA[0]++;
		}
	}

	OA[0] /= (size+EPS);
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

int intersection(int* array1, int* array2, int len1, int len2, int size){
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

void svmSetParameter(struct svm_parameter* param, int no_fea){
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

void svmSetProblem(struct svm_problem* prob, double* labels, int no_labels){
	int i;

	prob->l = no_labels;
	prob->y= labels;
	
	prob->x = (struct svm_node**)malloc((no_labels) * sizeof(struct svm_node*));
	for(i=0; i<no_labels; i++){
		prob->x[i]=(struct svm_node*)malloc((sizeof(struct svm_node) * (no_labels+2)));
	}
}

