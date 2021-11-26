#include "demo.h"

void generateSample(int* labels, int no_classes, int* sz, int* train_id, double* train_label, int* test_id, int* test_label, int* test_size, double* tmp_label, int* tmp_id, int* indices){
	int ii, i, size, len = 0;

	for (ii = 1; ii <= no_classes; ii++) {
		for (i = 0; i < (sz[0] * sz[1]); i++) {
			if (labels[i] == ii) {
					tmp_id[test_size[0]++] = i;
					tmp_label[test_size[0]] = ii;
			}
		}
	}

	int* W_class_index = (int*)malloc(sizeof(int) * test_size[0]);

	for (ii = 1; ii <= no_classes; ii++) {
		size = 0;

		for (i = 0; i < test_size[0]; i++) {
			if (tmp_label[i] == ii) {
				W_class_index[size++] = i;
			}
		}

		shuffle(W_class_index, size);

		for (i = 0; i < TRAIN_NUMBER; i++) {
			train_id[(ii - 1) * TRAIN_NUMBER + i] = tmp_id[W_class_index[i]];
			train_label[(ii - 1) * TRAIN_NUMBER + i] = tmp_label[W_class_index[i]];
			indices[(ii - 1) * TRAIN_NUMBER + i] = W_class_index[i];
		}
	}

	for (ii = 0; ii < test_size[0]; ii++) { //Removes from the test data the ones to be used for the train
		for (i = 0; i < no_classes * TRAIN_NUMBER; i++) {
			if (ii == indices[i]) {
				break;
			}
		}

		if (i == no_classes * TRAIN_NUMBER) {
				test_id[len++] = tmp_id[ii];
				test_label[len] = tmp_label[ii];
		}
	}

	test_size[0] = len;

	free(W_class_index);
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

void calcError(double* OA, double* class_accuracy, const int* test_label, const double* predicted_label, const int* test_id, int size, int no_classes, double* kappa, int* nrPixelsPerClass, int* errorMatrix){
	memset(nrPixelsPerClass, 0, sizeof(int) * no_classes);
	memset(errorMatrix, 0, sizeof(int) * no_classes * no_classes);

	errorMatrixGeneration(no_classes, test_label, predicted_label, nrPixelsPerClass, errorMatrix, test_id, size);
	
	KappaClassAccuracy(no_classes, errorMatrix, class_accuracy, kappa, nrPixelsPerClass);
	overallAccuracy(size, test_label, predicted_label, test_id, OA);
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


int getMax(int* arr, int n){
	int mx = arr[0];
	for (int i = 1; i < n; i++)
		if (arr[i] > mx)
			mx = arr[i];
	return mx;
}

void countSort(int* arr, int* output, int n, int exp){
	int i, count[10] = { 0 };

	for (i = 0; i < n; i++)
		count[(arr[i] / exp) % 10]++;

	for (i = 1; i < 10; i++)
		count[i] += count[i - 1];

	for (i = n - 1; i >= 0; i--) {
		output[count[(arr[i] / exp) % 10] - 1] = arr[i];
		count[(arr[i] / exp) % 10]--;
	}

	for (i = 0; i < n; i++)
		arr[i] = output[i];
}

void radixsort(int* arr, int n){
	int m = getMax(arr, n);
	int* output = (int*)malloc(sizeof(int) * n);

	for (int exp = 1; m / exp > 0; exp *= 10) {
		countSort(arr, output, n, exp);
	}

	free(output);
}

int intersection(int* arr1, int* arr2, int len1, int len2, int size){
	int t, intersectC = 0;
	int* tmp = (int*)malloc(sizeof(int) * size);

	radixsort(arr1, len1);
	radixsort(arr2, len2);

	int i = 0, j = 0;
	while (i < len1 && j < len2) {
		if (arr1[i] < arr2[j])
			i++;
		else if (arr2[j] < arr1[i])
			j++;
		else /* if arr1[i] == arr2[j] */
		{
			for (t = 0; t < intersectC; t++) {
				if (tmp[t] == arr1[j]) {
					break;
				}
			}
			if (t == intersectC) {
				tmp[intersectC++] = arr1[j];
			}
			i++;
		}
	}

	free(tmp);
	return intersectC;
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

