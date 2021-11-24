#include "demo.cuh"

int main(int argc, char* argv[]) {
	//INITIALIZATION
	int no_classes, wnd_sz, K, sz[4];
	int i, j, jj;
	char color_map[20];
	clock_t time;

	if (argc < 4) {
		printf("Parameter error\n"); //param.txt, HSI.txt, labels.txt
		exit(1);
	}

	FILE* f0 = fopen(argv[1], "r");
	FILE* f1 = fopen(argv[2], "r");
	FILE* f2 = fopen(argv[3], "r");
	FILE* f3 = fopen("lcmrfea_all.bin", "rb");
	//FILE* test = fopen("test.txt", "w");

	fscanf(f0, "%d", &no_classes);
	fscanf(f0, "%d", &wnd_sz);
	fscanf(f0, "%d", &K);
	for (i = 0; i < 4; i++) {
		fscanf(f0, "%d", &sz[i]);
	}
	fscanf(f0, "%s", color_map);

	double* RD_hsi = (double*)malloc(sizeof(double) * sz[0] * sz[1] * sz[2]);
	double* img = (double*)malloc(sizeof(double) * sz[0] * sz[1] * sz[3]);
	int* labels = (int*)malloc(sizeof(int) * sz[0] * sz[1]);
	double* lcmrfea_all = (double*)malloc(sizeof(double) * sz[2] * sz[2] * sz[0] * sz[1]);
	
	readLabels(f2, labels, sz);

	if (!f3) {
		readHSI(f1, img, sz);
		fun_myMNF(img, RD_hsi, sz);

		f3 = fopen("lcmrfea_all.bin", "wb");
		fun_LCMR_all(RD_hsi, wnd_sz, K, sz, lcmrfea_all);
		fwrite(lcmrfea_all, sizeof(double), sz[2] * sz[2] * sz[0] * sz[1],f3);
	}
	else {
		fread(lcmrfea_all, sizeof(double), sz[2] * sz[2] * sz[0] * sz[1], f3);
	}

	int* train_id = (int*)malloc(sizeof(int) * no_classes * TRAIN_NUMBER);
	double* train_label = (double*)malloc(sizeof(double) * no_classes * TRAIN_NUMBER);
	int* test_id = (int*)malloc(sizeof(int) * (no_classes * sz[0] * sz[1] - no_classes * TRAIN_NUMBER));
	int* test_label = (int*)malloc(sizeof(int) * (no_classes * sz[0] * sz[1] - no_classes * TRAIN_NUMBER));
	double* test_cov = (double*)malloc(sizeof(double) * sz[2] * sz[2] * sz[0] * sz[1]);
	double* train_cov = (double*)malloc(sizeof(double) * sz[2] * sz[2] * no_classes * TRAIN_NUMBER);
	double* OA = (double*)malloc(sizeof(double) * N_IT);
	double* predict_label = (double*)malloc(sizeof(double) * sz[0] * sz[1]);
	double* class_accuracy = (double*)malloc(sizeof(double) * no_classes);
	double kappa;

	double* tmp_label = (double*)malloc(sizeof(double) * no_classes * sz[0] * sz[1]);
	int* tmp_id = (int*)malloc(sizeof(int) * no_classes * sz[0] * sz[1]);
	int* indices = (int*)malloc(sizeof(int) * no_classes * TRAIN_NUMBER);
	int* nrPixelsPerClass = (int*)malloc(sizeof(int) * no_classes);
	int* errorMatrix = (int*)malloc(sizeof(int) * no_classes * no_classes);

	double* train_value = (double*)malloc(sizeof(double) * (no_classes * TRAIN_NUMBER) * (no_classes * TRAIN_NUMBER));
	double* test_value = (double*)malloc(sizeof(double) * (no_classes * TRAIN_NUMBER * sz[0] * sz[1]));
	
	memset(OA, 0, sizeof(double) * N_IT);

	//SVM
	struct svm_model* model;
	struct svm_parameter param;
	struct svm_problem prob; // = ktrain
	struct svm_node** testnode; // = ktest

	svmSetParameter(&param, no_classes * TRAIN_NUMBER);
	svmSetProblem(&prob, train_label, no_classes * TRAIN_NUMBER);

	testnode = (struct svm_node**)malloc(sz[0] * sz[1] * sizeof(struct svm_node*));
	for (i = 0; i < sz[0] * sz[1]; i++) {
		testnode[i] = (struct svm_node*)malloc((no_classes * TRAIN_NUMBER + 2) * sizeof(struct svm_node));
	}

	//CUDA SETTINGS
	double* d_train_cov, * d_test_cov, * d_lcmrfea_all, * d_train_value, * d_test_value;
	int* d_train_id;

	cudaMalloc((void**)&d_train_cov, sz[2] * sz[2] * no_classes * TRAIN_NUMBER * sizeof(double));
	cudaMalloc((void**)&d_test_cov, sizeof(double) * sz[2] * sz[2] * sz[0] * sz[1]);
	cudaMalloc((void**)&d_lcmrfea_all, sz[2] * sz[2] * sz[0] * sz[1] * sizeof(double));
	cudaMalloc((void**)&d_train_id, (no_classes * sz[0] * sz[1] - no_classes * TRAIN_NUMBER) * sizeof(int));

	cudaMalloc((void**)&d_train_value, sizeof(double) * (no_classes * TRAIN_NUMBER) * (no_classes * TRAIN_NUMBER));
	cudaMalloc((void**)&d_test_value, sizeof(double) * (no_classes * TRAIN_NUMBER * sz[0] * sz[1]));

	cudaMemcpy(d_lcmrfea_all, lcmrfea_all, sz[2] * sz[2] * sz[0] * sz[1] * sizeof(double), cudaMemcpyHostToDevice);

	dim3 dimBlock(8, 8); //64 threads
	dim3 dimGrid((no_classes * TRAIN_NUMBER) / dimBlock.x + 1, (sz[2] * sz[2]) / dimBlock.y + 1);

	cublasHandle_t handle;
	cublasCreate(&handle);

	cudaStream_t stream1;
	cudaStreamCreate(&stream1);

	double alpha = 1.0;
	double beta = 0;
	//////////////

	time = clock();

	//COMPUTATION
	for (i = 0; i < N_IT; i++) {
		//printf("N_IT: %d\n\n", i + 1);

		cudaMemcpyAsync(d_test_cov, lcmrfea_all, sizeof(double) * sz[2] * sz[2] * sz[0] * sz[1], cudaMemcpyHostToDevice, stream1);

		int test_size = 0;
		generateSample(labels, no_classes, sz, train_id, train_label, test_id, test_label, &test_size, tmp_label, tmp_id, indices);

		cudaMemcpy(d_train_id, train_id, (no_classes * TRAIN_NUMBER) * sizeof(int), cudaMemcpyHostToDevice);
		loadTrainData << <dimGrid, dimBlock >> > (d_train_cov, d_lcmrfea_all, d_train_id, sz[2], no_classes);
		
		if (i > 0) {
			svm_free_model_content(model);
		}

		cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, sz[0] * sz[1], no_classes * TRAIN_NUMBER, sz[2] * sz[2], &alpha, d_test_cov, sz[2] * sz[2], d_train_cov, sz[2] * sz[2], &beta, d_test_value, sz[0] * sz[1]);
		cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, no_classes* TRAIN_NUMBER, no_classes* TRAIN_NUMBER, sz[2] * sz[2], &alpha, d_train_cov, sz[2] * sz[2], d_train_cov, sz[2] * sz[2], &beta, d_train_value, no_classes* TRAIN_NUMBER);
		
		cudaMemcpy(test_value, d_test_value, sizeof(double)* (no_classes* TRAIN_NUMBER* sz[0] * sz[1]), cudaMemcpyDeviceToHost);
		cudaMemcpy(train_value, d_train_value, sizeof(double)* (no_classes* TRAIN_NUMBER)* (no_classes* TRAIN_NUMBER), cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();

		#pragma omp parallel for schedule (static)
		for (j = 0; j < no_classes * TRAIN_NUMBER; j++) {
			prob.x[j][0].index = 0;
			prob.x[j][0].value = j + 1;

			prob.x[j][(no_classes * TRAIN_NUMBER) + 1].index = -1;
			prob.x[j][(no_classes * TRAIN_NUMBER) + 1].value = 0;

			for (jj = 0; jj < no_classes * TRAIN_NUMBER; jj++) {
				prob.x[j][jj + 1].index = jj + 1;
				prob.x[j][jj + 1].value = train_value[jj * no_classes * TRAIN_NUMBER + j];
			}

			for (jj = 0; jj < sz[0] * sz[1]; jj++) {
				testnode[jj][0].index = 0;
				testnode[jj][0].value = jj + 1;

				testnode[jj][j + 1].index = j + 1;
				testnode[jj][j + 1].value = test_value[j * sz[0] * sz[1] + jj];

				testnode[jj][(no_classes * TRAIN_NUMBER) + 1].index = -1;
				testnode[jj][(no_classes * TRAIN_NUMBER) + 1].value = 0;
			}
		}

		model = svm_train(&prob, &param);

		for (j = 0; j < sz[0] * sz[1]; j++) {
			predict_label[j] = svm_predict(model, testnode[j]);
		}

		calcError(&OA[i], class_accuracy, test_label, predict_label, test_id, test_size, no_classes, &kappa, nrPixelsPerClass, errorMatrix);

		//printf("\n**********************\nMean class accuracy : % lf\nOverall accuracy : % lf\nKappa : % lf\n**********************\n", mean(class_accuracy, no_classes), OA[i], kappa);
	}

	time = clock() - time;

	printf("\nMean overall accuracy: %lf\n", mean(OA, N_IT));
	printf("\nElapsed computation time: %.5f seconds\n", ((double)time) / CLOCKS_PER_SEC);
	writeBMP(predict_label, sz[1], sz[0], "map.jpg", color_map);
	printf("Classification map image saved\n");

	fclose(f0);
	fclose(f1);
	fclose(f2);
	fclose(f3);
	//fclose(test);

	cudaFree(d_train_cov);
	cudaFree(d_test_cov);
	cudaFree(d_lcmrfea_all);
	cudaFree(d_train_value);
	cudaFree(d_test_value);
	cudaFree(d_train_id);

	cublasDestroy(handle);
	cudaStreamDestroy(stream1);

	free(tmp_id);
	free(tmp_label);
	free(indices);

	free(nrPixelsPerClass);
	free(errorMatrix);

	free(RD_hsi);
	free(labels);
	free(lcmrfea_all);
	free(train_id);
	free(train_label);
	free(test_id);
	free(test_label);
	free(test_cov);
	free(train_cov);
	free(OA);
	free(predict_label);
	free(class_accuracy);

	free(train_value);
	free(test_value);

	free(img);

	for (i = 0; i < no_classes * TRAIN_NUMBER; i++) {
		free(prob.x[i]);
	}
	free(prob.x);

	for (i = 0; i < sz[0] * sz[1]; i++) {
		free(testnode[i]);
	}
	free(testnode);

	svm_free_and_destroy_model(&model);
	svm_destroy_param(&param);

	return 0;
}