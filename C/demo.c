#include "header.h"

int main(int argc, char* argv[]) {
	//INITIALIZATION
	int no_classes, wnd_sz, K, sz[3];
	int i, j, jj;
	clock_t time;

	if (argc < 5) {
		printf("Parameter error\n"); //param.txt, MNF.txt, labels.txt, lcmrfea_all.txt
		exit(1);
	}

	FILE* f0 = fopen(argv[1], "r");
	FILE* f1 = fopen(argv[2], "r");
	FILE* f2 = fopen(argv[3], "r");
	FILE* f3 = fopen(argv[4], "r+");
	FILE* test = fopen("test.txt", "w");
	
	fscanf(f0, "%d", &no_classes);
	fscanf(f0, "%d", &wnd_sz);
	fscanf(f0, "%d", &K);
	for(i=0; i<3; i++){
		fscanf(f0, "%d", &sz[i]);
	}

	double* RD_hsi = (double*)malloc(sizeof(double) * sz[0] * sz[1] * sz[2]);
	int* labels = (int*)malloc(sizeof(int) * sz[0] * sz[1]);
	double* lcmrfea_all = (double*)malloc(sizeof(double) * sz[2] * sz[2] * sz[0] * sz[1]);

	readHSI(f1, RD_hsi, sz);
	readLabels(f2, labels, sz);
	
	if (!f3) {
		f3 = fopen(argv[4], "w");
		fun_LCMR_all(test,RD_hsi, wnd_sz, K, sz, lcmrfea_all);
		savelcmrFEA(f3,lcmrfea_all, sz);
	}else {
		readlcmrFEA(f3, lcmrfea_all, sz);
	}

	int* train_id = (int*)malloc(sizeof(int) * no_classes * TRAIN_NUMBER);
	double* train_label = (double*)malloc(sizeof(double) * no_classes * TRAIN_NUMBER);
	int* test_id = (int*)malloc(sizeof(int) * no_classes*sz[0]*sz[1]);
	int* test_label = (int*)malloc(sizeof(int) * no_classes * sz[0] * sz[1]);
	double* test_cov = (double*)malloc(sizeof(double) * sz[2] * sz[2] * sz[0] * sz[1]);
	double* train_cov = (double*)malloc(sizeof(double) * sz[2] * sz[2] * no_classes * TRAIN_NUMBER);
	double* OA = (double*)malloc(sizeof(double) * N_IT);
	double* predict_label = (double*)malloc(sizeof(double) * sz[0] * sz[1]);
	double* class_accuracy = (double*)malloc(sizeof(double) * no_classes);
	double kappa;
	
	memset(OA, 0, sizeof(double)*N_IT);

	//SVM
	struct svm_model *model;
	struct svm_parameter param;
	struct svm_problem prob;
	struct svm_node **testnode;

	svmSetParameter(&param);
	svmSetProblem(&prob, train_label, no_classes * TRAIN_NUMBER);

	testnode = (struct svm_node **)malloc(sz[0] * sz[1]*sizeof(struct svm_node*));
	for(i=0; i<sz[0] * sz[1]; i++){
		testnode[i] = (struct svm_node *)malloc(no_classes * TRAIN_NUMBER* sizeof(struct svm_node));
	}

	time = clock();
	
	//COMPUTATION
	for (i = 0; i < N_IT; i++) {
		printf("N_IT: %d\n\n", i);
		
		int test_size = 0;
		generateSample(labels, no_classes, sz, train_id, train_label, test_id, test_label, &test_size);

		for (j = 0; j < (no_classes * TRAIN_NUMBER); j++) {
			for (jj = 0; jj < sz[2]*sz[2]; jj++) {
				train_cov[j*sz[2]*sz[2]+jj] = lcmrfea_all[train_id[j]*sz[2]*sz[2] + jj];
			}
		}
		memcpy(test_cov, lcmrfea_all, sizeof(double) * sz[2] * sz[2] * sz[0] * sz[1]);

		logmkernel(test, prob.x, train_cov, train_cov, no_classes * TRAIN_NUMBER, sz[2] * sz[2],  no_classes * TRAIN_NUMBER);
		logmkernel(test, testnode, train_cov, test_cov, no_classes * TRAIN_NUMBER, sz[2] * sz[2],  sz[0] * sz[1]); //forse trasposizione
		
		
		if(svm_check_parameter(&prob,&param)){
			printf("SVM parameters error!\n");
			exit(1);
		}
		
		model = svm_train(&prob,&param);
		
		for(j=0; j<sz[0] * sz[1]; j++){
			predict_label[j]=svm_predict(model, testnode[j]);
		}
		
		svm_free_model_content(model);

		calcError(OA, class_accuracy, test_label, predict_label, test_id, i, test_size, no_classes, sz, &kappa);
	
		printf("Class Accuracy: ");
		for (j = 0; j < no_classes; j++) {
			printf("%lf ", class_accuracy[j]);
		}
		printf("\nMean class accuracy: %lf\nOverall accuracy: %lf\nKappa: %lf\n**********************\n", mean(class_accuracy), OA[i], kappa);
	}
	
	time = clock()-time;
	
	printf("\nMean overall accuracy: %lf\n", mean(OA));
	printf("\nElapsed time: %.5f seconds\n", ((double)time) / CLOCKS_PER_SEC);
	writeBMP(predict_label, sz[1], sz[2], "map.jpg", "india");
	printf("\nClassification map image saved\n");

	fclose(f0);
	fclose(f1);
	fclose(f2);
	fclose(f3);
	fclose(test);

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
	
	for(i=0;i < no_classes * TRAIN_NUMBER; i++){
		free(prob.x[i]);
	}
	free(prob.x);
	
	for(i=0; i<sz[0] * sz[1]; i++){
		free(testnode[i]);
	}
	free(testnode);
	
	svm_free_and_destroy_model(&model);
	svm_destroy_param(&param);
	
	return 0;
}
