% demo for "A New Spatial-Spectral Feature Extraction Method for Hyperspectral 
%              Images Using Local Covariance Matrix Representation" to be
%              appeared TGRS, Leyuan Fang et al.
%Hunan Univerisity
%written by Nanjun He (henanjun@hnu.edu.cn)

clc
close all
clear;clc;
load('../../hsi data/Pavia.mat')
load('../../hsi data/Pavia_gt.mat')
%load('ind_MNF_20.mat') % MNF by ENVI software
%[RD_hsi] = ind_MNF_20;
labels = pavia_gt;
no_classes = 9;
wnd_sz = 25;
K = 102;
train_number = ones(1,no_classes)*5;
tic
[RD_hsi]  = fun_MyMNF(pavia, 20); % MNF code written by author.
if(~exist('lcmrfea_all.mat'))
    [lcmrfea_all] =  fun_LCMR_all(RD_hsi,wnd_sz,K);
    save lcmrfea_all lcmrfea_all
else
    load lcmrfea_all
end
toc
sz = size(RD_hsi);

tic
for flag = 1:10
        [train_SL,test_SL,test_number]= GenerateSample(labels,train_number,no_classes);
        train_id = train_SL(1,:);
        train_label = train_SL(2,:);
        test_id = test_SL(1,:);
        test_label = test_SL(2,:);
        train_cov = lcmrfea_all(:,:,train_id);
        test_cov = lcmrfea_all;
        KMatrix_Train = logmkernel(train_cov, train_cov);
        KMatrix_Test = logmkernel(train_cov, test_cov);
        Ktrain = [(1:size(KMatrix_Train,1))',KMatrix_Train];     
        model = svmtrain(train_label', Ktrain, '-q -t 4');  
        Ktest = [(1:size(KMatrix_Test,2))', KMatrix_Test'];  
        tmp = ones(1,size(KMatrix_Test,2));
        [predict_label, accuracy, P1] = svmpredict(tmp',Ktest,model); 
        [OA(flag),Kappa(flag),AA(flag),CA(:,flag)] = calcError(test_SL(2,:)'-1,predict_label(test_id)-1,[1:no_classes]);
end
toc
mean(OA)
classification_map = reshape(predict_label,sz(1),sz(2));
classification_map=label2color(classification_map,'india');
imshow(classification_map)
imwrite(classification_map,'indian05.jpg')




