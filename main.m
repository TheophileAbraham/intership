clc;
clear;
close all;

nbTests = 2;

n = 6000;
m = 784;
d = 10;

config.l1 = 0;
config.m = d;
config.STORM_max_iters = 350;
config.STORM_ifnormalization = 1;
config.STORM_ifreplace = 1;
config.STORM_eps = 0.1;
config.STORM_lr = 0.1;
config.STORM_initial_bs=100;
config.STORM_loop_bs_g=100;
config.STORM_loop_bs_G=100;
config.STORM_loop_bs_F=100;
config.STORM_a_g=0.01;
config.STORM_a_G=0.01;
config.STORM_a_F=0.01;

% Load data set
Class = [0, 1, 2, 3];
dataLoaded = loadMNISTImages('train-images.idx3-ubyte')';
labelLoaded = loadMNISTLabels('train-labels.idx1-ubyte');
i = 1;
data.DL = zeros(m-1,m);
data.LL = zeros(m-1,1);
dataTest = [];
labelTest = [];
for j=1:length(labelLoaded)
    if (ismember(labelLoaded(j),Class))
        if (i<m)
            data.DL(i,:) = dataLoaded(j,:);
            data.LL(i) = labelLoaded(j);
            i = i+1;
        else
            dataTest(i-m+1,:) = dataLoaded(j,:);
            labelTest(i-m+1) = labelLoaded(j);
            i=i+1;
        end
    end
    if (length(labelTest) >= 4*nbTests)
        break;
    end
end
randint = randi(length(labelTest),[nbTests,1]);
data.DT = dataTest(randint,:);
data.LT = labelTest(randint);

k = 5;

%% PCA analysis
fprintf("-----PCA analysis-----\n");
[C,Xmean,W] = PCALearning(data.DL,d);
% [C,W,Xmean] = pca(data.DL);
% C = C(1:d,:)';

for i=1:size(data.DT,1)
    Ctest = PCATest(data.DT(i,:),Xmean,W,d);
    labelChosen = kNN(Ctest,C,data.LL,0:9,k);
    fprintf("Individu %d reconnu comme %d\n",data.LT(i),labelChosen);
end

%% STORM analysis
fprintf("-----STORM analysis-----\n");
for i=1:size(data.DT,1)
    data.P = [data.DL; data.DT(i,:)];
    Xreduced = STORMoptimisation(data,config);
    labelChosen = kNN(Xreduced(size(Xreduced,1),:),Xreduced(1:(size(Xreduced,1)-1),:),data.LL,0:9,k);
    fprintf("Individu %d reconnu comme %d\n",data.LT(i),labelChosen);
end