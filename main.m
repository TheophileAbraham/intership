clc;
clear;
close all;
addpath("data:kNN:PCA:SNE:SNE/function:util");

nbTests = 2;

n = 2000;
m = 784;
d = 10;

config.l1 = 0;
config.m = d;
config.STORM_max_iters = 1500;
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
data = loadMNISTDataSet(Class,n,m,nbTests);

k = 5;

%% PCA analysis
fprintf("-----PCA analysis-----\n");
[C,Xmean,W] = PCALearning(data.DL,d);

for i=1:size(data.DT,1)
    Ctest = PCATest(data.DT(i,:),Xmean,W,d);
    labelChosen = kNN(Ctest,C,data.LL,0:9,k);
    fprintf("Individu %d reconnu comme %d\n",data.LT(i),labelChosen);
end

%% STORM analysis
fprintf("-----STORM analysis-----\n");
for i=1:size(data.DT,1)
    data0 = [data.DL; data.DT(i,:)];
    dataO = data0(1:n,:)/10;
    dataI = Dist(dataO, 0);
    data.P = dataI ./ (sum(dataI, 1)-1);
    Xreduced = STORMoptimisation(data,config);
    labelChosen = kNN(Xreduced(size(Xreduced,1),:),Xreduced(1:(size(Xreduced,1)-1),:),data.LL,0:9,k);
    fprintf("Individu %d reconnu comme %d\n",data.LT(i),labelChosen);
end

function [dist_matrix] = Dist(data, sig)
    if sig == 0
        dist_matrix = exp(-squareform(pdist(data, 'euclidean')).^2);        
    else
        dist_matrix = exp(-squareform(pdist(data, 'euclidean')).^2./(2 * sig^2));
    end
end