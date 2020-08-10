clc;
clear;
close all;
addpath("data:kNN:PCA:SNE:SNE/function:util");

nbTests = 2;

n = 1000;
m = 784;

Class = [0, 1, 2, 3];
data = loadMNISTDataSet(Class,n,m,nbTests);
[data0,~,~] = PCALearning(data.DL,10);
dataI = Dist(data0, 0);
data.P = dataI ./ (sum(dataI, 1)-1);

indice0 = find(data.LL == 0);
indice3 = find(data.LL == 3);
mean0 = mean(data.P(indice0(1),indice0(2:end)));
mean3 = mean(data.P(indice0(1),indice3));
fprintf("la moyenne des distances d'un 0 et les autres est %f et celle de ce zeros et de tout les 3 est %f\n",mean0,mean3);