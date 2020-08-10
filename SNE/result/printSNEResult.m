clc;
clear;
close all;
addpath("../data:../util");

n = 1000;
m = 784;
d = 3;
nbTests = 2;

Class = [0, 1, 2, 3];
data = loadMNISTDataSet(Class,n,m,nbTests);

load('STORMwithPCAinit.mat');
plot3D(w,data.LL,num2str(Class'),'STORM result with PCA initialisation');
load('STORMwithRandomInit.mat');
plot3D(w,data.LL,num2str(Class'),'STORM result with random initialisation');
load('TSNE.mat');
plot3D(w,data.LL,num2str(Class'),'T-SNE result');
