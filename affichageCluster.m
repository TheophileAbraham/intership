clc;
clear;
close all;
addpath("data:kNN:PCA:SNE:SNE/function:util");

nbTests = 2;

n = 1000;
m = 784;
d = 3;

config.l1 = 0;
config.m = d;
config.STORM_max_iters = 1500;
config.STORM_ifnormalization = 0;
config.STORM_ifreplace = 0;
config.STORM_eps = 0.01;
config.STORM_lr = 0.02;
config.STORM_initial_bs=1000;
config.STORM_loop_bs_g=1000;
config.STORM_loop_bs_G=1000;
config.STORM_loop_bs_F=1000;
config.STORM_a_g=0.01;
config.STORM_a_G=0.01;
config.STORM_a_F=0.01;

% Load data set
% Class = 0:9;
Class = [0, 1, 2, 3];
data = loadMNISTDataSet(Class,n,m,nbTests);

%% PCA analysis
fprintf("-----PCA analysis-----\n");
[C,Xmean,W] = PCALearning(data.DL,d);
% [C,W,Xmean] = pca(data.DL');
% C = C(:,1:d);
plot3D(C,data.LL,num2str(Class'),'PCA result');

%% STORM analysis
fprintf("-----STORM analysis-----\n");
[data0,~,~] = PCALearning(data.DL,10); 
dataI = Dist(data0, 0);
data.P = dataI ./ (sum(dataI, 1)-1);
load('save.mat');
w0 = w;
% w0 = C;
% w0 = ones(size(data.P, 1), config.m) + [1:size(data.P,1)]'/size(data.P,1);
% w0 = (rand(size(data.P,1),config.m) - 0.5) * 0.01;
        %w=(w_1//...//w_n) each w_i is d-dimensional row vector
[storm, grad_storm, norm_storm, w] = STORMoptimisation(data,w0,config);
plot3D(w,data.LL,num2str(Class'),'SNE result');

%% plot the gradient
grad_storm = grad_storm/n;
minval = min(storm);
figure;
semilogy(grad_storm, smooth(norm_storm, 10), '-X', 'Color', [1 0 0], 'LineWidth', 1, 'MarkerSize', 5, 'MarkerIndices', 1:2:length(grad_storm));
legend('STORM-C');
%xlim([0, 50]);
xlabel('Grads Calculation/n');
ylabel('Objective Value Gap');
title('SNE via STORM-C on MNIST Dataset');

