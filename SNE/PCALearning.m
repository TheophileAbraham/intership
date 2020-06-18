function [C,Xmean,W] = PCALearning(X,N)
    % Perform a PCA algorithm on Data used for Learning
    % @param
    %   float[][] X : Data used for Learning
    %   int N : number of principal component kept
    % @result
    %   float[][] C : principal componant of Data
    %   float[][] Xmean : the average Data
    %   float[][] W : matrice used to compute the PCA for a Data test

    n=size(X,1);
    Xmean = sum(X,1)/n;
    Xc = X - Xmean;
    Sigma_2 = 1/n * Xc * transpose(Xc);
    [W_2,D] = eig(Sigma_2);
    [~,Indices]=sort(diag(D),'descend');
    W_2 = W_2(:,Indices);
    W_2(:,end) = zeros(n,1);
    W = transpose(Xc)*W_2;
    W = W./(sqrt(sum(W.^2,1)));
    C = Xc * W;
    C = C(:,1:N);
end

