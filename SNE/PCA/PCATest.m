function C = PCATest(X,Xmean,W,N)
    % Perform a PCA algorithm on a Data test
    % @param
    %   float[] X : the Data test we want to reduce
    %   float[][] Xmean : the average Data of the Learning
    %   float[][] W : the W of PCALearning
    %   int N : number of principal componant kept
    C = (X-Xmean)*W;
    C = C(:,1:N);
end

