%calculate the affinity matrix of input data points in a high-dimensional space
function [dist_matrix] = Dist(data, sig)
    if sig == 0
        % dist_matrix = exp(-squareform(pdist(data, 'euclidean')).^2);
        dist_matrix = exp(-squareform(pdist2(data)).^2);        
    else
        % dist_matrix = exp(-squareform(pdist(data, 'euclidean')).^2./(2 * sig^2));
        dist_matrix = exp(-squareform(pdist2(data)).^2./(2 * sig^2));
    end
end