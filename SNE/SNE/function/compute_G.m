%calculate G(x,B)=1/|B| G_i(x) where G_i(x) are the matrices consisting of grad g with respect to x_1,...,x_n
function [G] = compute_G(data, indexes)
    n = size(data, 1);
    d = size(data, 2);
    DI = Dist(data, 0);
    batch_size = length(indexes);
    G = zeros(d, n, n);
    for l =1:n
        mat = zeros(d, n);
        for batchindex = 1:batch_size
            mat_batchindex = zeros(d, n);
            mat_batchindex(:, indexes(batchindex)) = 2*n*(data(l, :) - data(indexes(batchindex), :))'*DI(l, indexes(batchindex));
            mat_batchindex(:, l) = -2*n*(data(l, :) - data(indexes(batchindex), :))'*DI(l, indexes(batchindex));
            mat = mat + mat_batchindex;
        end
        G(:, :, l) = mat / batch_size;
    end
end