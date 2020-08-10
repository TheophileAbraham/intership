%calculate g_{n+j}(x, B) for j=1,2,...,n
function [g] = compute_g(data, indexes)
    n = size(data, 1);
    batch_size = length(indexes);
    DI = Dist(data, 0);
    g = sum(DI(indexes, :), 1) * n/batch_size - 1;
end