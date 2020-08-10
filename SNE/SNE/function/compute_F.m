%calculate grad f^{(1)}(y, B) and grad f^{(2)}(y, B)
function [F_1, F_2] = compute_F(data, dataP, g, indexes);
    n = size(data, 1);
    d = size(data, 2);
    batch_size = length(indexes);
    F_1 = zeros(n, d);
    for j=1:batch_size
       sample_F = indexes(j);
       F_1_j = zeros(n, d);
       for l=1:n
           if l~=sample_F
               F_1_j(l, :) = 2 * n * (data(l, :) - data(sample_F, :))* dataP(sample_F, l);
           else
               F_1_j(l, :) = -2 * n * (data - data(sample_F, :))' * dataP(:, sample_F);
           end
       end
       F_1 = F_1 + F_1_j;
    end
    
   
    F_2 = 0;
    for j=1:batch_size
        sample_F = indexes(j);
        F_2j = (sum(dataP(sample_F, :), 1) * n .* (1./g))';
        F_2 = F_2 + F_2j;
    end
    
    F_1  = F_1  / batch_size;
    F_2  = F_2  / batch_size;
end