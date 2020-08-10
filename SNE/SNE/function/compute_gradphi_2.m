%calculate the second part of grad f
function [gradphi_2] = compute_gradphi_2(G, F_2)
    n = size(G, 2);
    d = size(G, 1);
    gradphi_2 = zeros(n, d);
    for i=1:n
        gradphi_2(i, :) = G(:, :, i) * F_2;
    end
end   