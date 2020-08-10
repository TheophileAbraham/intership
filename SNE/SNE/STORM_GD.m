%The STORM gradient initialization
function [g, G, F] = STORM_GD(data, w, batch_size, ifreplace)
    n = size(data.P, 1);    
    d = size(w, 2);
    dataP = data.P;
    %Choose batch of batch size batch_size
    if ifreplace == 1
        indexes = datasample([1:n], batch_size); %sample with replacement
    else
        indexes = randperm(n);
        indexes = indexes(1:batch_size);        %sample without replacement    
    end
%% compute g
    g = compute_g(w, indexes);
%% compute G
    G = compute_G(w, indexes);
%% compute F
    [F_dev_, F_dev] = compute_F(w, dataP, g, indexes);
%% compute gradient
    gradphi_2 = compute_gradphi_2(G, F_dev);
    F = gradphi_2 + F_dev_;
end

