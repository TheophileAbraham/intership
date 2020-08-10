%The STORM estimator for SNE problem
function [g, G, F] = STORM(data, w, w_t, g, G, F, batch_size_g, batch_size_G, batch_size_F, a_g, a_G, a_F, ifreplace)
    n = size(data.P, 1);
    d = size(w, 2);
    dataP = data.P;

%% compute g
    %Choose minibatch B_{t+1}^g
    if ifreplace == 1
        indexes = datasample([1:n], batch_size_g); %sample with replacement
    else
        indexes = randperm(n);
        indexes = indexes(1:batch_size_g);        %sample without replacement    
    end
    %g(x_{t+1}, B_{t+1}^g) 
    g_mat = compute_g(w, indexes);
    %g(x_t, B_{t+1}^g) 
    g_mat_t = compute_g(w_t, indexes);
    %g_t
    g_t = g;
    %%calculate g_{t+1} = (1-a_g)g_t + a_g g(x_{t+1}, B_{t+1}^g) + (1-a_g)[g(x_{t+1}, B_{t+1}^g)-g(x_t, B_{t+1}^g)]
    %                   = (1-a_g)g_t + g(x_{t+1}, B_{t+1}^g) - (1-a_g)g(x_t, B_{t+1}^g)
    g = (1-a_g)*g + g_mat - (1-a_g) * g_mat_t;

%% compute G = partial g at steps t and t+1 
    %Choose minibatch B_{t+1}^{partial g}
    if ifreplace == 1
        indexes = datasample([1:n], batch_size_G); %sample with replacement
    else
        indexes = randperm(n);
        indexes = indexes(1:batch_size_G);        %sample without replacement    
    end
    %G(x_{t+1}, B_{t+1}^{partial g})
    G_ = compute_G(w, indexes);
    %G(x_t, B_{t+1}^{partial g}
    G_t = compute_G(w_t, indexes);
    %%calculate G_{t+1} = (1-a_G)G_t + a_G G(x_{t+1}, B_{t+1}^g) + (1-a_G)[G(x_{t+1}, B_{t+1}^g)-G(x_t, B_{t+1}^g)]
    %                   = (1-a_G)G_t + G(x_{t+1}, B_{t+1}^g) - (1-a_G)G(x_t, B_{t+1}^g)
    G = (1-a_G) * G + G_ - (1-a_G) * G_t;
    
%% compute grad f at steps t and t+1
    %Choose minibatch B_{t+1}^f
    if ifreplace == 1
        indexes = datasample([1:n], batch_size_F); %sample with replacement
    else
        indexes = randperm(n);
        indexes = indexes(1:batch_size_F);        %sample without replacement    
    end
    %update grad f(x_{t+1}, B_{t+1}^f
    [F_dev_, F_dev] = compute_F(w, dataP, g, indexes);
    %calculate grad f(x_{t}, B_{t+1}^f)    
    [F_dev_t, F_devt] = compute_F(w_t, dataP, g_t, indexes);   
    
%% compute F update, gradient
    %calculate F_{t+1}=(1-a_F)F_t + a_F G^Tgrad_f(x_{t+1}, B_{t+1}^f) + (1-a_F)[G^Tgrad_f(x_{t+1}, B_{t+1}^f)- G_t^Tgrad_f(x_t, B_t)]
    %                 =(1-a_F)F_t + G^Tgrad_f(x_{t+1}, B_{t+1}^f) - (1-a_F) G_t^Tgrad_f(x_t, B_t)               
    gradphi_2 = compute_gradphi_2(G, F_dev);
    gradphi_2t = compute_gradphi_2(G_t, F_devt);

    F = (1-a_F)*F + gradphi_2 - (1-a_F)*gradphi_2t + F_dev_ - (1-a_F)*F_dev_t;
end