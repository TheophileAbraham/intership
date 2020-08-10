%optimization on SNE via STORM-C
function [resu_obj,resu_cal,resu_norm,w] = STORMoptimisation(data, w0, config)
    w=w0;
    w_t = w;
    
    n = size(data,1);
    d = config.m;

    resu_obj = NaN(1,config.STORM_max_iters);
    grad_cal = 0;
    resu_cal = NaN(1, config.STORM_max_iters);
    resu_norm = NaN(1, config.STORM_max_iters);


    %% initialize g, G, F by minibatch-sampling the initial batches
    [g, G, F] = STORM_GD(data, w, config.STORM_initial_bs, config.STORM_ifreplace);


    iter = 1;
    while (iter < config.STORM_max_iters && max(max(abs(F)))>0.01)
        save('save.mat','w');
        fprintf("%f\n",max(max(abs(F))));
        lr = config.STORM_lr/(iter^2);
        [g, G, F] = STORM(data, w, w_t, g, G, F, config.STORM_loop_bs_g, config.STORM_loop_bs_G, config.STORM_loop_bs_F, config.STORM_a_g, config.STORM_a_G, config.STORM_a_F, config.STORM_ifreplace);
        w_t = w;
        w_tilde = w - config.STORM_lr * F;
        % w_tilde = w - lr * F;
        if config.STORM_ifnormalization == 1
            gamma = min(1/2, config.STORM_lr*config.STORM_eps/norm(w_tilde-w)); %normalization step in STORM-Compositional Algorithm
            w = w + gamma * (w_tilde-w);
        else
            w = w_tilde;
            % w = w./norm(w);
        end
        grad_cal = grad_cal + config.STORM_loop_bs_g + config.STORM_loop_bs_G + config.STORM_loop_bs_F; %store how many gradient queries are taken
        if config.l1 ~= 0
            w = sign(w).* max(0, abs(w)-config.l1);
        end
        norm_F = norm(F);
        [obj, ~] = compute_tsne(data, w, config);
        resu_obj(iter) = obj;
        resu_norm(iter) = norm_F;
        resu_cal(iter) = grad_cal;
        iter = iter+1;
    end
end
 
