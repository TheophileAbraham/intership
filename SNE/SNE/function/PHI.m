function [f] = PHI(X,data,config)
    n = size(data,1);
    d = config.m;
    
    %% compute g
    g = zeros(n*d+n,n);
    for i=1:n
        g(1:n*d,i) = reshape(X',[1, d*n])';
        g(n*d+1:end,i) = n*exp(-dist(X,X(i,:)'));
    end
    g = sum(g,2)/n;
    
    %% compute f
    f = 0;
    for j=1:n
        for i=1:n
            f = f + n * data(i,j) * (norm(g((i-1)*d+1:i*d)-g((j-1)*d+1:j*d)) + log(g(d*n+i)));
        end
    end
    f = f/n;
end

