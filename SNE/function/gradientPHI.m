function [F] = gradientPHI(x,data,config)
    n = size(data,1);
    d = config.m;
    F = zeros(n,d);
    y = compute_y(x,n,d);
    for j=1:n
        [F_1, F_2] = compute_F(y,j,data,n,d);
        for i=1:n
            G = compute_G(x,i,n,d);
            F = F + (F_1 + reshape(G'*F_2,[n,d]))/(n^2);
        end
    end
end

function y = compute_y(x,n,d)
    y = zeros(d+1,n);
    y(1:d,:) = x';
    for k=1:n
        for i=1:n
            y(d+1,k) = y(d+1,k) + exp(-norm(x(k,:)-x(i,:))^2);
        end
    end
end

function [F_1, F_2] = compute_F(y,j,data,n,d)
    %% Compute F_1
    F_1 = zeros(n,d);
    % compute non-j row
    for p=1:(j-1)
        F_1(p,:) = (2*n*data(p,j)*(y(1:d,p)-y(1:d,j)));
    end
    for p=(j+1):n
        F_1(p,:) = (2*n*data(p,j)*(y(1:d,p)-y(1:d,j)));
    end
    %compute j row
    for k=1:n
        F_1(j,:) = F_1(j,:) - 2 * n * data(k,j) * (y(1:d,k) - y(1:d,j))';
    end
    
    %% Compute F_2
    F_2 = (n * data(j,:)./y(d+1,:))';
end

function G = compute_G(x,i,n,d)
    G = zeros(n,d*n);
    for k=1:n
        G(k,d*(k-1)+1:d*k) = -2 * n * exp(-norm(x(k,:)-x(i,:))^2) * (x(k,:)-x(i,:));
        G(k,d*(i-1)+1:d*i) = 2 * n * exp(-norm(x(k,:)-x(i,:))^2) * (x(k,:)-x(i,:));
    end
end