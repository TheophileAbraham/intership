function res = pdist2(data)
    n = size(data,1);
    res = zeros(1,(n)*(n-1)/2);
    t = 1;
    for i=1:n
        for j=i+1:n
            res(t) = norm(data(i,:)-data(j,:));
            t = t+1;
        end
    end
end