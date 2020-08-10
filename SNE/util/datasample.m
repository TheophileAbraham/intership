function res = datasample(list,nbElement)
    n = length(list);
    for i=1:nbElement
        res(i) = list(randi(n));
    end
end