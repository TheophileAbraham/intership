function [max,indmax] = Max(T)
    indmax = 1;
    max = T(1);
    for i=2:length(T)
        if (max<T(i))
            indmax = i;
            max = T(i);
        end
    end
end

