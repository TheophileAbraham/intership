function [indLabel,ClassOut] = searchClass(label,ClassIn)
    % Search the label in ClassIn. indLabel is the indice of the
    % corresponding label in ClassOut. If the label is in ClassIn, ClassOut
    % is ClassIn. Otherwize, ClassOut is ClassIn which we append the new
    % label
    % @param
    %   T label : the label we want to find in ClassIn
    %   T[] ClassIn : the list of label already encoutered
    % @result
    %   int indLabel : the indice of the corresponding label in ClassOut
    %   T[] ClassOut: If the label is in ClassIn, ClassOutis ClassIn.
    %       Otherwize, ClassOut is ClassIn which we append the new label
    i=1;
    while (i<length(ClassIn) && ClassIn(i) ~= label)
        i = i+1;
    end
    if (i<=length(ClassIn) && ClassIn(i) == label)
        indLabel = i;
        ClassOut = ClassIn;
    else
        indLabel = i+1;
        ClassOut = [ClassIn, label];
    end
end

