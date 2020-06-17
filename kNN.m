function label = kNN(dataTest,dataLearning,labelDataLearning,ListClass,k)
    % do the k-means algorithm to classify data_test
    % @param
    %   float[] data_test : the data we want to classify
    %   float[][] data_learning : the data used to perform the learning
    %       (each data is on one colomn of data_learning)
    %   T[] label_data_learning : the label of data_learning
    % @result
    %   T label : the label of data_test
    % 
    % T is the type of label (int, string....)
    % Determination de l'image d'apprentissage la plus proche (plus proche voisin) :
    distances = zeros(size(dataLearning,1),1);
    for point=1:size(dataLearning,1)
        distances(point) = sqrt(sum((dataLearning(point,:) - dataTest).^2));
    end
    [~,indices] = sort(distances,'ascend');
    indices = indices(1:k);
    nb_representants_classes = zeros(length(ListClass),1);
    for j=1:length(ListClass)
        nb_representants_classes(j) = length(find(labelDataLearning(indices) == ListClass(j)));
    end

    nb_max_representant = max(nb_representants_classes);
    classes_choisies = find(nb_representants_classes == nb_max_representant);
    if length(classes_choisies) == 1
        label = ListClass(classes_choisies);
    else
        label = ListClass(classes_choisies(1));
    end
end

