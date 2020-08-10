function data = loadMNISTDataSet(Class,n,m,nbTests)
    dataLoaded = loadMNISTImages('data/train-images.idx3-ubyte')';
    labelLoaded = loadMNISTLabels('data/train-labels.idx1-ubyte');
    i = 1;
    data.DL = zeros(n,size(dataLoaded,2));
    data.LL = zeros(n,1);
    dataTest = [];
    labelTest = [];
    for j=1:length(labelLoaded)
        if (ismember(labelLoaded(j),Class))
            if (i<n)
                data.DL(i,:) = dataLoaded(j,:);
                data.LL(i) = labelLoaded(j);
                i = i+1;
            else
                dataTest(i-m+1,:) = dataLoaded(j,:);
                labelTest(i-m+1) = labelLoaded(j);
                i=i+1;
            end
            if (i>4*n)
                break;
            end
        end
    end
    randint = randi(length(labelTest),[nbTests,1]);
    data.DT = dataTest(randint,:);
    data.LT = labelTest(randint);
end
