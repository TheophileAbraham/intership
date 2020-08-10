function [] = plot3D(donnees,label,legendVar,titleVar)
    n = size(donnees,1);
    X=NaN(10,n);
    Y=NaN(10,n);
    Z=NaN(10,n);
    for i=0:9
        j = 1;
        for k=1:n
            if (label(k) == i)
                X(i+1,j)=donnees(k,1);
                Y(i+1,j)=donnees(k,2);
                Z(i+1,j)=donnees(k,3);
                j=j+1;
            end
        end
    end
    color=[[1,0,0];[0,1,0];[0,0,1];[1,1,0];[1,0,1];[0,1,1];[0,0,0];[0.3,0.3,0.3];[0.6,0.6,0.6];[0.5,0,0]];
    figure;
    scatter3(X(1,:),Y(1,:),Z(1,:),50,color(1,:),'+');
    hold on;
    for i=2:10
        scatter3(X(i,:),Y(i,:),Z(i,:),50,color(i,:),'+');
    end
    legend(legendVar);
    title(titleVar);
end