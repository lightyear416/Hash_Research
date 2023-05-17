function [precision, recall]=precision_recall_coarse(ids, Lbase, Lquery, K)
    
    if ~exist('K','var')
        K = size(Lbase,1);
    end

    nquery = size(ids, 2);
%     P = zeros(K, nquery);

    for i = 1 : nquery
        label = Lquery(i, :);
        label(label == 0) = -1;
        idx = ids(:, i);
        
        imatch = sum(bsxfun(@eq, Lbase(idx(1:K), :), label), 2) > 0;
        Tp = sum(imatch);
        iall = sum(bsxfun(@eq, Lbase(idx, :), label), 2) > 0;
        P = sum(iall);
        precision=Tp/K;
        recall=Tp/P;
 
    end


end