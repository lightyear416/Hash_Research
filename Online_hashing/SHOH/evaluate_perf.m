function eva = evaluate_perf(top_K,B,queryBx,queryBy,trainL2,queryL2)
    % B: ntrain x r
    % queryBx, queryBy: ntrain x r
    % queryX: nquery x dx
    % Wx: dx x r
    BxTrain=compactbit(B>0);
    ByTrain=BxTrain;
    
%     queryBx=sign(queryBx);
%     queryBy=sign(queryBy);
    BxQuery=compactbit(sign(queryBx)>0);
    ByQuery=compactbit(sign(queryBy)>0);
    
    %% Cross-modal Retrieval
    % I->T
    Dist=hammingDist(BxQuery,ByTrain);
    [~,idx_rank]=sort(Dist,2);
    eva.map_image2text=mAP(idx_rank',trainL2,queryL2);
    [eva.precision_image2text, eva.recall_image2text] = precision_recall(idx_rank', trainL2, queryL2);
    eva.precisionK_image2text = precision_at_k(idx_rank', trainL2, queryL2, top_K);
    


    % T->I
    Dist=hammingDist(ByQuery,BxTrain);
    [~,idx_rank]=sort(Dist,2);
    eva.map_text2image=mAP(idx_rank',trainL2,queryL2);
    [eva.precision_text2image, eva.recall_text2image] = precision_recall(idx_rank', trainL2, queryL2);
    eva.precisionK_text2image = precision_at_k(idx_rank', trainL2, queryL2, top_K);
    


end

