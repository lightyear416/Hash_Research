function eva = evaluate_perf_for_OSCMFH(top_K,B,queryBx,queryBy,trainL2,queryL2,trainL1,queryL1)

    BxTrain=compactbit(B>0);
    % BxTrain=compactbit(train.X(1:train.size,:)*train.Wx'>0);
    % ByTrain=compactbit(train.Y(1:train.size,:)*train.Wy'>0);
    ByTrain=BxTrain;
    % BxTrain=ByTrain;
    
    BxQuery=compactbit(queryBx>0);
    ByQuery=compactbit(queryBy>0);
    
    %cross-modal
    Dist=hammingDist(BxQuery,BxTrain);
    [~,idx_rank]=sort(Dist,2);
    eva.map_image2text=mAP(idx_rank',trainL2,queryL2);
    [eva.precision_image2text, eva.recall_image2text] = precision_recall(idx_rank', trainL2, queryL2);
    eva.precisionK_image2text = precision_at_k(idx_rank', trainL2, queryL2, top_K);

%     if exist('trainL1','var') && exist('queryL1','var')
%         eva.maps_image2text=mAPS(idx_rank',trainL2,queryL2,trainL1,queryL1);
%     end
%     eva.smap_image2text=
    
    
    Dist=hammingDist(ByQuery',ByTrain);
    [~,idx_rank]=sort(Dist,2);
    eva.map_text2image=mAP(idx_rank',trainL2,queryL2);
    [eva.precision_text2image, eva.recall_text2image] = precision_recall(idx_rank', trainL2, queryL2);
    eva.precisionK_text2image = precision_at_k(idx_rank', trainL2, queryL2, top_K);
    
%     if exist('trainL1','var') && exist('queryL1','var')
%         eva.maps_text2image=mAPS(idx_rank',trainL2,queryL2,trainL1,queryL1);
%     end
    
end

