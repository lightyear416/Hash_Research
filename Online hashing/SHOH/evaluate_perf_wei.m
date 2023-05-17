function eva = evaluate_perf_wei(top_K,B,queryBx,queryBy,trainL2,queryL2)
    % B: ntrain x r
    % queryBx, queryBy: ntrain x r
    % queryX: nquery x dx
    % Wx: dx x r
    BxTrain=B;
    ByTrain=BxTrain;
    
    BxQuery=sign(queryBx);
    ByQuery=sign(queryBy);
%     BxQuery=compactbit(queryBx>0);
%     ByQuery=compactbit(queryBy>0);
    
    %% Cross-modal Retrieval
    % I->T
%     Dist=hammingDist(BxQuery,ByTrain);
%     wei=zeros(size(queryBx));
    wei=abs(queryBx);
    wei(wei>=1)=1;
%     Dist=sum(wei,2).*(size(B,2)-BxQuery*ByTrain')/2;
%     Dist=(repmat(sum(abs(queryBx),2),1,size(BxTrain,1))-abs(queryBx).*BxQuery*ByTrain')/2;
    Dist=(repmat(sum(wei,2),1,size(BxTrain,1))-wei.*BxQuery*ByTrain')/2;
%     Dist=(size(B,2)-BxQuery*ByTrain')/2;
%     Dist=(size(B,2)-abs(queryBx).*BxQuery*ByTrain')/2;
    [~,idx_rank]=sort(Dist,2);
    eva.map_image2text=mAP(idx_rank',trainL2,queryL2);
    [eva.precision_image2text, eva.recall_image2text] = precision_recall(idx_rank', trainL2, queryL2);
    eva.precisionK_image2text = precision_at_k(idx_rank', trainL2, queryL2, top_K);
    


    % T->I
    wei=abs(queryBy);
    wei(wei>=1)=1;
    Dist=(repmat(sum(wei,2),1,size(ByTrain,1))-wei.*ByQuery*BxTrain')/2;
%     Dist=(size(B,2)-abs(queryBy).*ByQuery*BxTrain')/2;
%     Dist=hammingDist(ByQuery,BxTrain);
    [~,idx_rank]=sort(Dist,2);
    eva.map_text2image=mAP(idx_rank',trainL2,queryL2);
    [eva.precision_text2image, eva.recall_text2image] = precision_recall(idx_rank', trainL2, queryL2);
    eva.precisionK_text2image = precision_at_k(idx_rank', trainL2, queryL2, top_K);
    


end

