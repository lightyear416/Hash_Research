function [pK_T2I_c,rK_T2I_c] =perform_coarse_retrieval(EXPparam,queryBy,B,L1Train)


    B=sign(B);
    BxTrain=compactbit(B>0);
    
    ByQuery=compactbit(sign(queryBy)>0);
    Dist=hammingDist(ByQuery,BxTrain);
    [~,idx_rank]=sort(Dist,2);
    pK_T2I_c = zeros(1,20);
    rK_T2I_c = zeros(1,20);
    for j=1:20
        [pK_T2I_c(j),rK_T2I_c(j)] = precision_recall_coarse(idx_rank', L1Train, [0 1 0 0], j*100);
    end


    res=EXPparam.train_idx(idx_rank);

    path = 'D:\workspace\matlab\other\mat2png\Ssense\bmp\';
    for i=1:13696
        if ~L1Train(idx_rank(i),2)
            first_not_match=i;break
        end
    end
    rows=10;
    cols=10;
    num=rows*cols;
    str=floor(first_not_match/num)*num;
    figure
    for i=1:num
        file_name=[path num2str(res(i+str)) '.bmp'];
        c=mod(i-1,cols);
        r=ceil(i/cols);
        plt=subplot('position',[c/cols,...
            1-(1+r)/(rows+1),...
            1/cols,1/(rows+1)]);
        imshow(file_name);
        if ~L1Train(idx_rank(str+i),2)
            set(plt, 'Visible', 'on','XTick', [],'YTick', [],'XColor','r','YColor','r');
        else
            set(plt, 'Visible', 'off','XTick', [],'YTick', [],'XColor','w','YColor','w');
        end
    end

    set(gcf,'Position',[100,100,565,622]);
    sgtitle(['results(' num2str(str+1) '-' num2str(str+rows*cols) 'th) of ' EXPparam.hash_method ' retrieving "BAG", the fisrt unmatched result is the ' num2str(first_not_match) '-th'],'FontSize',10)


    
    output_dir=fullfile(EXPparam.rec_dir, EXPparam.hash_method, EXPparam.ds_name, EXPparam.samp_method, 'coarse_retrieval');
    if ~exist(output_dir,'dir')
        mkdir(output_dir);
    end
    output_name=fullfile(output_dir,['_test' num2str(EXPparam.t) '_' EXPparam.hash_method '_' num2str(EXPparam.nbits) '_' EXPparam.avai_labels ]);

    fig = gcf;
    fig.PaperPositionMode = 'auto';
    fig_pos = fig.PaperPosition;
    fig.PaperSize = [fig_pos(3) fig_pos(4)];
    print(output_name,'-vector','-dpdf','-r600');
end