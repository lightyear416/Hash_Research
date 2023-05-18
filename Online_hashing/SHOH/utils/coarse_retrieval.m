function coarse_retrieval(train,ByTrain,)
    extra_query=zeros(1,4945);
    % extra_query(108)=1;     % black
    % extra_query(1900)=1;    % striped
    extra_query(1224)=1;    % bag

    EyQuery=compactbit(extra_query*train.Wy'>0);
    Dist=hammingDist(EyQuery,ByTrain);
    [~,idx_rank]=sort(Dist,2);
    res=param.train_idx(idx_rank);

    path = 'D:\workspace\matlab\other\mat2png\Ssense\bmp\';
    for i=1:train.size
        if ~train.L1(idx_rank(i),2)
            first_not_match=i;break
        end
    end
    rows=10;
    cols=10;
    num=rows*cols;
    str=floor(first_not_match/num)*num;
    figure
    for i=1:num
        %         rectangle('edgecolor','r')
        file_name=[path num2str(res(i+str)) '.bmp'];
        c=mod(i-1,cols);
        r=ceil(i/cols);
        plt=subplot('position',[c/cols,...
            1-(1+r)/(rows+1),...
            1/cols,1/(rows+1)]);
        imshow(file_name);
        if ~train.L1(idx_rank(str+i),2)
            set(plt, 'Visible', 'on','XTick', [],'YTick', [],'XColor','r','YColor','r');
        else
            set(plt, 'Visible', 'off','XTick', [],'YTick', [],'XColor','w','YColor','w');
        end
    end

    set(gcf,'Position',[100,100,560,616]);
    %     p = get(gcf,'Position');
    %     k = [224 224]/(224+224);
    %     set(gcf,'Position',[p(1),p(2),(p(3)+p(4)*1.1).*k])
    sgtitle(['results(' num2str(str+1) '-' num2str(str+rows*cols) 'th) of SHOH retrieving "BAG", the fisrt unmatched result is the ' num2str(first_not_match) '-th'],'FontSize',10)

    output_dir='C:\Users\Schatten\Desktop\SHOH\figure Five Methods\retrieval_set\';
    if ~exist(output_dir,'dir')
        mkdir(output_dir);
    end
    output_name=fullfile(output_dir,['SHOH_' param.ds_name '_' num2str(param.nbits)]);

    fig = gcf;
    fig.PaperPositionMode = 'auto';
    fig_pos = fig.PaperPosition;
    fig.PaperSize = [fig_pos(3) fig_pos(4)];
    print(output_name,'-vector','-dpdf','-r600');