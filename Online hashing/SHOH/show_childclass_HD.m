function show_childclass_HD(B,NL,param)

    idx=cell(size(B,2),1);


    cla2cla=zeros(param.num_class2,param.num_class2);
    sam2cla=zeros(param.num_class2,param.num_class2);
    for i=1:param.num_class2
        idx{i,1}=find(NL==i);
    end

    for i=1:param.num_class2
        idx1=idx{i,1};
        b1=sign(mean(B(:,idx1),2));
        b1(b1==0)=-1;
        CC(:,i)=b1;
    end

    for i=1:param.num_class2
        c1=CC(:,i);
        for j=i:param.num_class2
            c2=CC(:,j);
            cla2cla(i,j)=(param.nbits-c1'*c2)/2;
            cla2cla(j,i)=cla2cla(i,j);
        end
    end

    for i=1:param.num_class2
        c1=CC(:,i);
        
        for j=1:param.num_class2
            idx2=idx{j,1};
            sam2cla(i,j)=sum(bsxfun(@minus,param.nbits,c1'*B(:,idx2))/2)/length(idx2);
        end
    end
    
    if strcmp(param.ds_name,'FashionVC')
        label={'Fc1','Fc2','Fc3','Fc4','Fc5','Fc6','Fc7','Fc8','Fc9','Fc10',...
            'Fc11','Fc12','Fc13','Fc14','Fc15','Fc16','Fc17','Fc18','Fc19','Fc20',...
            'Fc21','Fc22','Fc23','Fc24','Fc25','Fc26','Fc27'};
    elseif strcmp(param.ds_name,'Ssense')
        label={'Sc1','Sc2','Sc3','Sc4','Sc5','Sc6','Sc7','Sc8','Sc9','Sc10',...
            'Sc11','Sc12','Sc13','Sc14','Sc15','Sc16','Sc17','Sc18','Sc19','Sc20',...
            'Sc21','Sc22','Sc23','Sc24','Sc25','Sc26','Sc27','Sc28'};
    end

%     ax.ColorLimits=[0 param.nbits/2];
%     set(ax,'Units', 'points')
%     set(ax,'Position',[40,40,270,270])
%     set(gcf, 'Position', [700,600,480,440]);
%     set(ax,'FontSize',12)
%     title(param.hash_method);
%     ax.Colormap=hot;
%     fig=gcf;
%     fig.Units='points';
%     width = fig.Position(3);
%     height = fig.Position(4);
%     set(gcf,'PaperUnits','points');
%     set(gcf,'PaperSize',[width height]);
%     
%     
%     output_dir='D:\workspace\matlab\SHOH-master\methods\heatmap\';
%     if ~exist(output_dir,'dir')
%         mkdir(output_dir);
%     end
%     output_name=fullfile(output_dir,[param.hash_method '_childclass_' param.ds_name '_' num2str(param.nbits) '_c2c']);
% %     title([param.hash_method ' parentclass in ' param.ds_name ' at ' num2str(param.nbits) '-bit']);
%     print(output_name,'-dpdf', '-r600');
% 
% 
    figure
    ax=heatmap(label,label, sam2cla);
    ax.CellLabelColor = 'none'; 

    ax.ColorLimits=[0 param.nbits/2];
    set(ax,'Units', 'points')
    set(ax,'Position',[40,40,270,270])
    set(gcf, 'Position', [700,600,480,440]);
    set(ax,'FontSize',12)
    title(param.hash_method);
    ax.Colormap=hot;
    fig=gcf;
    fig.Units='points';
    width = fig.Position(3);
    height = fig.Position(4);
    set(gcf,'PaperUnits','points');
    set(gcf,'PaperSize',[width height]);
%     
%     output_name=fullfile(output_dir,[param.hash_method '_childclass_' param.ds_name '_' num2str(param.nbits) '_s2c']);
% %     title([param.hash_method ' parentclass in ' param.ds_name ' at ' num2str(param.nbits) '-bit']);
%     print(output_name,'-dpdf', '-r600');

end






