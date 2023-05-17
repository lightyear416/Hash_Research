function plot_parentclass_heatmap(B,NL,param)

    idx=cell(param.num_class1,1);

    cla2cla=zeros(param.num_class1,param.num_class1);
    sam2cla=zeros(param.num_class1,param.num_class1);
    for i=1:param.num_class1
        idx{i,1}=find(NL==i);
    end

    for i=1:param.num_class1
        idx1=idx{i,1};
        b1=sign(mean(B(:,idx1),2));
        b1(b1==0)=-1;
        CC(:,i)=b1;
    end

    for i=1:param.num_class1
        c1=CC(:,i);
        for j=i:param.num_class1
            c2=CC(:,j);
            cla2cla(i,j)=(param.nbits-c1'*c2)/2;
            cla2cla(j,i)=cla2cla(i,j);
        end
    end

    for i=1:param.num_class1
        c1=CC(:,i);
        
        for j=1:param.num_class1
            idx2=idx{j,1};
            sam2cla(i,j)=sum(bsxfun(@minus,param.nbits,c1'*B(:,idx2))/2)/length(idx2);
        end
    end
    
    if strcmp(param.ds_name,'FashionVC')
        label={'Fp1','Fp2','Fp3','Fp4','Fp5','Fp6','Fp7','Fp8'};
    elseif strcmp(param.ds_name,'Ssense')
        label={'Sp1','Sp2','Sp3','Sp4'};
    end

    figure
    

%     pcolor(heat);
    
    %% 仿python
    imagesc(cla2cla);
    c=colorbar; c.FontSize = 18; %set(c,'box','off');
    ax=gca;
    set(gca,'box','off');
    ax.Colormap=hot;
    tick=1:length(label);
    ax.XTick = tick;
    ax.XTickLabel = label;
    ax.YTick = tick;
    ax.YTickLabel = label;

%     ax.FontSize = 14;
    set(ax, 'TickDir', 'out');
    ax.XColor = 'none'; ax.YColor = 'none';
    ax.XAxis.TickLabelColor = 'black'; ax.YAxis.TickLabelColor = 'black';
%         ax.XLabel.FontSize = 14;
%     ax.YLabel.FontSize = 14;
    % 绘制白色边框线
    for i = 1:size(cla2cla,1)
        for j = 1:size(cla2cla,2)
            rectangle('Position',[j-0.5 i-0.5 1 1],'EdgeColor','k','LineWidth',0.1);
        end
    end

%     ax=heatmap(label,label,cla2cla);
%     ax.CellLabelColor = 'none'; 
% 
    ax.CLim=[0 param.nbits/2];
        set(ax,'Units', 'points')
    set(ax,'Position',[40,35,260,260])
    
    set(gcf, 'Position', [700,600,470,430]);
    set(ax,'FontSize',12)
    title(param.hash_method,'FontSize',20);
    ax.Colormap=hot;
    fig=gcf;
    fig.Units='points';
    width = fig.Position(3);
    height = fig.Position(4);
    set(gcf,'PaperUnits','points');
    set(gcf,'PaperSize',[width height]);
    
    
    output_dir='D:\workspace\matlab\SHOH-master\methods\heatmap\';
    if ~exist(output_dir,'dir')
        mkdir(output_dir);
    end
    output_name=fullfile(output_dir,[param.hash_method '_parentclass_' param.ds_name '_' num2str(param.nbits) '_c2c']);
%     title([param.hash_method ' parentclass in ' param.ds_name ' at ' num2str(param.nbits) '-bit']);
    print(output_name,'-dpdf', '-r600');


%     figure
%     ax=heatmap(label,label,sam2cla);
%     ax.CellLabelColor = 'none'; 
% 
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
    
%     output_name=fullfile(output_dir,[param.hash_method '_parentclass_' param.ds_name '_' num2str(param.nbits) '_s2c']);
%     title([param.hash_method ' parentclass in ' param.ds_name ' at ' num2str(param.nbits) '-bit']);
%     print(output_name,'-dpdf', '-r600');
end



