function LEMON(EXPparam,train,query)
% close all; clear; clc;
addpath(genpath('./utils/'));
addpath(genpath('./codes/'));

% result_URL = './results/';
% if ~isdir(result_URL)
%     mkdir(result_URL);
% end


% db = {'IAPRTC-12','MIRFLICKR','NUSWIDE10'};
db={EXPparam.ds_name};
hashmethods = {'LEMON'};
% loopnbits = [8 16 32 64 128];
loopnbits=EXPparam.nbits;

param.top_K = EXPparam.top_K;
%% ---------------------------pass parameters--------------------------- %%
    param.chunksize = EXPparam.chunk_size;
    param.nchunks = floor(train.size/EXPparam.chunk_size);

    % for save records
    trainsize = train.size;
    querysize = query.size;
    
%% --------------------------------------------------------------------- %%



for dbi = 1     :length(db)
    db_name = db{dbi}; param.db_name = db_name;
    


    %% load train and query to keep data consistent for all the methods
    % LEMON默认数据加载方法，chunk_size不能整除train_size，剩余数据并入最后一个chunk

    XChunk = cell(param.nchunks,1);
    YChunk = cell(param.nchunks,1);
    LChunk = cell(param.nchunks,1);
    
%     sampleInds=1:trainsize;
%     sampleInds=sampleInds(randperm(train.size));
% 
%     for subi = 1:param.nchunks-1
%         XChunk{subi,1} = train.X(sampleInds(param.chunksize*(subi-1)+1:param.chunksize*subi),:);
%         YChunk{subi,1} = train.Y(sampleInds(param.chunksize*(subi-1)+1:param.chunksize*subi),:);
%         LChunk{subi,1} = train.L2(sampleInds(param.chunksize*(subi-1)+1:param.chunksize*subi),:);
%     end
%     XChunk{param.nchunks,1} = train.X(sampleInds(param.chunksize*subi+1:end),:);
%     YChunk{param.nchunks,1} = train.Y(sampleInds(param.chunksize*subi+1:end),:);
%     LChunk{param.nchunks,1} = train.L2(sampleInds(param.chunksize*subi+1:end),:);


    for subi = 1:param.nchunks
        idx_strt=(subi-1)*param.chunksize+1;
        idx_end=min(subi*param.chunksize,train.size);
        XChunk{subi,1} = train.X(idx_strt:idx_end,:);
        YChunk{subi,1} = train.Y(idx_strt:idx_end,:);
        LChunk{subi,1} = train.L2(idx_strt:idx_end,:);
        L1Chunk{subi,1} = train.L1(idx_strt:idx_end,:);
    end
    XChunk{subi,1} = train.X(idx_strt:end,:);
    YChunk{subi,1} = train.Y(idx_strt:end,:);
    LChunk{subi,1} = train.L2(idx_strt:end,:);
    L1Chunk{subi,1} = train.L1(idx_strt:end,:);


    %%
    
    XTest=query.X;
    YTest=query.Y;
    LTest=query.L2;
    L1Test=query.L1;
%     clear train query
  
    
    %% Methods
    eva_info = cell(length(hashmethods),length(loopnbits));
    
    
    for ii =1:length(loopnbits)
        fprintf('======%s: start %d bits encoding======\n\n',db_name,loopnbits(ii));
        param.nbits = loopnbits(ii);
        for jj = 1:length(hashmethods)
            
            switch(hashmethods{jj})
                case 'LEMON'
                    fprintf('......%s start...... \n\n', 'LEMON');
                    LEMONparam = param;
                    LEMONparam.alpha = 10000; LEMONparam.beta = 10000; LEMONparam.theta = 1;
                    LEMONparam.gamma = 0.1; LEMONparam.xi = 1;
                    [eva_info,time] = evaluate_LEMON(XChunk,YChunk,LChunk,XTest,YTest,LTest,LEMONparam,L1Chunk,L1Test,EXPparam,train.NL1,train.NL2);
            end
%             eva_info{jj,ii} = eva_info_;
%             clear eva_info_
        end
    end
    

    %% MAP - origin
%     for ii = 1:length(loopnbits)
%         for jj = 1:length(hashmethods)
%             Table_ItoT_MAP(jj,ii) = eva_info{jj,ii}{param.nchunks}.Image_VS_Text_MAP;
%             Table_TtoI_MAP(jj,ii) = eva_info{jj,ii}{param.nchunks}.Text_VS_Image_MAP;
%             
%             for kk = 1:param.nchunks
%                 % MAP
%                 Image_VS_Text_MAP{ii}{jj,kk} = eva_info{jj,ii}{kk}.Image_VS_Text_MAP;
%                 Text_VS_Image_MAP{ii}{jj,kk} = eva_info{jj,ii}{kk}.Text_VS_Image_MAP;
%                 
%                 % Precision VS Recall
%                 Image_VS_Text_recall{ii}{jj,kk,:}    = eva_info{jj,ii}{kk}.Image_VS_Text_recall';
%                 Image_VS_Text_precision{ii}{jj,kk,:} = eva_info{jj,ii}{kk}.Image_VS_Text_precision';
%                 Text_VS_Image_recall{ii}{jj,kk,:}    = eva_info{jj,ii}{kk}.Text_VS_Image_recall';
%                 Text_VS_Image_precision{ii}{jj,kk,:} = eva_info{jj,ii}{kk}.Text_VS_Image_precision';
% 
%                 % Top number Precision
%                 Image_To_Text_Precision{ii}{jj,kk,:} = eva_info{jj,ii}{kk}.Image_To_Text_Precision;
%                 Text_To_Image_Precision{ii}{jj,kk,:} = eva_info{jj,ii}{kk}.Text_To_Image_Precision;
% 
%                 trainT{ii}{jj,kk} = eva_info{jj,ii}{kk}.trainT;
%             end
%         end
%     end

%     for i=1:param.nchunks
%         % i2t
%         eva_info{i,1}.map_image2text            =   eva_info_{i,1}.Image_VS_Text_MAP;
%         eva_info{i,1}.precision_image2text      =   eva_info_{i,1}.Image_VS_Text_precision;
%         eva_info{i,1}.recall_image2text         =   eva_info_{i,1}.Image_VS_Text_recall;
%         eva_info{i,1}.precisionK_image2text     =   eva_info_{i,1}.Image_To_Text_Precision;
% %         eva_info{i,1}.maps_image2text           =   eva_info_{i,1}.Image_VS_Text_MAPS;
% 
% 
% 
%         % t2i
%         eva_info{i,1}.map_text2image            =   eva_info_{i,1}.Text_VS_Image_MAP;
%         eva_info{i,1}.precision_text2image      =   eva_info_{i,1}.Text_VS_Image_precision;
%         eva_info{i,1}.recall_text2image         =   eva_info_{i,1}.Text_VS_Image_recall;
%         eva_info{i,1}.precisionK_text2image     =   eva_info_{i,1}.Text_To_Image_Precision;
% %         eva_info{i,1}.maps_text2image           =   eva_info_{i,1}.Text_VS_Image_MAPS;


end

    
%     eva_info{param.nchunks,1}.precisionHD_image2text_L2 =   eva_info_{param.nchunks,1}.precisionHD_image2text_L2;
%     eva_info{param.nchunks,1}.precisionHD_image2text_L1 =   eva_info_{param.nchunks,1}.precisionHD_image2text_L1;
%     eva_info{param.nchunks,1}.precisionHD_text2image_L2 =   eva_info_{param.nchunks,1}.precisionHD_text2image_L2;
%     eva_info{param.nchunks,1}.precisionHD_text2image_L1 =   eva_info_{param.nchunks,1}.precisionHD_text2image_L1;
% 
%     eva_info{param.nchunks,1}.recallHD_image2text_L2 =   eva_info_{param.nchunks,1}.recallHD_image2text_L2;
%     eva_info{param.nchunks,1}.recallHD_image2text_L1 =   eva_info_{param.nchunks,1}.recallHD_image2text_L1;
%     eva_info{param.nchunks,1}.recallHD_text2image_L2 =   eva_info_{param.nchunks,1}.recallHD_text2image_L2;
%     eva_info{param.nchunks,1}.recallHD_text2image_L1 =   eva_info_{param.nchunks,1}.recallHD_text2image_L1;

    
    fprintf('MAP in I->T: %.4g\n',eva_info{param.nchunks,1}.map_image2text);
    fprintf('MAP in T->I: %.4g\n',eva_info{param.nchunks,1}.map_text2image);
    
    %% Save

%-------------------------------SAVE RECORDS------------------------------%
record_dir=fullfile(EXPparam.rec_dir,EXPparam.hash_method,EXPparam.ds_name,EXPparam.samp_method);
if(~exist(record_dir,'dir'))
    mkdir(record_dir);
end

record_name=['test', num2str(EXPparam.t), ...
    '_TrainSize=',num2str(trainsize), ...
    '_QuerySize=',num2str(querysize), ...
    '_ChunkSize=',num2str(param.chunksize), ...
    '_NumBits=',num2str(param.nbits), ...
    '_', EXPparam.avai_labels, ...
    '.mat'];

save(fullfile(record_dir,record_name),'EXPparam','param','eva_info','time','-v7.3');


%-------------------------------------------------------------------------%

%     save(result_name,'eva_info','param','loopnbits','hashmethods',...
%         'XChunk','XTest','YChunk','YTest','LChunk','LTest',...
%         'trainT','Image_VS_Text_MAP','Text_VS_Image_MAP','Image_VS_Text_recall','Image_VS_Text_precision',...
%         'Table_ItoT_MAP','Table_TtoI_MAP',...
%         'Text_VS_Image_recall','Text_VS_Image_precision','Image_To_Text_Precision','Text_To_Image_Precision','-v7.3');
end
