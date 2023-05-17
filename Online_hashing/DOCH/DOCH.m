% close all; clear; clc;
function DOCH(EXPparam,train,query)
% nbits_set=[8 16 32 64 128];
nbits_set=EXPparam.nbits;
param.top_K=EXPparam.top_K;
%% load dataset

% set = 'MIRFlickr';
set = EXPparam.ds_name;



%% initialization
fprintf('initializing...\n')
alphas = [0.45 0.35 0.25 0.15 0.1];
param.datasets = set;
if strcmp(set,'MIRFlickr')
    load('MIRFLICKR.mat');
    param.iter = 3; 
    param.num_anchor = 50;
    param.theta = 0.1;
    param.chunk_size = 2000;
    X = XAll; Y = YAll; L = LAll;
    R = randperm(size(L,1));
    queryInds = R(1:2000);
    sampleInds = R(2001:end);
    param.nchunks = floor(length(sampleInds)/param.chunk_size);
       
    XChunk = cell(param.nchunks,1);
    YChunk = cell(param.nchunks,1);
    LChunk = cell(param.nchunks,1);
    for subi = 1:param.nchunks-1
        XChunk{subi,1} = X(sampleInds(param.chunk_size*(subi-1)+1:param.chunk_size*subi),:);
        YChunk{subi,1} = Y(sampleInds(param.chunk_size*(subi-1)+1:param.chunk_size*subi),:);
        LChunk{subi,1} = L(sampleInds(param.chunk_size*(subi-1)+1:param.chunk_size*subi),:);
    end
    XChunk{param.nchunks,1} = X(sampleInds(param.chunk_size*subi+1:end),:);
    YChunk{param.nchunks,1} = Y(sampleInds(param.chunk_size*subi+1:end),:);
    LChunk{param.nchunks,1} = L(sampleInds(param.chunk_size*subi+1:end),:);
        
    XTest = X(queryInds, :); YTest = Y(queryInds, :); LTest = L(queryInds, :);
    clear X Y L
end

if strcmp(set,'Ssense') || strcmp(set,'FashionVC') || strcmp(set,'MIRFlickr-25K')
%     load('MIRFLICKR.mat');
    param.iter = 3; 
    param.num_anchor = 50;
    param.theta = 0.1;
%     param.chunk_size = 2000;
    param.chunk_size = EXPparam.chunk_size;

%     X = XAll; Y = YAll; L = LAll;
%     R = randperm(size(L,1));
%     queryInds = R(1:2000);
%     sampleInds = R(2001:end);
    X = train.X; Y = train.Y; L = train.L2; 
    
    L1 = train.L1;                          %%
    sampleInds=1:size(X,1);
    param.nchunks = floor(length(sampleInds)/param.chunk_size);
    
       
    XChunk = cell(param.nchunks,1);
    YChunk = cell(param.nchunks,1);
    LChunk = cell(param.nchunks,1);
    L1Chunk = cell(param.nchunks,1);        %%
    for subi = 1:param.nchunks-1
        XChunk{subi,1} = X(sampleInds(param.chunk_size*(subi-1)+1:param.chunk_size*subi),:);
        YChunk{subi,1} = Y(sampleInds(param.chunk_size*(subi-1)+1:param.chunk_size*subi),:);
        LChunk{subi,1} = L(sampleInds(param.chunk_size*(subi-1)+1:param.chunk_size*subi),:);
        L1Chunk{subi,1} = L1(sampleInds(param.chunk_size*(subi-1)+1:param.chunk_size*subi),:);
    end
    XChunk{param.nchunks,1} = X(sampleInds(param.chunk_size*subi+1:end),:);
    YChunk{param.nchunks,1} = Y(sampleInds(param.chunk_size*subi+1:end),:);
    LChunk{param.nchunks,1} = L(sampleInds(param.chunk_size*subi+1:end),:);
    L1Chunk{param.nchunks,1} = L1(sampleInds(param.chunk_size*subi+1:end),:);       %%
        
    XTest = query.X; YTest = query.Y; LTest = query.L2;
    L1Test = query.L1;                      %%

    trainsize=train.size;
    querysize=query.size;
%     clear train query
    clear X Y L L1
end


for bit=1:length(nbits_set) 
    nbits = nbits_set(bit);
%     param.alpha = alphas(bit);
    switch(nbits)
        case 8
            param.alpha=0.45;
        case 16
            param.alpha=0.35;
        case 32
            param.alpha=0.25;
        case 64
            param.alpha=0.15;
        case 128
            param.alpha=0.1;
        otherwise
            error('undefined alpha!\n');
    end
    
    %% DOCH
    param.nbits=nbits;
    [eva_info,time]=evaluate_DOCH(XChunk,YChunk,LChunk,XTest,YTest,LTest,param,L1Chunk,L1Test,EXPparam,train.NL1,train.NL2);



%-------------------------------SAVE RECORDS------------------------------%
record_dir=fullfile(EXPparam.rec_dir,EXPparam.hash_method,EXPparam.ds_name,EXPparam.samp_method);
if(~exist(record_dir,'dir'))
    mkdir(record_dir);
end

record_name=['test', num2str(EXPparam.t), ...
    '_TrainSize=',num2str(trainsize), ...
    '_QuerySize=',num2str(querysize), ...
    '_ChunkSize=',num2str(param.chunk_size), ...
    '_NumBits=',num2str(param.nbits), ...
    '_', EXPparam.avai_labels, ...
    '.mat'];

save(fullfile(record_dir,record_name),'EXPparam','param','eva_info','time','-v7.3');


%-------------------------------------------------------------------------%

end
end