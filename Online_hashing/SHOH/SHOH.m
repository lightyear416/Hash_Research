% close all; clear; clc;
function SHOH(param,train,query)
    close all
    addpath(genpath('./utils/'));
    if ~exist('param','var')
        param=[];
    end

    if ~isfield(param,'nbits')
        param.nbits=16;
    end

    if ~isfield(param,'hash_method')
        param.hash_method='SHOH';
    end

    if ~isfield(param,'ds_name')
        param.ds_name='Ssense';
    end
    
    if ~isfield(param,'feature')
        param.feature='GIST';
    end

    if ~isfield(param,'hierarchy')
        param.hierarchy='hie';      % 'non-hie' 'hie'
    end
    
    if ~isfield(param,'avai_labels')
        param.avai_labels='all';       % 'fine'
    end
    
    if ~isfield(param,'similar_labels')
        param.similar_labels='soft';
    end

    param.ds_dir = 'D:\workspace\matlab\datasets\';

    %% load data if not done before 
    if ~exist('train','var') || ~exist('query','var')
        [param,train,query]=load_dataset(param);
    end

    if ~isfield(param,'coarse_retrieval')
        if strcmp(param.ds_name,'Ssense')
            param.coarse_retrieval=true;
        else
            param.coarse_retrieval=false;
        end
    end
    
    nchunks= floor(train.size/param.chunk_size);


    if ~isfield(param,'t')
        param.t=0;
    else
        fprintf('test %s\n',num2str(param.t));
    end
    
%% --------------------- default hyperparameters ----------------------- %%    
    if ~isfield(param,'alpha1')
        param.alpha1=0.2;
%         param.alpha1=0.1;
        param.alpha2=1-param.alpha1;
    else
        param.alpha2=1-param.alpha1;
        fprintf('α1 = %s, α2 = %s\n',num2str(param.alpha1),num2str(param.alpha2));
    end
    if ~isfield(param,'eta')
        param.eta=10;
%         param.eta=1;
    else
        fprintf('η = %s\n',num2str(param.eta));
    end
    
    if ~isfield(param, 'gamma')
        param.gamma=1;
    else
        fprintf('γ = %s\n',num2str(param.gamma));
    end


    if ~isfield(param,'xi')
        param.xi=1;
    else
        fprintf('xi = %s\n',num2str(param.xi));
    end

    if ~isfield(param,'mu')
        param.mu=1000;
    else
        fprintf('mu = %s\n',num2str(param.mu));
    end
    
    if ~isfield(param,'max_iter')
        param.max_iter=7;
    else
        fprintf('iter = %s\n',num2str(param.max_iter));
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    %% train
    eva_info=cell(nchunks,1);
    


% 
% 
% %     param.mu2=10;
% Yanchor=train.Y(1:2000,:);
% correlation1 = corr(Xanchor, train.L2(1:2000,:));
% threshold=0.5;
% [sorted_A, index] = sort(A, 'descend');
% % 使用find函数查找排序后向量中前500个元素的位置
% top500_indices = index(1:500);
% correlation2 = corr(Yanchor, train.L2(1:2000,:));
% selected_features2 = find(abs(correlation2) > threshold);
% 
% train.X=train.X(:,selected_features1);
% train.Y=train.Y(:,selected_features2);

    

    %% RBF
%     train.X = exp(-sqdist(train.X', Xanchor')/(2*param.mu));
%     query.X = exp(-sqdist(query.X', Xanchor')/(2*param.mu));
%     train.Y = exp(-sqdist(train.Y', Yanchor')/(2*param.mu2));
%     query.Y = exp(-sqdist(query.Y', Yanchor')/(2*param.mu2));
    
    %% tanh
%     train.X = tanh((train.X-repmat(mean(train.Xanchor),size(train.X,1),1))* train.Xanchor');
%     query.X = tanh((query.X-repmat(mean(train.Xanchor),size(query.X,1),1))* train.Xanchor');
%     train.Y = tanh(train.Y* train.Yanchor');
%     query.Y = tanh(query.Y* train.Yanchor');

% * train.Xanchor').^2


    %% linear
%     train.X = train.X-repmat(mean(train.Xanchor),size(train.X,1),1);
%     query.X = query.X-repmat(mean(train.Xanchor),size(query.X,1),1);
%     train.Y = tanh(train.Y* train.Yanchor');
%     query.Y = tanh(query.Y* train.Yanchor');

    %% Chi-squared Kernel
%     train.X = chi2_kernel(train.X,train.Xanchor,param.mu);
%     query.X = chi2_kernel(query.X,train.Xanchor,param.mu);
%     train.Y = chi2_kernel(train.Y,train.Yanchor,param.mu);
%     query.Y = chi2_kernel(query.Y,train.Yanchor,param.mu);

    %% polynomial_kernel
%     train.X = polynomial_kernel(train.X,train.Xanchor);
%     query.X = polynomial_kernel(query.X,train.Xanchor);
%     train.Y = polynomial_kernel(train.Y,train.Yanchor);
%     query.Y = polynomial_kernel(query.Y,train.Yanchor);






% train.X = chi2_kernel(train.X,Xanchor,param.mu);
% query.X = chi2_kernel(query.X,Xanchor,param.mu);

% train.X = exp(-sqdist(train.X', Xanchor')/(2*param.mu));
% query.X = exp(-sqdist(query.X', Xanchor')/(2*param.mu));
% meanY=mean(Yanchor,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% meanX=mean(train.X,1);s

% train.X=bsxfun(@minus,train.X,meanX);
% query.X=bsxfun(@minus,query.X,meanX);
% 
% train.Y=bsxfun(@minus,train.Y,meanY);
% query.Y=bsxfun(@minus,query.Y,meanY);


% 计算协方差矩阵
% X_decenter=bsxfun(@minus,train.X,meanX);


% PCA降维
% k = 500;  % 设置降维后的维度
% [coeff, score, latent] = pca(X_decenter, 'NumComponents', k);
% 
% 
% % 降维后的特征矩阵
% train.X = X_decenter * coeff;
% 
% query.X = bsxfun(@minus,query.X,meanX)*coeff;
% 
% Xanchor = train.X(1:500,:);
% % train.X = chi2_kernel(train.X,Xanchor,0.1);
% % query.X = chi2_kernel(query.X,Xanchor,0.1);
% train.X = exp(-sqdist(train.X', Xanchor')/(2*param.mu));
% query.X = exp(-sqdist(query.X', Xanchor')/(2*param.mu));

    % show progress
    fprintf('training progress: ');
    backNum = fprintf('%.1f %%\n',0);
    first_round=true;

    for i=1:nchunks
        idx_strt=(i-1)*param.chunk_size+1;
        if(i~=nchunks)
            idx_end=idx_strt-1+param.chunk_size;
        else
            idx_end=train.size;
        end
        %% hierarchy: 'hie' 'non-hie'
        
        if strcmp(param.avai_labels,'fine')       
            train = train_SHOH_nonhie(param,train,idx_strt:idx_end,first_round);            % 'non-hie' 'fine'
            suffix='fine';
        elseif strcmp(param.avai_labels,'all')
            if strcmp(param.hierarchy,'non-hie')
                train = train_SHOH_nonhie(param,train,idx_strt:idx_end,first_round);        % 'non-hie' 'all'
                suffix='all';
            elseif strcmp(param.hierarchy,'hie')
                suffix=param.similar_labels;
                train = train_SHOH(param,train,idx_strt:idx_end,first_round);               % 'hie';    similar_labels:   'soft' 'hard'
            end
        end

        percent = i/nchunks*100;
        
        fprintf(repmat('\b',1,backNum));
        backNum = fprintf('%.1f %%\n',percent);
        first_round = false;
%         eva_info{i,1}=evaluate_perf(param.top_K,train.B',query.X*train.Wx',query.Y*train.Wy',train.L2(1:train.trained,:),query.L2);
       
        
    end
    eva_info{i,1}=evaluate_perf(param.top_K,train.B',query.X*train.Wx',query.Y*train.Wy',train.L2(1:train.trained,:),query.L2);
    eva_info_wei{i,1}=evaluate_perf_wei(param.top_K,train.B',query.X*train.Wx',query.Y*train.Wy',train.L2(1:train.trained,:),query.L2);
    
    fprintf('MAP in I->T: %.4g\n',eva_info{nchunks,1}.map_image2text);
    fprintf('MAP in T->I: %.4g\n',eva_info{nchunks,1}.map_text2image);

    fprintf('MAP in I->T: %.4g\n',eva_info_wei{nchunks,1}.map_image2text);
    fprintf('MAP in T->I: %.4g\n',eva_info_wei{nchunks,1}.map_text2image);
    time=train.time;


%% --------------TSNE for image TRAIN WITHOUT class-wise hash codes -----------%
%     plot.Y=tsne([train.B'],'Algorithm','barneshut','Distance','hamming');
%     symbol3=[];
%     size3=[];
%     for i=1:param.num_class2
%         symbol3=[symbol3 '.'];
%         size3=[size3 6];
%     end
% 
%     [all_themes, all_colors] = GetColors();
% %     scatter3(plot.Y(:,1),plot.Y(:,2),1,train.NL1)
% 
%     symbol3=[];
%     size3=[];
%     for i=1:param.num_class2
%         symbol3=[symbol3 '.'];
%         size3=[size3 6];
%     end
%     if strcmp(param.ds_name,'Ssense')
%         gscatter(plot.Y(:,1), ...
%             plot.Y(:,2), ...
%             [train.NL2], ...
%             all_colors([1:32,5:32],:), ...
%             ...all_colors([ 1 1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4],:), ...
%             [  symbol3], ...
%             [  size3]);
%     elseif strcmp(param.ds_name,'FashionVC')
%         gscatter(plot.Y(:,1), ...
%             plot.Y(:,2), ...
%             [train.NL2], ...
%             all_colors([1:35 9:35],:), ...
%             ...all_colors([1 2 2 2 3 3 3 3 3 3 3 4 4 4 5 5 6 7 7 7 8 8 8 8 8 8 8],:), ...
%             [  symbol3], ...
%             [  size3]);
%     end

    
    %% Coarse retrieval
    % T->I
    % Wy: 4945 x r
    % BxTrain: ntrain x r
    % L1Train: ntrain x 4


%     if param.coarse_retrieval==true
%         coarse_query=zeros(1,4945);
%         %     extra_query(108)=1;     % black
%         %     extra_query(1900)=1;    % striped
%         coarse_query(1224)=1;    % bag
%         if exist('meanY','var')
%             coarse_query=bsxfun(@minus,coarse_query,meanY);
%         end
%         [eva_info{i,1}.precisionK_text2image_coarse, eva_info{i,1}.recallK_text2image_coarse] ...
%             = perform_coarse_retrieval(param,coarse_query*train.Wy',train.B',train.L1);
%     end


    %% heatmap
%         plot_parentclass_heatmap(train.B,train.NL1,param);
%         plot_childclass_heatmap(train.B,train.NL2,param);

%% ----------------------------- SAVE RECORDS -------------------------- %%
    if ~isfield(param,'rec_dir')
        record_dir=fullfile(pwd,'results','SHOH',param.ds_name);
    else
        record_dir=fullfile(param.rec_dir,param.hash_method,param.ds_name,param.samp_method);
    end
    if(~exist(record_dir,'dir'))
        mkdir(record_dir);
    end
    if param.opti_param==false
        record_name=['test', num2str(param.t), ...
            '_TrainSize=',num2str(train.size), ...
            '_QuerySize=',num2str(query.size), ...
            '_ChunkSize=',num2str(param.chunk_size), ...
            '_NumBits=',num2str(param.nbits), ...
            '_iter=',num2str(param.max_iter), ...
            '_',suffix, ...                                  % 'soft' 'hard' 'all' 'fine'
            '.mat'];
    else
        record_name=['test', num2str(param.t), ...
            '_α1=',num2str(param.alpha1), ...
            '_α2=',num2str(param.alpha2), ...
            '_η=',num2str(param.eta), ...
            '_γ=',num2str(param.gamma), ...            
            '_ξ=',num2str(param.xi), ...
            '_μ=',num2str(param.mu), ...
            '_NumBits=',num2str(param.nbits), ...
            '_iter=',num2str(param.max_iter), ...
            '.mat'];
    end
    save(fullfile(record_dir,record_name),'param','eva_info','eva_info_wei','time','-v7.3');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end


function [K,time] = chi2_kernel1(X,Y)
% Compute the Chi-squared Kernel between two matrices X and Y
% X: n*d matrix
% Y: m*d matrix
% K: n*m kernel matrix
tic;
[n,d] = size(X);
[m,~] = size(Y);

% Compute the pairwise distance
D = zeros(n,m);
for i = 1:d
    D = D + bsxfun(@minus,X(:,i),Y(:,i)').^2./(X(:,i)+Y(:,i)');
end

% Compute the kernel matrix
K = exp(-D/2);
time=toc;
end

function K = chi2_kernel(X, Y, mu)
% Compute the chi-squared kernel between two matrices X and Y
% with parameter gamma
    if nargin < 3
        mu = 0.5;  
    end
    n1 = size(X,1);
    n2 = size(Y,1);
    K = zeros(n1,n2);
    
    for i = 1:n1
        for j = 1:n2
            num = (X(i,:)-Y(j,:)).^2;
            den = X(i,:)+Y(j,:);
            K(i,j) = sum(num./den);
        end
    end
    K = exp(-mu*K);
end

function K = polynomial_kernel(X, Y, degree, gamma, coef0)
% X和Y分别为大小为n1 x d和n2 x d的矩阵，表示两组输入数据
% degree、gamma和coef0是核函数的参数

if nargin < 3
    degree = 3;  % 默认为3次多项式
end

if nargin < 4
    gamma = 1;  % 默认gamma为1
end

if nargin < 5
    coef0 = 1;  % 默认coef0为1
end

% 计算核矩阵
K = (gamma*X*Y' + coef0).^degree;
end
