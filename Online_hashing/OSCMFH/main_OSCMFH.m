function [eva_info, time] = main_OSCMFH(streamdata,L1_tr,...
    I_te, T_te, L_te, L1_te,...
    bits,EXPparam,NL1,NL2,...
    iter,lambda, mu, gamma)

if ~exist('lambda','var')
    lambda = 0.1;
end
if ~exist('mu','var')
    mu = 0.01;
end
if ~exist('gamma','var')
    gamma = 0.01;
end
if ~exist('iter','var')
    iter = 40;
end

%% Training
nstream = size(streamdata,2);
eva_info=cell(nstream,1);

% Initialization
% traintime1 = cputime;
tic;
Itrain = streamdata{1,1};  Ttrain = streamdata{2,1};    LTrain = streamdata{3,1};
numdata = size(Itrain,1);
mean_I = mean(Itrain, 1);
mean_T = mean(Ttrain, 1);
Itrain = bsxfun(@minus, Itrain, mean_I);
Ttrain = bsxfun(@minus, Ttrain, mean_T);
mean_I = mean_I';
mean_T = mean_T'; 
%disp(['Batch:' num2str(1)  ' Total:' num2str(nstream)]);
[HH, HX, HY, R, U1, U2, P1, P2, obj] = solveOSCMFH(Itrain', Ttrain', LTrain', lambda, mu, gamma, bits, iter);

V = [HH;HY];  %V ：潜在语义表示矩阵
%P1，P2：线性投影矩阵
PI = V * Itrain / (Itrain' * Itrain + gamma * eye(size(Itrain,2)));
PT = V * Ttrain / (Ttrain' * Ttrain + gamma * eye(size(Ttrain,2)));

% Training:2--n
mFea1 = size(Itrain,2);
mFea2 = size(Ttrain,2);
E1 = Itrain' * HH';
E2 = Ttrain' * HH';
E3 = Itrain' * HX';
E4 = Ttrain' * HY';
C1 = HH * HH' + (gamma / lambda) * eye(size(HH,1));      
C2 = C1; 
C3 = HX * HX' + (gamma / lambda) * eye(size(P1,2)); 
C4 = HY * HY' + (gamma / lambda) * eye(size(P2,2)); 
F = HY * HX';
F1 = V*Itrain;
F2 = V*Ttrain;
W = HX *HX'+ (gamma / mu) * eye(size(HX,1));
W1 = Itrain'*Itrain + gamma * eye(mFea1);
W2 = Ttrain'*Ttrain + gamma * eye(mFea2);
HX1 = HX;
HY1 = HY;
L = LTrain';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cnt_trained=size(streamdata{1,1},1);
Y_tr = sign((bsxfun(@minus, V , mean(V,2)))');
Y_tr(Y_tr<0) = 0;
% 训练集散列码

Yi_te = sign((bsxfun(@minus,PI * I_te' , mean(V,2)))');
Yt_te = sign((bsxfun(@minus,PT * T_te' , mean(V,2)))');
Yi_te(Yi_te<0) = 0;
Yt_te(Yt_te<0) = 0;


time.train_time=toc;

if strcmp(EXPparam.avai_labels,'all')
    L2Train=L(EXPparam.num_class1+1:end,:)';
else
    L2Train=L';
end

% eva_info{1,1}=evaluate_perf(EXPparam.top_K,Y_tr,Yi_te,Yt_te,L2Train,L_te);

% eva_info{1,1}=evaluate_perf_for_OCMFH(1000,Y_tr,Yi_te,Yt_te,L_tr(1:cnt_trained,:),L_te);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


for i = 2:nstream 
%     traintime1 = cputime;
    tic;
    Itrain = streamdata{1,i}';  Ttrain = streamdata{2,i}';  LTrain = streamdata{3,i}';
    numdata_tmp = size(Itrain,2);
    mean_Itmp = mean(Itrain, 2);        
    mean_Ttmp = mean(Ttrain, 2);
    mean_I = (numdata*mean_I + numdata_tmp*mean_Itmp)/(numdata + numdata_tmp);
    mean_T = (numdata*mean_T + numdata_tmp*mean_Ttmp)/(numdata + numdata_tmp);
    Itrain = bsxfun(@minus, Itrain, mean_I);
    Ttrain = bsxfun(@minus, Ttrain, mean_T);
    numdata = numdata + numdata_tmp;
%    disp(['Batch:' num2str(i)  ' Total:' num2str(nstream)]);
    W1 = W1 + Itrain*Itrain';
    W2 = W2 + Ttrain*Ttrain';   
    [L, U1, U2, P1, P2, E1, E2, E3, E4, C1, C2, C3, C4, F, W, HH, HX, HY, HX1, HY1, R, Vt] = mysolveOSCMFH(Itrain, Ttrain, LTrain, L, U1, U2, P1, P2, E1, E2, E3, E4, C1, C2, C3, C4, F, W, HH, HX, HY, HX1, HY1, R, lambda, mu, gamma, iter);
      
    F1 = F1 + Vt*Itrain';  
    F2 = F2 + Vt*Ttrain';
    PI = F1 / W1;
    PT = F2 / W2;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    V = [HH;HY]; 
    cnt_trained=cnt_trained+size(streamdata{1,i},1);
    Y_tr = sign((bsxfun(@minus, V , mean(V,2)))');
%     Y_tr(Y_tr<0) = 0;
    % 训练集散列码

    Yi_te = sign((bsxfun(@minus,PI * I_te' , mean(V,2)))');
    Yt_te = sign((bsxfun(@minus,PT * T_te' , mean(V,2)))');

    
    time.train_time=[time.train_time;toc];

    if strcmp(EXPparam.avai_labels,'all')
        L2Train=L(EXPparam.num_class1+1:end,:)';
    else
        L2Train=L';
    end
%     eva_info{i,1}=evaluate_perf(EXPparam.top_K,Y_tr,Yi_te,Yt_te,L2Train,L_te);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
eva_info{i,1}=evaluate_perf(EXPparam.top_K,Y_tr,Yi_te,Yt_te,L2Train,L_te);
if EXPparam.coarse_retrieval==true
    % Wy: 4945 x r
    % BxTrain: ntrain x r
    % L1Train: ntrain x 4

    coarse_query=zeros(1,4945);
    %     extra_query(108)=1;     % black
    %     extra_query(1900)=1;    % striped
    coarse_query(1224)=1;    % bag
    [eva_info{i,1}.precisionK_text2image_coarse, eva_info{i,1}.recallK_text2image_coarse] ...
        = perform_coarse_retrieval(EXPparam,sign((bsxfun(@minus,PT * coarse_query' , mean(V,2)))'),Y_tr,L1_tr);
end

% plot_parentclass_heatmap(Y_tr',NL1,EXPparam);
% plot_childclass_heatmap(Y_tr',NL2,EXPparam);

% V = [HH;HY];  
%% Calculate hash codes
% Y_tr = sign((bsxfun(@minus, V , mean(V,2)))');
% Y_tr(Y_tr<0) = 0;
% B_Tr = compactbit(Y_tr);
% B_Ir = B_Tr;
% traintime2 = cputime;
% traintime = traintime2 - traintime1;
% 
% testtime1 = cputime;
% Yi_te = sign((bsxfun(@minus,PI * I_te' , mean(V,2)))');
% Yt_te = sign((bsxfun(@minus,PT * T_te' , mean(V,2)))');
% Yi_te(Yi_te<0) = 0;
% Yt_te(Yt_te<0) = 0;
% B_Te = compactbit(Yt_te);
% B_Ie = compactbit(Yi_te);
% testtime2 = cputime;
% testtime = testtime2 - testtime1;