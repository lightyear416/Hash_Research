function [eva_info,time] = evaluate_LEMON(XChunk,YChunk,LChunk,XTest,YTest,LTest,param,L1Chunk,L1Test,EXPparam,NL1,NL2)
    
    eva_info = cell(param.nchunks,1);
    for chunki = 1:param.nchunks
%         fprintf('-----chunk----- %3d\n', chunki);
        
        LTrain = cell2mat(LChunk(1:chunki,:));
        L1Train = cell2mat(L1Chunk(1:chunki,:));
        XTrain_new = XChunk{chunki,:};
        YTrain_new = YChunk{chunki,:};
        if strcmp(EXPparam.avai_labels,'all')
            LTrain_new = [L1Chunk{chunki,:} LChunk{chunki,:}];          %feed all labels
        else
            LTrain_new = LChunk{chunki,:};
        end
        GTrain_new = NormalizeFea(LTrain_new,1);
        
        % Hash code learning
        if chunki == 1
%             tic;
            [BB,XW,YW,HH,time] = train_LEMON0(XTrain_new,YTrain_new,LTrain_new,GTrain_new,param);
%             traintime=toc;  % Training Time
%             time.train_time=toc;
            % evaluation_info.trainT=traintime;
        else
%             tic;
            [BB,XW,YW,HH,time] = train_LEMON(XTrain_new,YTrain_new,LTrain_new,GTrain_new,BB,HH,param,time);
%             traintime=toc;  % Training Time
%             time.train_time=[time.train_time;toc];
            % evaluation_info.trainT=traintime;
        end

%         tic;
%         BxTest = compactbit(XTest*XW>0);
%         ByTest = compactbit(YTest*YW>0);
% %         evaluation_info.compressT=toc;   
        B = cell2mat(BB(1:end,:));
%         BxTrain = compactbit(B>0);
%         ByTrain = BxTrain; 

        tic;

%         eva_info{chunki,1}=evaluate_perf(param.top_K,B,XTest*XW,YTest*YW,LTrain,LTest);

%         DHamm = hammingDist(BxTest, BxTrain);
%         [~, orderH] = sort(DHamm, 2);
%         evaluation_info.Image_VS_Text_MAP = mAP(orderH', LTrain, LTest);
%         [evaluation_info.Image_VS_Text_precision, evaluation_info.Image_VS_Text_recall] = precision_recall(orderH', LTrain, LTest);
%         evaluation_info.Image_To_Text_Precision = precision_at_k(orderH', LTrain, LTest,param.top_K);
% 
%         DHamm = hammingDist(ByTest, ByTrain);
%         [~, orderH] = sort(DHamm, 2);
%         evaluation_info.Text_VS_Image_MAP = mAP(orderH', LTrain, LTest);
%         [evaluation_info.Text_VS_Image_precision,evaluation_info.Text_VS_Image_recall] = precision_recall(orderH', LTrain, LTest);
%         evaluation_info.Text_To_Image_Precision = precision_at_k(orderH', LTrain, LTest,param.top_K);

%         evaluation_info.testT=toc;
%         eva{chunki} = evaluation_info;
%         clear evaluation_info   
    end

    eva_info{chunki,1}=evaluate_perf(param.top_K,B,XTest*XW,YTest*YW,LTrain,LTest);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if EXPparam.coarse_retrieval==true
        % Wy: 4945 x r
        % BxTrain: ntrain x r
        % L1Train: ntrain x 4
    
        coarse_query=zeros(1,4945);
        %     extra_query(108)=1;     % black
        %     extra_query(1900)=1;    % striped
        coarse_query(1224)=1;    % bag
        [eva_info{chunki,1}.precisionK_text2image_coarse, eva_info{chunki,1}.recallK_text2image_coarse] ...
            = perform_coarse_retrieval(EXPparam,coarse_query*YW,B,L1Train);
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%      plot_parentclass_heatmap(B',NL1,EXPparam);
%      plot_childclass_heatmap(B',NL2,EXPparam);



end
