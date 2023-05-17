function [eva_info,time]=evaluate_DOCH(XChunk,YChunk,L2Chunk,XTest,YTest,L2Test,param, ...
    L1Chunk,L1Test,EXPparam,NL1,NL2)
    eva_info=cell(param.nchunks,1);
    LChunk=cell(param.nchunks,1);
    
    if strcmp(EXPparam.avai_labels,'all')
        % feed all labels
        for chunki = 1:param.nchunks
            LChunk{chunki,1}=[L1Chunk{chunki,1} L2Chunk{chunki,1} ];    
        end
    else
        % feed fine-grained labels
        LChunk=L2Chunk;                                     
    end
    
    for chunki = 1:param.nchunks
        L1Train = cell2mat(L1Chunk(1:chunki,:));                
        L2Train = cell2mat(L2Chunk(1:chunki,:));
        XTrain_new = XChunk{chunki,:};
        YTrain_new = YChunk{chunki,:};
        LTrain_new = LChunk{chunki,:};
        
        % Hash code learning  

        if chunki == 1
%             tic;
            [Wx,Wy,BB,MM,time] = train0(XTrain_new,YTrain_new,param,LTrain_new);
%             time.train_time=toc;
        else
%             tic;
            [Wx,Wy,BB,MM,time] = train(XTrain_new,YTrain_new,param,LChunk,BB,MM,chunki,time);
%             time.train_time=[time.train_time;toc];
        end
        
        B = cell2mat(BB(1:end,1));

%         eva_info{chunki,1}=evaluate_perf(param.top_K,B,XTest*Wx,YTest*Wy,L2Train,L2Test);

        clear evaluation_info
    end
    eva_info{chunki,1}=evaluate_perf(param.top_K,B,XTest*Wx,YTest*Wy,L2Train,L2Test);
    fprintf('MAP in I->T: %.4g\n',eva_info{param.nchunks,1}.map_image2text);
    fprintf('MAP in T->I: %.4g\n',eva_info{param.nchunks,1}.map_text2image);

    if EXPparam.coarse_retrieval==true
        % Wy: 4945 x r
        % BxTrain: ntrain x r
        % L1Train: ntrain x 4
    
        coarse_query=zeros(1,4945);
        %     extra_query(108)=1;     % black
        %     extra_query(1900)=1;    % striped
        coarse_query(1224)=1;    % bag
        [eva_info{chunki,1}.precisionK_text2image_coarse, eva_info{chunki,1}.recallK_text2image_coarse] ...
            = perform_coarse_retrieval(EXPparam,coarse_query*Wy,B,L1Train);
    end

%     plot_parentclass_heatmap(B',NL1,EXPparam);
%     plot_childclass_heatmap(B',NL2,EXPparam);

    fprintf('\n');
end
