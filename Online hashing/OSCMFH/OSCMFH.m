
% clc;clear 
% load mirflickr25k.mat
%% Parameter setting
function OSCMFH(EXPparam,train,query)

    I_tr=train.X; 
%     I_tr=pca_X(query.size+1:end,:);
    T_tr=train.Y; L_tr=train.L2; L1_tr=train.L1;
    I_te=query.X; 
%     I_te=pca_X(1:query.size,:);
    T_te=query.Y; L_te=query.L2; L1_te=query.L1;
    run = 1;
%     Bits = [16,32,64,128];
    Bits=EXPparam.nbits;
    %% Preprocessing data 
    numbatch = EXPparam.chunk_size;
    %[streamdata,streamdata_non,nstream,L_tr,I_tr,T_tr,I_tr_non,T_tr_non,I_te_non,T_te_non] = predata_stream(I_tr,T_tr,L_tr,I_te,T_te,numbatch);
    %% feed all label
    [streamdata,~,~,~,~] = predata_stream_OSCMFH(I_tr,T_tr,L_tr,L1_tr,I_te,T_te,numbatch,EXPparam);
    for i = 1:length(Bits)
        for j=1:run
    
           %% --------------------OURS----------------------------%%
            % [eva_info,time] = main_OSCMFH(streamdata, I_te, T_te, Bits(i));
            [eva_info,time] = main_OSCMFH(streamdata,L1_tr,I_te, T_te, L_te, L1_te, Bits(i),EXPparam,train.NL1,train.NL2);
            
%             Dhamm = hammingDist(tB_I, B_T)';    
%             [~, HammingRank]=sort(Dhamm,1);
%             mapIT = map_rank(L_tr,L_te,HammingRank); 
%             Dhamm = hammingDist(tB_T, B_I)';    
%             [~, HammingRank]=sort(Dhamm,1);
%             mapTI = map_rank(L_tr,L_te,HammingRank); 
%             map(j, 1) = mapIT(100);
%             map(j, 2) = mapTI(100);



            %-----------------------save record---------------------------------------%
            if ~isfield(EXPparam,'rec_dir')
                record_dir=fullfile(pwd,'records',EXPparam.hash_func,EXPparam.ds_name,EXPparam.samp_method);
            else
                record_dir=fullfile(EXPparam.rec_dir,EXPparam.hash_method,EXPparam.ds_name,EXPparam.samp_method);
            end
            if(~exist(record_dir,'dir'))
                mkdir(record_dir);
            end


            if EXPparam.opti_param==true
                record_name=['test', num2str(EXPparam.t), ...
                    '_lambda=',num2str(lambda), ...
                    '_mu=',num2str(mu), ...
                    '_gamma=',num2str(gamma), ...
                    '_iter=',num2str(iter),...
                    '_cmfhiter=',num2str(cmfhiter)];
            else
                record_name=['test', num2str(EXPparam.t), ...
                    '_TrainSize=',num2str(train.size), ...
                    '_QuerySize=',num2str(query.size), ...
                    '_ChunkSize=',num2str(EXPparam.chunk_size), ...
                    '_NumBits=',num2str(EXPparam.nbits), ...
                    '_', EXPparam.avai_labels, ...
                    '.mat'];
            end


            save(fullfile(record_dir,record_name),'EXPparam','eva_info','time','-v7.3');
            
        end
%     fprintf('\nbits = %d\n', Bits(i));
%     fprintf('average map over %d runs for ImageQueryOnTextDB: %.4f, chunks_size:%d\n', run, mean(map( : , 1)),numbatch);
%     fprintf('average map over %d runs for TextQueryOnImageDB: %.4f, chunks_size:%d\n', run, mean(map( : , 2)),numbatch);
    end
end
