close all; clear; clc;
warning off
hash_method={'SHOH'};           % ,'LEMON','DOCH','OSCMFH'
avai_labels={'all','fine'};                                   %'all','fine'
similar_labels={'soft','hard'};                         % 
hierarchy={'hie','non-hie'};                            % ,'non-hie'
% hash_method={'OSCMFH'};
addpath(genpath('D:/workspace/matlab/SHOH-master/methods'));
param.feature='GIST';                    % 'VGG'     'GIST'      'GIST-4096'


param.ds_dir = 'D:/workspace/datasets/';

 

ds_name={'FashionVC'};                             % 'Ssense'  'FashionVC' 'MIRFlickr-25K' 'NUSWIDE10'
samp_method={'random'};                         % uniform, random 
nbits=[64 128];                                    % 16 32 64 128
chunk_size=2000;
param.nquery=2000;
param.opti_param=false;
% param.rec_dir = 'D:/workspace/matlab/SHOH-master/record-cmp-chi2_kernel';
% param.rec_dir = 'D:/workspace/matlab/SHOH-master/all_cmp_WO_norm_derivatives';
param.rec_dir = 'D:/workspace/matlab/SHOH-master/tt';
param.coarse_retrieval=false;



if strcmp(ds_name,'FashionVC')
    param.coarse_retrieval=false; 
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
seq_cmp=true;
test_times=5
max_iter=[7];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% spmd
for t=1:test_times
    param.t=t;
    % DATASET
    for ds=1:length(ds_name)
        param.ds_name=ds_name{ds};
        % SAMPLING METHO
        for sm=1:length(samp_method)
            param.samp_method=samp_method{sm};
            [param,train,query]=load_dataset(param);
            % LENGTH OF HASH CODES
            for nb=1:length(nbits)
                param.nbits=nbits(nb);
                %CHUNK SIZE
                for cs=1:length(chunk_size)
                    param.chunk_size=chunk_size(cs);
                    for al=1:length(avai_labels)
                        param.avai_labels=avai_labels{al};
                    % HASH METHOD
                        for hm=1:length(hash_method)
                            param.hash_method=hash_method{hm};
%                             fprintf(['DATASET: ' param.ds_name '\n' ...
%                                 'SAMPLING METHOD: ' param.samp_method '\n' ...
%                                 'LENGTH OF HASH CODES: ' num2str(param.nbits) '\n' ...
%                                 'CHUNK SIZE: ' num2str(param.chunk_size) '\n' ...
%                                 'HASH METHOD: ' hash_method{hm} '\n']);
                            switch hash_method{hm}
                                case 'SHOH'
                                        if strcmp(param.avai_labels,'fine')
                                            suffix=param.avai_labels
                                            SHOH(param,train,query);
                                        elseif strcmp(param.avai_labels,'all')
                                            for h=1:length(hierarchy)
                                                param.hierarchy=hierarchy{h};
                                                if strcmp(param.hierarchy,'non-hie')
                                                    suffix=param.avai_labels
                                                    SHOH(param,train,query);
                                                elseif strcmp(param.hierarchy,'hie')
                                                    for sl=1:length(similar_labels)
                                                        param.similar_labels=similar_labels{sl};            % 'soft' 'hard'
                                                        suffix=param.similar_labels
                                                        SHOH(param,train,query);
                                                    end
                                                end
                                            end
%                                             SHOH(param,train,query);                                % 'all' 'fine'
                                        end
                                  
%                                     for mi=1:length(max_iter)
%                                         param.max_iter=max_iter(mi);
%                                         SHOH(param,train,query);
%                                     end
    %                                 end
                                case 'LEMON'
                                    LEMON(param,train,query);
                                case 'DOCH'
                                    DOCH(param,train,query);
                                case 'OCMFH'
                                    OCMFH(param,train,query);
                                case 'OSCMFH'
                                    OSCMFH(param,train,query);
                            end
                        end
                    end
                end
            end
        end
    end
end

%-----------------------------hyperparameters-----------------------------%
% eta=[0.0001 0.001 0.01 0.1 1 linspace(10,100,10) 1000 10000 100000 1000000];
% max_iter=[1,2,3,4,5];
% max_iter=[1];
% normalizeX=[1];
% alpha1=[linspace(0.8,1,21)];
% variable=sort(alpha1);
%-------------------------------------------------------------------------%

% for t=1:test_times
%     param.t=t;
%     [param,train,query]=load_dataset(param);
%     for mi=1:size(max_iter,2)
%         param.max_iter=max_iter(mi);
%         for n=1:size(normalizeX,2)
%             param.normalizeX=normalizeX(n);
%             for i=1:size(variable,2)
%                 if exist('alpha1','var')
%                     param.alpha1=variable(i);
%                     param.alpha2=1-variable(i);
%                 elseif exist('delta','var')
%                     param.delta=variable(i);
%                 elseif exist('lambda','var')
%                     param.lambda=variable(i);
%                 elseif exist('mu','var')
%                     param.mu=variable(i);
%                 elseif exist('eta','var')
%                     param.eta=variable(i);
%                 end
%                 for hm=1:length(hash_method)
%                     switch(hash_method{hm})
%                         case 'SCOTH_hetero'
%                             param.hash_func='hetero';
%                         case 'SCOTH_homo'
%                             param.hash_func='homo';
%                         otherwise
%                             error('hash function: ERROR!\n');
%                     end
%                     SCO4H(param,train,query);
%                 end
%             end
%         end
%     end
% end