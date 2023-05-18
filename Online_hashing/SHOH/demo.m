close all; clear; clc;

hash_method={'SHOH'};           %,'OCMFH', 'LEMON','DOCH','OSCMFH'
avai_labels={'all','fine'};
% hash_method={'OSCMFH'};
addpath(genpath('D:/workspace/matlab/SHOH-master/SHOH'));       % SHOH path
addpath(genpath('D:/workspace/matlab/SHOH-master/LEMON'));      % LEMON path
addpath(genpath('D:/workspace/matlab/SHOH-master/DOCH'));       % DOCH path
addpath(genpath('D:/workspace/matlab/SHOH-master/OSCMFH'));     % OSCMFH path
addpath(genpath('D:/workspace/matlab/SHOH-master/OCMFH'));      % OCMFH path


param.ds_dir = 'D:/workspace/datasets/';


param.rec_dir = 'D:/workspace/matlab/SHOH-master/record';
ds_name={'Ssense'};                             % 'Ssense'  'FashionVC' 'MIRFlickr-25K' 'NUSWIDE10'
samp_method={'random'};                         % uniform, random 
nbits=[16 32 64 128];
chunk_size=2000;
param.nquery=2000;
param.opti_param=false;
param.coarse_retrieval=true;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
seq_cmp=true;
test_times=1;
max_iter=[7];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% spmd
for t=1:test_times
    param.t=t;
    % DATASET
    for ds=1:length(ds_name)
        param.ds_name=ds_name{ds};
        % SAMPLING METHOD
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
                            fprintf(['DATASET: ' param.ds_name '\n' ...
                                'SAMPLING METHOD: ' param.samp_method '\n' ...
                                'LENGTH OF HASH CODES: ' num2str(param.nbits) '\n' ...
                                'CHUNK SIZE: ' num2str(param.chunk_size) '\n' ...
                                'HASH METHOD: ' hash_method{hm} '\n']);
                            switch hash_method{hm}
                                case 'SHOH'
                                    for mi=1:length(max_iter)
                                        param.max_iter=max_iter(mi);
                                        SHOH(param,train,query);
                                    end
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