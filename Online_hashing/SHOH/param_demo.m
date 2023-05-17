close all; clear; clc;
warning off
hash_method={'SHOH'};           %'OCMFH' ,'LEMON','DOCH','OSCMFH'
% avai_labels={'all','fine'};
% hash_method={'OSCMFH'};
addpath(genpath('D:/workspace/matlab/SHOH-master/SHOH'));       % SHOH path
param.feature='GIST'
param.avai_labels='all';
param.similar_labels='soft';
param.hierarchy='hie';
param.ds_dir = 'D:/workspace/datasets/';


% param.rec_dir = 'D:/workspace/matlab/SHOH-master/param_tuning_chi2_dece/alpha';
param.rec_dir = 'D:/workspace/matlab/SHOH-master/a_param_GIST_WO_norm/xi';
ds_name='FashionVC';                             % 'Ssense'  'FashionVC' 'MIRFlickr-25K' 'NUSWIDE10'
                          % 'VGG' 'GIST'
samp_method={'random'};                         % uniform, random 
nbits=[16 32 64 128];                           %  16 
chunk_size=2000;
param.nquery=2000;
param.opti_param=true;
param.coarse_retrieval=false;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
seq_cmp=true;
test_times=5;
max_iter=[7];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%-----------------------------hyperparameters-----------------------------%

% max_iter=[1,2,3,4,5];
% normalizeX=[1];
% alpha1=[linspace(0,1,11)];
% variable=sort(alpha1);
% 
% eta=[0.0001 0.001 0.01 0.1 1 10 100 1000 10000];
% variable=sort(eta);
% 
% gamma=[linspace(0,1,11)];
% variable=sort(gamma);

xi=[0.0001 0.001 0.01 0.1 1 10 100 1000 10000];
variable=sort(xi);


% mu=[0.00001];   %linspace(0.1,1,10) 0.00001 
% mu=[0 1 10 100 1000 10000 100000];
% mu=2000;
% variable=sort(mu);
%-------------------------------------------------------------------------%

for t=1:test_times
    param.t=t;
    % SAMPLING METHOD
    for sm=1:length(samp_method)
        param.samp_method=samp_method{sm};
        param.ds_name=ds_name;
        [param,train,query]=load_dataset(param);
        % LENGTH OF HASH CODES
        for nb=1:length(nbits)
            param.nbits=nbits(nb);
            for mi=1:size(max_iter,2)
                param.max_iter=max_iter(mi);
                for i=1:size(variable,2)
                        if exist('alpha1','var')
                            param.alpha1=variable(i);
                            param.alpha2=1-variable(i);
                        elseif exist('eta','var')
                            param.eta=variable(i);
                        elseif exist('gamma','var')
                            param.gamma=variable(i);
                        elseif exist('xi','var')
                            param.xi=variable(i);
                        elseif exist('mu','var')
                            param.mu=variable(i);
                        end

                        SHOH(param,train,query);
                end
            end
        end
    end
end