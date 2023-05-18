function [streamdata,nstream,L2_tr,I_tr,T_tr] = predata_stream_OSCMFH(I_tr,T_tr,L2_tr,L1_tr,I_te,T_te,numbatch,EXPparam)
%rand('seed',1);
anchors = 500;
anchor_idx = randsample(size(I_tr,1), anchors);
XAnchors = I_tr(anchor_idx,:);
anchor_idx = randsample(size(T_tr,1), anchors);
YAnchors = T_tr(anchor_idx,:);
[I_tr_non,I_te_non]=Kernel_Feature(I_tr,I_te,XAnchors);
[T_tr_non,T_te_non]=Kernel_Feature(T_tr,T_te,YAnchors);

[ndata,~] = size(I_tr);
% Rdata = randperm(ndata);
% I_tr = I_tr(Rdata,:);
% T_tr = T_tr(Rdata,:);
% L_tr = L_tr(Rdata,:);

% I_tr_non = I_tr_non(Rdata,:);
% T_tr_non = T_tr_non(Rdata,:);

% nstream = ceil(ndata/numbatch);
nstream = floor(ndata/numbatch);
streamdata = cell(3,nstream);
% streamdata_non = cell(3,nstream);
for i = 1:nstream-1
    start = (i-1)*numbatch+1;
    endl = i*numbatch;
    streamdata{1,i} = I_tr(start:endl,:);
    streamdata{2,i} = T_tr(start:endl,:);
    if strcmp(EXPparam.avai_labels,'all')
        % feed all labels
        streamdata{3,i} = [L1_tr(start:endl,:) L2_tr(start:endl,:)];         %feed all labels
    else
        % feed fine-grained labels
        streamdata{3,i} = L2_tr(start:endl,:);         %feed fine-grained labels
    end

    %     streamdata_non{1,i} = I_tr_non(start:endl,:);
    %     streamdata_non{2,i} = T_tr_non(start:endl,:);
    %     streamdata_non{3,i} = L_tr(start:endl,:);
end
start = (nstream-1)*numbatch+1;
streamdata{1,nstream} = I_tr(start:end,:);
streamdata{2,nstream} = T_tr(start:end,:);

if strcmp(EXPparam.avai_labels,'all')
    % feed all labels
    streamdata{3,nstream} = [L1_tr(start:end,:) L2_tr(start:end,:)];         %feed all labels
else
    % feed fine-grained labels
    streamdata{3,nstream} = [L2_tr(start:end,:)];         %feed fine-grained labels
end

% streamdata_non{1,nstream} = I_tr_non(start:end,:);
% streamdata_non{2,nstream} = T_tr_non(start:end,:);
% streamdata_non{3,nstream} = L_tr(start:end,:);

