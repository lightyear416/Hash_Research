function [param,train,query] = load_dataset(param)

if ~isfield(param,'ds_name')
%     param.ds_name='Ssense';
    param.ds_name='FashionVC';
end

if ~isfield(param,'train_size')
    param.train_size=20000;         % train.size=min(param.train_size,N-nquery)
end

if ~isfield(param,'nquery')
    param.nquery=2000;
end

if ~isfield(param,'chunk_size')
    param.chunk_size=2000;
end

if ~isfield(param,'top_K')
    param.top_K=1000;
end

if strcmp(param.ds_name,'Ssense')
    param.num_class1=4;
    param.num_class2=28;
elseif strcmp(param.ds_name,'FashionVC')
    param.num_class1=8;
    param.num_class2=27;
else
    error('DATASET NAME: ERROR!\n');
end


fprintf('LOAD DATASET: %s\n',param.ds_name);
if strcmp(param.ds_name,'Ssense')
    load(fullfile(param.ds_dir,[param.ds_name,'.mat']));
    Y=Tag;              clear Tag
    L=Label;            clear Label
    if strcmp(param.feature,'VGG')
        X=Image_vgg16;      clear Image_vgg16
    elseif strcmp(param.feature,'GIST')
        load('D:\workspace\datasets\Ssense\gist.mat')
        X=gist;             clear gist
    elseif strcmp(param.feature,'GIST-4096')
        load('D:\workspace\datasets\Ssense\gist-4096.mat')
        X=gist;             clear gist
    end

elseif strcmp(param.ds_name,'FashionVC')
    load(fullfile(param.ds_dir,[param.ds_name,'.mat']));
    Y=Tag;              clear Tag 
    L=Label;            clear Label
    if strcmp(param.feature,'VGG')
        X=Image_vgg16;      clear Image_vgg16
    elseif strcmp(param.feature,'GIST')
        load('D:\workspace\datasets\FashionVC\gist.mat')
        X=gist;             clear gist
    elseif strcmp(param.feature,'GIST-4096')
        load('D:\workspace\datasets\FashionVC\gist-4096.mat')
        X=gist;             clear gist
    end
end

[param, train, query] = split_dataset(X,Y,L,param);
train.A1_2=A1_2;        clear A1_2
[~,param.dx]=size(X);
[~,param.dy]=size(Y);
end

