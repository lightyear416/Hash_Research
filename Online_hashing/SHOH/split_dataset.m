function [param, train, query] = split_dataset(X, Y, L, param)
% X: original features
% Y: original text
% L: original labels
% param.nquery: # the number of test points
num_class1=param.num_class1;
num_class2=param.num_class2;


[N, ~] = size(L);
label1=zeros(N,1);
label2=zeros(N,1);


%% convert {0,1} to class index
for i = 1:num_class1
        idx=find(L(:,i)==1);
        label1(idx)=i;
end
for j = 1:num_class2
        % find examples in this class, randomize ordering
        idx = find(L(:,j+num_class1) == 1);
        label2(idx)=j;
end

%% construct test and train set
rng shuffle
param.seed=rng;
R = randperm(N);
nquery = param.nquery;
ntrain = N - nquery;
iquery = R(1:nquery);
itrain = R(nquery+1:N);



% randomize again
itrain = itrain(randperm(ntrain));
iquery = iquery(randperm(nquery));

param.train_idx=itrain;
% query.query_idx=iquery;

query.X=X(iquery,:);
query.Y=Y(iquery,:);
query.NL1=label1(iquery);
query.NL2=label2(iquery);
query.L1=L(iquery,1:num_class1);
query.L2=L(iquery,num_class1+1:end);
query.size=nquery;

train.X=X(itrain,:);
train.Y=Y(itrain,:);
train.NL1=label1(itrain);
train.NL2=label2(itrain);
train.L1=L(itrain,1:num_class1);
train.L2=L(itrain,num_class1+1:end);
train.size=min(param.train_size,N-nquery);

end



