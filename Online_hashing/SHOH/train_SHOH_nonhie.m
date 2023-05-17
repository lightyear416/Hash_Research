function train = train_SHOH(param,train,idx,first_round)
tic;
r = param.nbits;
num_class1=param.num_class1;
num_class2=param.num_class2;
dx=size(train.X,2);
dy=size(train.Y,2);
n_t=numel(idx);

%% hyperparameters
eta=param.eta;
alpha1=param.alpha1;
gamma=param.gamma;
alpha2=1;
mu=param.mu;
max_iter = param.max_iter;


X_new=train.X(idx,:)';          % image feature
Y_new=train.Y(idx,:)';          % text feature
L1_new=train.L1(idx,:);         % label at the 1st layer
L2_new=train.L2(idx,:);         % label at the 2nd layer
if strcmp(param.avai_labels,'fine')
    L_new=L2_new;
    num_class=num_class2;
elseif strcmp(param.avai_labels,'all')
    L_new=[L1_new L2_new];
    num_class=num_class1+num_class2;
end
% A1_2=train.A1_2;                % cross-layer affiliation

if first_round==true
    train.B=[];
    train.trained=0;
    train.time.train_time=[];
    train.cnt=zeros(num_class,1);

    
    %% initialize tempory variables
    T=cell(1,2);
    

    % D2 = B_old * S2_old + B_new * S2_new
    T{1,1} = zeros(r,num_class);
    
    % D1 = B_old * S1_old + B_new * S1_new
%     T{1,2} = zeros(r,num_class1);

    % E = b_old * B_old_' + b_new * B_new_'
    T{1,3} = zeros(r,r-1);
    
    % F1 = B_old * X_old' + B_new * X_new'
%     T{1,4} = zeros(r,param.dx);
    T{1,4} = zeros(r,dx);
    
    % F2 = B_old * Y_old' + B_new * Y_new'
%     T{1,5} = zeros(r,param.dy);
    T{1,5} = zeros(r,dy);

    % G1 = X_old * X_old' + X_new * X_new'
%     T{1,6} = zeros(param.dx,param.dx);
    T{1,6} = zeros(dx,dx);
    
    % G2 = Y_old * Y_old' + Y_new * Y_new'
%     T{1,7} = zeros(param.dy,param.dy);
    T{1,7} = zeros(dy,dy);

    
%     T{1,11}=zeros(dx,num_class1);
    T{1,12}=zeros(dx,num_class);
%     T{1,13}=zeros(dy,num_class1);
    T{1,14}=zeros(dy,num_class);

    %% initialize C1, C2 randomly
    % C1=sign(randn(r,num_class1));C1(C1==0)=-1;
    C=sign(randn(r,num_class));C(C==0)=-1;
else
   
    T=train.T;

%     C1=train.C1;
    C=train.C;

    

end

%% initialize B_new randomly
B_new=randn(r,n_t);B_new(B_new==0)=-1;

% time.Wx=[];
% time.Wy=[];
% time.C1=[];
% time.C2=[];
% time.B_new=[];


%% hash code learning
for i = 1:max_iter
    %% B_new-step
%     tic;
%     P=r*alpha1*C1*S1_new'+r*alpha2*C2*S2_new';
    P=r*alpha2*C*L_new';
    for l=1:r
        idx_exc=setdiff(1:r,l);
        p=P(l,:);
%         c1=C1(l,:); C1_ = C1(idx_exc,:);
        c=C(l,:); C_ = C(idx_exc,:);
        B_new_=B_new(idx_exc,:);

%         b_new=sign(p-(alpha1*c1*C1_'+alpha2*c2*C2_')*B_new_);
        b_new=sign(p-alpha2*c*C_'*B_new_);b_new(b_new==0)=-1;
        B_new(l,:)=b_new;
    end

%     time.B_new=[time.B_new toc];


    %% C2-step
%     tic;
%     Q2=r*(alpha2*(B_new*S2_new+T{1,1})+eta*C1*A1_2);
    Q2=r*(alpha2*(B_new*L_new+T{1,1}));

    for l=1:r
        idx_exc=setdiff(1:r,l);
        q2=Q2(l,:);
        b_new=B_new(l,:);B_new_=B_new(idx_exc,:);
%         c1=C1(l,:);C1_=C1(idx_exc,:);
        C_=C(idx_exc,:);

%         c2=sign(q2-(alpha2*b_new*B_new_'+alpha2*T{1,3}(l,:)+eta*c1*C1_')*C2_);
        c=sign(q2-(alpha2*b_new*B_new_'+alpha2*T{1,3}(l,:))*C_);c(c==0)=-1;
        C(l,:)=c;
    end
%     time.C2=[time.C2 toc];

    %% C1-step
%     tic;
%     Q1=r*(alpha1*(B_new*S1_new+T{1,2})+eta*C2*A1_2');
%     for l=1:r
%         idx_exc=setdiff(1:r,l);
%         b_new = B_new(l,:); B_new_ = B_new(idx_exc,:);
%         C1_ = C1(idx_exc,:);
%         c2 = C2(l,:); C2_ = C2(idx_exc,:);
%         q1 = Q1(l,:);
% 
%         c1=sign(q1-(alpha1*b_new*B_new_'+alpha1*T{1,3}(l,:)+eta*c2*C2_')*C1_);
%         C1(l,:)=c1;
%     end
%     time.C1=[time.C1 toc];

end


train.B = [train.B B_new];


T{1,1} = T{1,1} + B_new*L_new;
% T{1,2} = T{1,2} + B_new*S1_new;

for l=1:r
    idx_exc=setdiff(1:r,l);
    b=B_new(l,:);
    B_=B_new(idx_exc,:);
    T{1,3}(l,:) = T{1,3}(l,:)+b*B_';
end
T{1,4} = T{1,4} + B_new*X_new';
T{1,5} = T{1,5} + B_new*Y_new';
T{1,6} = T{1,6} + X_new*X_new';
T{1,7} = T{1,7} + Y_new*Y_new';


%% hash function learning
% tic;
% Wx=(T{1,4})/(T{1,6}+eye(param.dx,param.dx));
% Wx=(T{1,4})/(T{1,6}+eye(dx,dx));
% time.Wx=[time.Wx toc];

for i=1:num_class
    ddx=find(L_new(:,i)==1);
    cnt=numel(ddx);
    if cnt>0
        T{1,12}(:,i)= (T{1,12}(:,i)*train.cnt(i)+sum(X_new(:,ddx),2))./(train.cnt(i)+numel(ddx));
        T{1,14}(:,i)= (T{1,14}(:,i)*train.cnt(i)+sum(Y_new(:,ddx),2))./(train.cnt(i)+numel(ddx));
    %     T{1,11}(:,i)= T{1,11}(:,i)+sum(X_new(:,ddx),2);
        train.cnt(i)=train.cnt(i)+numel(ddx);
    end
end

Wx=(T{1,4}+mu*alpha2*C*T{1,12}')/(T{1,6}+mu*alpha2*T{1,12}*T{1,12}'+eye(dx,dx));



% tic;
% Wy=(T{1,5})/(T{1,7}+eye(param.dy,param.dy));
% Wy=(T{1,5})/(T{1,7}+eye(dy,dy));
% time.Wy=[time.Wy toc];

Wy=(T{1,5}+mu*alpha2*C*T{1,14}')/(T{1,7}+mu*alpha2*T{1,14}*T{1,14}'+eye(dy,dy));





%% return variables
train.T=T;
train.Wx=Wx;
train.Wy=Wy;
% train.C1=C1;
train.C=C;

train.trained=train.trained+n_t;

% train.time.Wx=[train.time.Wx;time.Wx];
% train.time.Wy=[train.time.Wy;time.Wy];
% train.time.C1=[train.time.C1;time.C2];
% train.time.C2=[train.time.C2;time.C2];
% train.time.B_new=[train.time.B_new;time.B_new];
train.time.train_time=[train.time.train_time;toc];

end


