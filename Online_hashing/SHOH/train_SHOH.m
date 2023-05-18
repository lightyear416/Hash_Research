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
alpha2=param.alpha2;
gamma=param.gamma;
xi=param.xi;
mu=param.mu;
max_iter = param.max_iter;


X_new=train.X(idx,:)';          % image feature
Y_new=train.Y(idx,:)';          % text feature
L1_new=train.L1(idx,:);         % label at the 1st layer
L2_new=train.L2(idx,:);         % label at the 2nd layer
A1_2=train.A1_2;                % cross-layer affiliation

if first_round==true
    train.B=[];
    train.trained=0;
    train.time.train_time=[];
    train.cnt1=zeros(num_class1,1);
    train.cnt2=zeros(num_class2,1);

    
    %% initialize tempory variables
    T=cell(1,2);
    

    % D2 = B_old * S2_old + B_new * S2_new
    T{1,1} = zeros(r,num_class2);
    
    % D1 = B_old * S1_old + B_new * S1_new
    T{1,2} = zeros(r,num_class1);

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


    T{1,11}=zeros(dx,num_class1);
    T{1,12}=zeros(dx,num_class2);
    T{1,13}=zeros(dy,num_class1);
    T{1,14}=zeros(dy,num_class2);


    %% initialize C1, C2 randomly
    C1=sign(randn(r,num_class1));C1(C1==0)=-1;
    C2=sign(randn(r,num_class2));C2(C2==0)=-1;

%     A1_2_=NormalizeFea(A1_2);
%     A1_2_=A1_2;
% 
%     hd1=(r-C1'*C1)/2;
%     heatmap(hd1);
    
%     for ii=1:5
        %% C2
%         Q2=r*(eta*C1*A1_2_);
%         for l=1:r
%             idx_exc=setdiff(1:r,l);
%             q2=Q2(l,:);
%             c1=C1(l,:);C1_=C1(idx_exc,:);
%             C2_=C2(idx_exc,:);
% 
%             c2=sign(q2-eta*c1*C1_'*C2_);c2(c2==0)=-1;
%             C2(l,:)=c2;    
%         end 

        %% C1-step
%             Q1=r*(eta*C2*A1_2_');
%             for l=1:r
%                 idx_exc=setdiff(1:r,l);
%                 C1_ = C1(idx_exc,:);
%                 c2 = C2(l,:); C2_ = C2(idx_exc,:);
%                 q1 = Q1(l,:);
%         
%                 c1=sign(q1-eta*c2*C2_'*C1_);c1(c1==0)=-1;
%                 C1(l,:)=c1;
%             end
% 
%         hd2=(r-C2'*C2)/2;
%         heatmap(hd2);
%     end

else
   
    T=train.T;

    C1=train.C1;
    C2=train.C2;

    

end

%% initialize B_new randomly
B_new=randn(r,n_t);B_new(B_new==0)=-1;
% A1_2_=NormalizeFea(A1_2);
A1_2_=A1_2;


%% calculate soft similarity labels


% 

if strcmp(param.similar_labels,'soft')
    S1_new=L1_new;
    V2_new=L1_new*A1_2+train.L2(idx,:);
    S2_new=zeros(size(L2_new));
    for row=1:n_t
        S2_new(row,:)=V2_new(row,:)/norm(V2_new(row,:))+gamma*L2_new(row,:);
    end
elseif strcmp(param.similar_labels,'hard')
    S1_new=L1_new;
    S2_new=L2_new;
end



% time.Wx=[];
% time.Wy=[];
% time.C1=[];
% time.C2=[];
% time.B_new=[];


%% hash code learning
for i = 1:max_iter
    %% B_new-step
%     tic;
    P=r*alpha1*C1*S1_new'+r*alpha2*C2*S2_new';
    for l=1:r
        idx_exc=setdiff(1:r,l);
        p=P(l,:);
        c1=C1(l,:); C1_ = C1(idx_exc,:);
        c2=C2(l,:); C2_ = C2(idx_exc,:);
        B_new_=B_new(idx_exc,:);

        b_new=sign(p-(alpha1*c1*C1_'+alpha2*c2*C2_')*B_new_);b_new(b_new==0)=-1;
        B_new(l,:)=b_new;
    end
%     time.B_new=[time.B_new toc];


    %% C2-step
%     tic;
    Q2=r*(alpha2*(B_new*S2_new+T{1,1})+eta*C1*A1_2_);

    for l=1:r
        idx_exc=setdiff(1:r,l);
        q2=Q2(l,:);
        b_new=B_new(l,:);B_new_=B_new(idx_exc,:);
        c1=C1(l,:);C1_=C1(idx_exc,:);
        C2_=C2(idx_exc,:);

        c2=sign(q2-(alpha2*b_new*B_new_'+alpha2*T{1,3}(l,:)+eta*c1*C1_')*C2_);c2(c2==0)=-1;
        C2(l,:)=c2;
    end
%     time.C2=[time.C2 toc];

    %% C1-step
%     tic;
    Q1=r*(alpha1*(B_new*S1_new+T{1,2})+eta*C2*A1_2_');
    for l=1:r
        idx_exc=setdiff(1:r,l);
        b_new = B_new(l,:); B_new_ = B_new(idx_exc,:);
        C1_ = C1(idx_exc,:);
        c2 = C2(l,:); C2_ = C2(idx_exc,:);
        q1 = Q1(l,:);

        c1=sign(q1-(alpha1*b_new*B_new_'+alpha1*T{1,3}(l,:)+eta*c2*C2_')*C1_);c1(c1==0)=-1;
        C1(l,:)=c1;
    end
%     time.C1=[time.C1 toc];


    
end


train.B = [train.B B_new];


T{1,1} = T{1,1} + B_new*S2_new;
T{1,2} = T{1,2} + B_new*S1_new;

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
% Wx=(T{1,4})/(T{1,6}+xi*eye(dx,dx)); % +0.001*eye(dx,dx)


for i=1:num_class1
    ddx=find(L1_new(:,i)==1);
    cnt=numel(ddx);
    if cnt>0
        T{1,11}(:,i)= (T{1,11}(:,i)*train.cnt1(i)+sum(X_new(:,ddx),2))./(train.cnt1(i)+numel(ddx));
        T{1,13}(:,i)= (T{1,13}(:,i)*train.cnt1(i)+sum(Y_new(:,ddx),2))./(train.cnt1(i)+numel(ddx));
    %     T{1,11}(:,i)= T{1,11}(:,i)+sum(X_new(:,ddx),2);
        train.cnt1(i)=train.cnt1(i)+numel(ddx);
    end
end

for i=1:num_class2
    ddx=find(L2_new(:,i)==1);
    cnt=numel(ddx);
    if cnt>0
        T{1,12}(:,i)=(T{1,12}(:,i)*train.cnt2(i)+sum(X_new(:,ddx),2))./(train.cnt2(i)+numel(ddx));
        T{1,14}(:,i)= (T{1,14}(:,i)*train.cnt2(i)+sum(Y_new(:,ddx),2))./(train.cnt2(i)+numel(ddx));
%     T{1,12}(:,i)=T{1,12}(:,i)+sum(X_new(:,ddx),2);
        train.cnt2(i)=train.cnt2(i)+numel(ddx);
    end
end
% alpha1=0;alpha2=1;
% alpha1=alpha1*10000;alpha2=alpha2*10000;
Wx=(T{1,4}+mu*alpha1*C1*T{1,11}'+mu*alpha2*C2*T{1,12}')/(T{1,6}+mu*alpha1*T{1,11}*T{1,11}'+mu*alpha2*T{1,12}*T{1,12}'+xi*eye(dx,dx));


% Wx=(C2*T{1,12}')/(T{1,6}+T{1,12}*T{1,12}'+eye(param.dx,param.dx));

% L1=L1_new;L2=L2_new;
% FA=C1*C1';
% FB=T{1,11}*T{1,11}';
% FC=C2*C2';
% FD=T{1,12}*T{1,12}';
% FE=alpha1*C1*(2*L1'*L1-ones(num_class1,num_class1))*T{1,11}'+alpha2*C2*(2*L2'*L2-ones(num_class2,num_class2))*T{1,12}';
% Wx=lyap(alpha1*(FC\FA),alpha2*(FD/FB),-FC\FE/FB);


% Wx=(eye(n_t,n_t)+alpha1*(B_new'*B_new))\B_new/((eyes(n_t,n_t)+alpha1*r*V_new*V_new')*X_new'*(X_new*X_new'));

% time.Wx=[time.Wx toc];
% for kk =1 :10
%     Wx=randn(r,dx);
%     T0=tanh(pi*Wx*X_new);
%     Wx=Wx-0.001*(2*pi*(T0-B_new).*(ones(r,n_t)-T0.^2)*X_new'+2*Wx);
% end


% tic;
% Wy=(T{1,5})/(T{1,7}+eye(param.dy,param.dy));
% Wy=(T{1,5})/(T{1,7}+xi*eye(dy,dy));
Wy=(T{1,5}+mu*alpha1*C1*T{1,13}'+mu*alpha2*C2*T{1,14}')/(T{1,7}+mu*alpha1*T{1,13}*T{1,13}'+mu*alpha2*T{1,14}*T{1,14}'+xi*eye(dy,dy));
% time.Wy=[time.Wy toc];


% unmatched=abs(train.B-sign(Wx*train.X(1:train.trained+n_t,:)'));ss=sum(unmatched,'all');
% fprintf('\n%d',ss);



%% return variables
train.T=T;
train.Wx=Wx;
train.Wy=Wy;
train.C1=C1;
train.C2=C2;

train.trained=train.trained+n_t;
% show_childclass_HD(train.B,train.NL2(1:train.trained),param);
% train.time.Wx=[train.time.Wx;time.Wx];
% train.time.Wy=[train.time.Wy;time.Wy];
% train.time.C1=[train.time.C1;time.C2];
% train.time.C2=[train.time.C2;time.C2];
% train.time.B_new=[train.time.B_new;time.B_new];
train.time.train_time=[train.time.train_time;toc];

end


