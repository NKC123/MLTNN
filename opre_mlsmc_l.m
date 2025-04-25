 % ----------------------------------------------------------- %
% function [sums, cost] = opre_mlsmc_l(l,N,sigma,data,x_data)
% single level routine
% inputs:  l      = level
%          N      = number of samples
%          sigma  = std deviation of the error of y - G
%          data   = values of y
%          x_data = points corresponding to observations
% outputs: sums(1) = SN increments estimator at level l
%          sums(2) = unnormalised increments phi at level l
%          sums(3) = unnormalised increments 1 at level l
%          cost    = computational cost
% ----------------------------------------------------------- %
%function [sums, cost] = opre_mlsmc_l(l,N,K,sigma,data,x_data,s,n,m,D)
function [sums, cost] = opre_mlsmc_l(l,N,K,sigma,data,x_data,s,n,m,D,rel,eps)
y = data;
sums(1:3) = 0;
QOI_c = zeros(N,1);
QOI_f = zeros(N,1);
ESSmin = round(N/2);

% set up the temporing step
lambda = 0;

% initialisation
u = cell(N,1);
Z = 0; %logsetting
ZZ = 0; %logsetting
ZZ_max = 0; %logsetting
x = ones(size(x_data,1),1)*2;

% multi-level increments
%H   = zeros(N,1);
G_k = zeros(N,1);
f_f = zeros(N,1);
f_c = zeros(N,1);


parfor i = 1:N
    [Af_1,bf_1,Af_D,bf_D,Af_d,bf_d] = tnn_prior(l,s,n,m,D);
    u{i} = {Af_1,bf_1,Af_D,bf_D,Af_d,bf_d};
    g_f = tnn_layer(x_data,D,Af_1,bf_1,Af_D,bf_D,Af_d,bf_d);
    l_f_temp = -0.5*(g_f - y)*(g_f - y)'/sigma/sigma;
    if l == K
        g_c = -Inf;
        l_c_temp = -Inf;
    else
        Ac_1 = Af_1(1:2^(l-1),:);
        bc_1 = bf_1(1:2^(l-1),:);
        Ac_D = Af_D(:,1:2^(l-1));
        bc_D = bf_D;
        Ac_d = Af_d(1:2^(l-1),1:2^(l-1),:);
        bc_d = bf_d(1:2^(l-1),:);
        g_c = tnn_layer(x_data,D,Ac_1,bc_1,Ac_D,bc_D,Ac_d,bc_d);
        l_c_temp = -0.5*(g_c - y)*(g_c - y)'/sigma/sigma;
    end
    G_k(i) = max( l_f_temp, l_c_temp );
    f_f(i) = l_f_temp;
    f_c(i) = l_c_temp;
end

beta = 0.4;
r = 1.5;
count = 0;
while lambda < 1
    
    lambdaold = lambda;
    lambda = temstep(lambdaold,G_k,ESSmin);
    
    if count == 0 && lambda == 1
        lambda = 0.5;
    end
    
    count = count + 1;

    % normalising constant
    H = G_k.*(lambda-lambdaold);
    Z = Z + log(mean(exp(H)));
    W = exp( H - max(H) )./sum(exp( H - max(H)) );
    ZZ = ZZ + log(mean(exp(H - max(H))));
    ZZ_max = ZZ_max + max(H);

    % resampling step
    A = Multinomial_Resampling(W);
    u = u(A');

    u_rate = 0;
    parfor k = 1:N
        rate = 0;
        for i = 1:10
            [v,ll_f_temp,ll_c_temp] = mutation(l,u{k},x_data,data,K,s,n,m,D,sigma,beta);
            G_star = max( ll_f_temp, ll_c_temp );
            alpha_temp = lambda*(G_star-G_k(k));
            alpha = min( 0, alpha_temp );
            uni = rand(1);
            if log(uni) < alpha
                u{k} = v;
                G_k(k) = G_star;
                f_f(k) = ll_f_temp;
                f_c(k) = ll_c_temp;
                rate = rate + 1;
            end
        end
           rate = rate/10;
           u_rate = u_rate + rate/N;
        QOI_f(k) = tnn_layer(x,D,u{k}{1},u{k}{2},u{k}{3},u{k}{4},u{k}{5},u{k}{6});
        if l > K
            Af_1 = u{k}{1};
            bf_1 = u{k}{2};
            Af_D = u{k}{3};
            bf_D = u{k}{4};
            Af_d = u{k}{5};
            bf_d = u{k}{6};
            Ac_1 = Af_1(1:2^(l-1),:);
            bc_1 = bf_1(1:2^(l-1),:);
            Ac_D = Af_D(:,1:2^(l-1));
            bc_D = bf_D;
            Ac_d = Af_d(1:2^(l-1),1:2^(l-1),:);
            bc_d = bf_d(1:2^(l-1),:);
            QOI_c(k) = tnn_layer(x,D,Ac_1,bc_1,Ac_D,bc_D,Ac_d,bc_d);
        end
    end
    if lambda == 1
        fprintf('level: %d lambda: %.4f accep_rate: %.5f beta: %.4f temp_step: %d ESS: %.4f N: %d\n',l,lambda,u_rate,beta,count,1/sum(W.^2),N)
    end
    if u_rate < 0.2
        beta = beta/r;
    elseif u_rate > 0.8
        beta = beta*r;
        if beta > 1
            beta = 1;
        end
    end
end

Z - (ZZ+ZZ_max)

if l == K
    sums(1) = sum(QOI_f)/N;
    %sums(2) = exp(Z)*sum(QOI_f)/N;
    sums(2) = exp(ZZ)*exp(ZZ_max)*sum(QOI_f)/N;
    %sums(3) = exp(Z);
    sums(3) = exp(ZZ)*exp(ZZ_max);
    cost = 2^(2*l);
else
    sums(1) = sum(QOI_f .* exp(f_f-G_k))/sum(exp(f_f-G_k)) - ...
        sum(QOI_c .* exp(f_c-G_k))/sum(exp(f_c-G_k));
    %sums(2) = exp(Z)*( sum(QOI_f .* exp(f_f-G_k))- sum(QOI_c .* exp(f_c-G_k)) )/N;
    sums(2) = exp(ZZ)*exp(ZZ_max)*( sum(QOI_f .* exp(f_f-G_k))- sum(QOI_c .* exp(f_c-G_k)) )/N;
    %sums(3) = exp(Z)*( sum(exp(f_f-G_k)) - sum(exp(f_c-G_k)) )/N;
    sums(3) = exp(ZZ)*exp(ZZ_max)*( sum(exp(f_f-G_k)) - sum(exp(f_c-G_k)) )/N;
    cost = 2*2^(2*l);
    %if any(exp(f_f-G_k) == 0) || any(exp(f_c-G_k) == 0)
    %    fprintf('eps %.4f rel %d\n', eps, rel)
    %    fprintf('loglike f_f %.4f\n', f_f(exp(f_f-G_k) == 0)-G_k(exp(f_f-G_k) == 0))
    %    fprintf('loglike f_c %.4f\n', f_c(exp(f_f-G_k) == 0)-G_k(exp(f_f-G_k) == 0))
    %end
end

end
