function [v,ll_f_temp,ll_c_temp] = mutation(l,u,x_data,data,k,s,n,m,D,sigma,beta)
y = data;
Af_1 = u{1};
bf_1 = u{2};
Af_D = u{3};
bf_D = u{4};
Af_d = u{5};
bf_d = u{6};
% pCN rate
%beta = 0.4;%0.9-lambda*0.7;%0.25;%0.95;
% mutate samples with pCN
[A_1,b_1,A_D,b_D,A_d,b_d] = tnn_prior(l,s,n,m,D);
v1 = sqrt(1-beta^2).*Af_1 + beta.*A_1;
v2 = sqrt(1-beta^2).*bf_1 + beta.*b_1;
v3 = sqrt(1-beta^2).*Af_D + beta.*A_D;
v4 = sqrt(1-beta^2).*bf_D + beta.*b_D;
v5 = sqrt(1-beta^2).*Af_d + beta.*A_d;
v6 = sqrt(1-beta^2).*bf_d + beta.*b_d;
g_f = tnn_layer(x_data,D,v1,v2,v3,v4,v5,v6);
ll_f_temp = -0.5*(g_f - y)*(g_f - y)'/sigma/sigma;
if l == k
    %ll_c_temp = 0;
    ll_c_temp = -Inf;
else
    v11 = v1(1:2^(l-1),:);
    v22 = v2(1:2^(l-1),:);
    v33 = v3(:,1:2^(l-1));
    v44 = v4;
    v55 = v5(1:2^(l-1),1:2^(l-1),:);
    v66 = v6(1:2^(l-1),:);
    g_c = tnn_layer(x_data,D,v11,v22,v33,v44,v55,v66);
    %ll_c_temp = exp(C + -0.5*(g_c - y)*(g_c - y)'/sigma/sigma);
    ll_c_temp = -0.5*(g_c - y)*(g_c - y)'/sigma/sigma;
end
v = {v1,v2,v3,v4,v5,v6};
end
