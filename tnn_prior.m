function [A_1,b_1,A_D,b_D,A_d,b_d] = tnn_prior(l,s,n,m,D)

N = 2^l;
kA1 = [1:N]' * ones(1,n);%[1:n];
kAD = ones(m,1) * [1:N];%[1:m]' * [1:N];
kAd = [1:N]' * [1:N];
kd = [1:N]';
kD = [1:m]';
A_1 = randn(N,n)./kA1.^s;
b_1 = randn(N,1)./kd.^s;
A_D = randn(m,N)./kAD.^s;
b_D = randn(m,1)./kD.^s;
A_d = randn(N,N,D)./kAd.^s;
b_d = randn(N,1,D)./kd.^s;

end