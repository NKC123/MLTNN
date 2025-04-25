% --------------------------------------------------------- %
% function mlmc_test(mlmc_l, N, L, Eps, fp, theta, data)
%
% multilevel Monte Carlo test routine
%
% mlmc_l   = function for level l estimator
% N        = number of samples for convergence tests
% L        = number of levels for convergence tests
% Eps      = desired accuracy array for MLMC calcs
% fp       = file handle for printing to file
% sigma    = std deviation of the error of y - G
% data     = observations
% --------------------------------------------------------- %
function [we,se,cost] = ml_empirical(mlsmc_l, N, Lmax, sigma, data, x_data,s,n,m,D,K)

%
% first, convergence tests
%

we1  = zeros(Lmax-K+1,1);
we2  = zeros(Lmax-K+1,1);
we3  = zeros(Lmax-K+1,1);
se1  = zeros(Lmax-K+1,1);
se2  = zeros(Lmax-K+1,1);
se3  = zeros(Lmax-K+1,1);
se4  = zeros(Lmax-K+1,1);
se5  = zeros(Lmax-K+1,1);
se6  = zeros(Lmax-K+1,1);
cost = zeros(Lmax-K+1,1);

for l = K:Lmax  

    % plot autocorrelation of samples
%     figure(l)
%     autocorr(u,30)
%     shg
    
    sums = [];
    cst  = 0;

    M = 30;%100,200;
    
    parfor j = 1:M
        %[Af_1,bf_1,Af_D,bf_D,Af_d,bf_d] = tnn_prior(l,s,n,m);
        %u = {Af_1,bf_1,Af_D,bf_D,Af_d,bf_d};
        [sums_j, cst_j] = mlsmc_l(l,N,K,sigma,data,x_data,s,n,m,D);
        sums(j,:) = sums_j;
        cst  = cst  + cst_j/M;  
        fprintf('finish %d realizations\n', j)
    end
    
    cost(l-K+1) = cst;
    we1(l-K+1) = abs( sum(sums(:,1))/M );
    we2(l-K+1) = abs( sum(sums(:,2))/M );
    we3(l-K+1) = abs( sum(sums(:,3))/M );
    se1(l-K+1) = N*sum( (sums(:,1) - sum(sums(:,1))/M).^2 )/M;
    se2(l-K+1) = N*sum( (sums(:,2) - sum(sums(:,2))/M).^2 )/M;
    se3(l-K+1) = N*sum( (sums(:,3) - sum(sums(:,3))/M).^2 )/M;
    se4(l-K+1)  = sum(sums(:,1).^2)/M;
    se5(l-K+1)  = sum(sums(:,2).^2)/M;
    se6(l-K+1)  = sum(sums(:,3).^2)/M;
        
end

we = {we1,we2,we3};
se = {se1,se2,se3,se4,se5,se6};

[filepart,~,~] = fileparts(pwd); 
savepath = fullfile(filepart,'Results','empirical results','mlsmc_empirical.mat');
save(savepath,'Lmax','cost','we','se')

end