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
function ml_complexity(mlsmc_l, Eps, M, sigma, data, x_data,K,s,n,m,D)

resn = zeros(M,length(Eps));
rere = zeros(M,length(Eps));
mlcosts = zeros(1,length(Eps));

alpha = 3/2;%3/2,2;
beta  = 3;

[filepart,~,~] = fileparts(pwd);
%loadpath = fullfile(filepart, 'Results','empirical results','mlsmc_empirical.mat');
%load(loadpath,'se','cost')
%se1 = se{1};

% analytic solution of the quantity of interest (u^2)
loadpath = fullfile(filepart, 'Results','solutions.mat');
load(loadpath,'solutions')

sol = solutions;

for i = 1:length(Eps)
    eps = Eps(i);
    
%%%%% number of levels required for eps by constants
    deno = (2^alpha - 1)*eps;
    L = max(ceil( log2(sqrt(2)/deno)/alpha ), K+1); %max(ceil( log2(sqrt(2)/deno)/alpha ) + 3, K+1);   
    
%%%%% number of samples required for eps by constants
%     L_set = (1:1:L);
%     K_l = sqrt(var1(1)*2^(-1))+ ...
%         sum( sqrt(c2*c3*2.^(-L_set(2:end)*(beta - gamma))) );
%         
%     N_l = ceil(  2*eps^(-2)*K_l.*sqrt(c2*2.^-(L_set.*(beta+gamma)))/c3);
%     
%     N_l(1) = ceil( 2*eps^(-2)*K_l*sqrt(var1(1)*2) );   

    %N_l = ceil(2*eps^(-2)*sum(sqrt(se1(1:(L-K+1)).*cost(1:(L-K+1)))).* ...
    %    sqrt(se1(1:(L-K+1))./cost(1:(L-K+1))) );

    L_set = (K:1:L);
    var   = 2.^(-beta*L_set);
    %var(1) = 0.01;
    cost_tem  = 2.^(2*L_set);
    K_l = sum(sum( sqrt(var.*cost_tem) ));
    N_l = max(ceil( 2* eps^(-2)*K_l.*sqrt(var./cost_tem) ), 2)

    
    costl = 0;
    
    % calculate MSE
    parfor j = 1:M
        %[P,cl] = mlsmc(mlsmc_l,eps,L,N_l,sigma,data,x_data);
        sumlsn = 0;
        sumlret = 0;
        sumlreb = 0;
        for l=K:L
             Nl = N_l(l-K+1);
            [sums, cost] = mlsmc_l(l,Nl,K,sigma,data,x_data,s,n,m,D,j,eps);
            sumlsn  = sumlsn + sums(1);
            sumlret = sumlret + sums(2);
            sumlreb = sumlreb + sums(3);
            costl   = costl + cost*Nl;
        end
        resn(j,i) = sumlsn;
        rere(j,i) = sumlret/sumlreb;
    end
    mlcosts(i) = costl/M;
end
mlsn_solsn = sum((resn-sol(1)).^2)./M;
mlsn_solre = sum((resn-sol(2)).^2)./M;
mlre_solsn = sum((rere-sol(1)).^2)./M;
mlre_solre = sum((rere-sol(2)).^2)./M;


savepath = fullfile(filepart,'Results','complexity results','mlsmc_complexity.mat');
save(savepath,'mlsn_solsn','mlsn_solre','mlre_solsn','mlre_solre','mlcosts','resn','rere')

end