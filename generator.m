%% pre-settings
close all; clear all;

randn('seed',20)
rand('seed',20)

sigma  = 0.1;%0.01;0.1
eps = 1e-4;%1e-4
% observation points
%x_data = 2 + sqrt(0.5)*randn(10,100);

%[filepart,~,~] = fileparts(pwd);

%% generate data
%tic;
%[data,u] =  data_generator(x_data,sigma);
%toc;
%savepath = fullfile(filepart, 'Results','observations.mat');
%save(savepath,'data','x_data','u')
%% approximate reference solution
%addpath(fullfile(filepart,'MLSMC_2D_nlin'));
K = 4;
%loadpath = fullfile(filepart,'Results','observations.mat');
%load(loadpath,'data','x_data')
load('observations.mat','data','x_data')
%save(savepath,'data','x_data','u')
tic;
[solutions] = mlsmc_solution(@opre_mlsmc_l,eps,sigma,data,x_data,K);
toc;
savepath = fullfile(filepart, 'Results','solutions.mat');
save(savepath,'solutions');
%rmpath(fullfile(filepart,'MLSMC_2D_nlin'))
%rmpath(fullfile(filepart,'solver2Dnlin'));
%% data generator
% ------------------------------------------------------ %
% function [data] = data_generator(x_data, u, sigma)
% generate data based on the model y = G(u) + v
% input:  x_data = observation points
%         u      = random inputs
%         sigma  = std deviation of noise
% output: data   = data from the model
% ------------------------------------------------------ %
function [data,u] = data_generator(x_data, sigma)
D = 3;
n = size(x_data,1);
m = 1;
l = 8; %7,8,9
s = 1;
%Gu = 0;

%for i=1:100
[A_1,b_1,A_D,b_D,A_d,b_d] = tnn_prior(l,s,n,m,D);
[Gu] = tnn_layer(x_data,D,A_1,b_1,A_D,b_D,A_d,b_d);
u = {A_1,b_1,A_D,b_D,A_d,b_d};
%Gu = Gu + Guu/100;
%end


data = Gu + sigma*randn(size(x_data,2), 1)';
end
%% solution generator
% --------------------------------------------------------------------- %
% function [solutions] = mlsmc_solution(mlsmc_l,eps,sigma,data,x_data,k)
% inputs: mlsmc_l      = function for level l estimator
%         eps          = required accuracy
%         sigma        = std deviation of the error of y - G
%         data         = observations (values of y) vector
%         x_data       = corresponding observation points
%         K            = starting refinement level
% ouputs: solutions(1) = reference by sn
%         solutions(2) = reference by re 
% --------------------------------------------------------------------- %
function [solutions] = mlsmc_solution(mlsmc_l,eps,sigma,data,x_data,K)
% loading emperical results first
%[filepart,~,~] = fileparts(pwd);
%loadpath = fullfile(filepart, 'Results','empirical results','mlsmc_empirical.mat');
%load(loadpath,'se','cost')
%se1 = se{1};

solutions = zeros(1,2);

%L = 7;
%K = 1;
s = 1;
n = size(x_data,1);
m = 1;
D = 3;

%vc1 = 2*sum(sum(sqrt(se1.*cost))).*sqrt(se1./cost);

%L = ceil(log2(sqrt(2)/eps/3)/2);
alpha = 3/2;%3/2;
deno = (2^alpha - 1)*eps;
L = max(K+1,ceil( log2(sqrt(2)/deno)/alpha ));    
%M = ceil(vc1./eps/eps);
%M = ceil(2*eps^(-2)*sum(sqrt(se1(1:L-K+1).*cost(1:L-K+1))).* ...
%        sqrt(se1(1:L-K+1)./cost(1:L-K+1)) );

beta = 3; 
L_set = (K:1:L);
var   = 2.^(-beta*L_set);
%var(1) = 0.01;
cost_tem  = 2.^(2*L_set);
K_l = sum(sum( sqrt(var.*cost_tem) ));
N_l = max(ceil( 2* eps^(-2)*K_l.*sqrt(var./cost_tem) ), 2);

sumlsn  = 0;
sumlret = 0;
sumlreb = 0;

for l = K:L
    N = N_l(l-K+1);
    [sums,~] = mlsmc_l(l,N,K,sigma,data,x_data,s,n,m,D);
    sumlsn  = sumlsn  + sums(1);
    sumlret = sumlret + sums(2);
    sumlreb = sumlreb + sums(3);
end
solutions(1) = sumlsn;
solutions(2) = sumlret/sumlreb;

end
