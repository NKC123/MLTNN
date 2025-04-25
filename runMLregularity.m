close all; clear all;

% nslots = str2double(getenv('NSLOTS'));
% parpool(nslots);
% pp = gcp;
% poolsize = pp.NumWorkers;

[filepart,~,~] = fileparts(pwd); 
%addpath(fullfile(filepart, 'pde_solver_1D'))

randn('seed',20)
rand('seed',20)


N      = 1000;%1000;       % samples for convergence tests
Lmax      = 7;           % levels for convergence tests

% generate data
%x_data = [1/3; 2/3]; % corresponding point of data
%x_data = linspace(0.1,0.9,9)';
sigma  = 0.01;%0.01;0.5;0.1
%data   = -0.5*rand(1)*(x_data.^2-x_data) + sigma*randn(length(x_data),1);

loadpath = fullfile(filepart,'Results','observations.mat');
load(loadpath,'data','x_data')

s = 1; %\alpha = 2*s;
n = size(x_data,1);
m = 1;
D = 3;
K = 3;

tic
ml_empirical(@opre_mlsmc_l, N, Lmax, sigma, data, x_data,s,n,m,D,K);
toc
