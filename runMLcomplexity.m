close all; clear all;

% nslots = str2double(getenv('NSLOTS'));
% parpool(nslots);
% pp = gcp;
% poolsize = pp.NumWorkers;

[filepart,~,~] = fileparts(pwd); 
%addpath(fullfile(filepart, 'pde_solver_1D'))

randn('seed',20)
rand('seed',20)

Eps    = [0.004 0.008 0.016];%[0.0005 0.001 0.002 0.004 0.008 0.016];%[ 0.0025 0.005 0.01 0.02 0.04 ];
%Eps    = 0.00025;%[ 0.00025 0.0005 0.001 ];
M = 100;%30; %100ls



% generate data
%x_data = [1/3; 2/3]; % corresponding point of data
%x_data = linspace(0.1,0.9,9)';
loadpath = fullfile(filepart,'Results','observations.mat');
load(loadpath,'data','x_data')
sigma = 0.1;
K = 3;
s = 1;
n = size(x_data,1);
m = 1;
D = 3;

tic
ml_complexity(@opre_mlsmc_l,Eps, M,sigma, data, x_data,K,s,n,m,D)
toc

% plot results
% mlmc_plot(filename);


