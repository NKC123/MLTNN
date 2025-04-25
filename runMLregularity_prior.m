% estimate convergence of 3 layer NN f_N(x;theta_N)

randn('seed',20)
rand('seed',20)
[filepart,~,~] = fileparts(pwd); 
loadpath = fullfile(filepart,'Results','observations.mat');
load(loadpath,'data','x_data')
sigma = 0.1;
K = 4;
s = 1;
n = size(x_data,1);
m = 1;
D = 3;
M = 30;
Lmax = 7;
x = ones(size(x_data,1),1)*2;

err1 = zeros(Lmax-K+1,M);
err2 = zeros(Lmax-K+1,M);

for l = K:Lmax  

    sums = [];
    cst  = 0;
    
    for j = 1:M
        [Af_1,bf_1,Af_D,bf_D,Af_d,bf_d] = tnn_prior(l,s,n,m,D);
        fine = tnn_layer(x,D,Af_1,bf_1,Af_D,bf_D,Af_d,bf_d);
        Ac_1 = Af_1(1:2^(l-1),:);
        bc_1 = bf_1(1:2^(l-1),:);
        Ac_D = Af_D(:,1:2^(l-1));
        bc_D = bf_D;
        Ac_d = Af_d(1:2^(l-1),1:2^(l-1),:);
        bc_d = bf_d(1:2^(l-1),:);
        coarse = tnn_layer(x,D,Ac_1,bc_1,Ac_D,bc_D,Ac_d,bc_d);
        err1(l-K+1,j) = (fine-coarse)^2;
        err2(l-K+1,j) = abs(fine-coarse);
        %[sums_j, cst_j] = mlsmc_l(l,N,K,sigma,data,x_data,s,n,m,D);
        %sums(j,:) = sums_j;
        %cst  = cst  + cst_j/M;  
        %fprintf('finish %d realizations\n', j)
    end
        
end

se = sum(err1,2)/M;
we = sum(err2,2)/M;

[filepart,~,~] = fileparts(pwd); 
savepath = fullfile(filepart,'Results','empirical results','mlsmc_empirical_prior.mat');
save(savepath,'Lmax','se','err1','we','err2')

figure(1);
set(gcf,'Position',[400 400 500 400]);
plot(K:Lmax,log2(se)','--or','linewidth',2)
line = refline(-3,0);
line.LineStyle = '--';
line.Color = 'k';
xlabel('level $l$','Interpreter','latex','FontSize',20); 
ylabel('$\log_2 V_l$','Interpreter','latex','FontSize',20);
current = axis; axis([ K Lmax current(3:4) ]);
set(gca,'xtick',(1:Lmax));
grid on
legend('','$\mathcal{O}(-3l)$', ...
        'Interpreter','latex','Location','NorthEast','FontSize',15)
saveas(1,fullfile(filepart,'Results','empirical results','tnns_var_prior.fig'))
saveas(1,fullfile(filepart,'Results','empirical results','tnns_var_prior.png'))

figure(2);
set(gcf,'Position',[400 400 500 400]);
plot(K:Lmax,log2(we)','--or','linewidth',2)
line = refline(-3/2,0);
line.LineStyle = '--';
line.Color = 'k';
xlabel('level $l$','Interpreter','latex','FontSize',20); 
ylabel('$\log_2 B_l$','Interpreter','latex','FontSize',20);
current = axis; axis([ K Lmax current(3:4) ]);
set(gca,'xtick',(1:Lmax));
grid on
legend('','$\mathcal{O}(-3/2l)$', ...
        'Interpreter','latex','Location','NorthEast','FontSize',15)
saveas(2,fullfile(filepart,'Results','empirical results','tnns_mean_prior.fig'))
saveas(2,fullfile(filepart,'Results','empirical results','tnns_mean_prior.png'))


