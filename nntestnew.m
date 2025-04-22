clear;set(0,'defaultaxesfontsize',20);
% scalar input x=1, scalar output
% estimate convergence of 2 layer NN f_N(x;theta_N)
L=10;R=100
x=1;a=1/2

for l=1:L
 
    for r=1:R
    N=2^l;
    w1=randn(N,1)/N^a;
    b1=randn(N,1)/N^a;
    w2=randn(N,N,1)/N^a;
    b2=randn(N,1)/N^a;
    w3=randn(1,N,1)/N^a;
    b3=0*randn;
    
    fine = w3*max(w2*max(w1*x+b1,0)+b2,0)+b3;
    coarse = 2^a*w3(1,1:N/2)*max(2^a*w2(1:N/2,1:N/2)*max(2^a*w1(1:N/2)*x+2^a*b1(1:N/2),0)+2^a*b2(1:N/2),0)+b3;
    
    err(l,r)=(fine-coarse)^2;
    est(l,r)=fine;
    end
    
end

out=sum(est,2)/R;
out2=sum(est.^2,2)/R
err1=sum(err,2)/R;

figure(1)
plot(1:10,log2(err1),'-o');grid;hold;
plot(1:10,-2*a*(1:10));hold
legend('log 2nd moment',strcat('-',num2str(2*a),'l'))
xlabel('level')
ylabel('log increment 2nd moment')


