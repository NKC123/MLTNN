clear;set(0,'defaultaxesfontsize',20);
% scalar input x=1, scalar output
% estimate convergence of 2 layer NN f_N(x;theta_N)
L=10;R=100
x=1;a=1

 
for r=1:R

    N=1*2^L;k=[1:N]';
    w1=randn(N,1)./k.^a;
    b1=randn(N,1)./k.^a;
    w2=diag(1./k.^a)*randn(N,N,1)*diag(1./k.^a);
    b2=randn(N,1)./k.^a;
    w3=randn(1,N,1)./(k').^a;
    b3=0*randn;

    for l=1:L

    fine = w3(1:2^l)*max(w2(1:2^l,1:2^l)*max(w1(1:2^l)*x+b1(1:2^l),0)+b2(1:2^l),0)+b3;
    coarse = w3(1,1:2^(l-1))*max(w2(1:2^(l-1),1:2^(l-1))*max(w1(1:2^(l-1))*x+b1(1:2^(l-1)),0)+...
        b2(1:2^(l-1)),0)+b3;
    
    err(l,r)=(fine-coarse)^2;
    err2(l,r)=(fine^2-coarse^2)^2;
    est(l,r)=fine;
    
    end
    
end

out=sum(est,2)/R
out2=sum(est.^2,2)/R
err1=sum(err,2)/R    
err22=sum(err2,2)/R

figure(1)
plot(1:10,log2(err1),'-o');grid;hold;
plot(1:10,-3*(1:10));hold
legend('log 2nd moment',strcat('-',num2str(3),'l'))
xlabel('level')
ylabel('log increment 2nd moment')


