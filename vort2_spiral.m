clear all

GridSize=128;
Visc=0.005;

h=2*pi/GridSize;
axis=h*[1:1:GridSize];

[x,y]= meshgrid(axis,axis);


FinTime=50;
dt=0.1;
t=0;

FrameRate=5;
Mov(10)=struct('cdata',[],'colormap',[]);

k=0;
j=1;


H=exp(-((x-pi+pi/5).^2+(y-pi+pi/5).^2)/(0.3))-exp(-((x-pi-pi/5).^2+(y-pi+pi/5).^2)/(0.2))+exp(-((x-pi-pi/5).^2+(y-pi-pi/5).^2)/(0.4));

epsilon=0.9;
Noise=random('unif',-1,1,GridSize,GridSize);
w=H+epsilon*Noise;
%w=H;
w_hat=fft2(w);


kx=1i*ones(1,GridSize)'*(mod ((1:GridSize)-ceil(GridSize/2+1),GridSize)-floor(GridSize/2));
ky=1i*(mod((1:GridSize)'-ceil(GridSize/2+1),GridSize)-floor(GridSize/2))*ones(1,GridSize);

AliasCor=kx<2/3*GridSize&ky<2/3*GridSize;


Lap_hat=kx.^2+ky.^2;
ksqr=Lap_hat; 
ksqr(1,1)=1;

while t<FinTime
    psi_hat = -w_hat./ksqr;
    u = real(ifft2(ky.*psi_hat));
    v = real(ifft2(-kx.* psi_hat));
    w_x = real(ifft2(kx.* w_hat));
    w_y = real(ifft2(ky.* w_hat));
    
    VgradW= u.*w_x+v.*w_y;
    VgradW_hat = fft2(VgradW);
    VgradW_hat = AliasCor.*VgradW_hat;
   
     
    
    w_hat_update= 1./(1/dt-0.5*Visc*Lap_hat).*((1/dt+0.5*Visc*Lap_hat).*w_hat-VgradW_hat);
    
    
    if(k==FrameRate)
        
        w=real(ifft2(w_hat_update));
        
        
        vel=sqrt(u.^2+v.^2);
        
        %contourf(x,y,w,80);
        surf(x,y,w,'edgecolor','none');
        view(2);
        axis off;
        axis square;
        colorbar;
        
        shading flat; colormap('Jet');
        drawnow
        
        Mov(j)=getframe;
        k=0;
        j=j+1;
        
        
    end
    
    w_hat = w_hat_update;
    
    t=t+dt;
    k=k+1;
    
end

    
        
    
    

    
    
    
    
    
    


