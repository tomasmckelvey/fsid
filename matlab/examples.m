clear variables;
addpath '/Users/mckelvey/Box Sync/Work/fsid/code/fsid/matlab'
%% Create frequency set

N = 100; % number of samples
fset = (0:N-1)/N;
w = 2*pi * fset;
z = exp(1i * w); %  samples at unit circle
s = 1i * w; % samples on imginary axis
gam = randn(N,1)+1i*randn(N,1); % random complex samples
%% Model sizes

n = 2; % model order
m = 1; % number inputs
p = 1; % number outputs
%% Estimte model from fddata sampled at gam

A = randn(n, n);
B = randn(n, m);
C = randn(p, n);

D = randn(p, m);
% create frequency function data
ff = fresp(gam, A, B, C, D);
% Estimate ss model from data 
sys1 = gfdsid({gam, ff, ones(N,1)}, n, n*2, false);
% Frequency response of estimated model
[Ae,Be,Ce,De] = sys1{1:4};
ffe = fresp(gam, Ae, Be, Ce, De);
% Print relative estimation error sys1
norm(ff(:)-ffe(:))/norm(ff(:))


%% Estimte model from DT ffdata

A = randn(n, n);
B = randn(n, m);
C = randn(p, n);
D = randn(p, m);
% create frequency function data
ff = fresp(z, A, B, C, D);
% Estimate ss model from data 
sys1 = gfdsid({z, ff, ones(N,1)}, n, n*2, false);
sys2 = fdsid({w, ff, ones(N,1)}, n, n*2, false);
sys3 = ffsid(w, ff, n, n*2);
% Frequency response of estimated model
[Ae,Be,Ce,De] = sys1{1:4};
ffe = fresp(z, Ae, Be, Ce, De);
% Print relative estimation error sys1
norm(ff(:)-ffe(:))/norm(ff(:))
[Ae,Be,Ce,De] = sys2{1:4};
ffe = fresp(z, Ae, Be, Ce, De);
% Print relative estimation error sys2
norm(ff(:)-ffe(:))/norm(ff(:))
[Ae,Be,Ce,De] = sys3{1:4};
ffe = fresp(z, Ae, Be, Ce, De);
% Print relative estimation error sys3
norm(ff(:)-ffe(:))/norm(ff(:))
%% Estimate model from CT ffdata

n = 2; p=3; m=1;
A = randn(n, n);
B = randn(n, m);
C = randn(p, n);
D = randn(p, m);
% create frequency function data
ff = fresp(s, A, B, C, D);
% Estimate ss model from data 
sys1 = gfdsid({s, ff, ones(N,1)}, n, 3*n, false);
% Using Bilinear transform
%[sys, s] = fdsid({w,y,u}, n, 2*n, estTrans, dtype, estimd, CT, T, w)
sys2 = fdsid({w, ff, ones(N,1)}, n, 3*n, false, [], [], true);
% Frequency response of estimated model
[Ae,Be,Ce,De] = sys1{1:4};
ffe = fresp(s, Ae, Be, Ce, De);
% Print relative estimation error sys1
norm(ff(:)-ffe(:))/norm(ff(:))
[Ae,Be,Ce,De] = sys2{1:4};
ffe = fresp(s, Ae, Be, Ce, De);
% Print relative estimation error sys2
norm(ff(:)-ffe(:))/norm(ff(:))
%% MIMO case
% Model sizes

n = 2; % model order
m = 3; % number inputs
p = 4; % number outputs
A = randn(n, n);
B = randn(n, m);
C = randn(p, n);
D = randn(p, m);
% Generate fd-data
u = randn(N,m)+1i*randn(N,m);
y = fdsim({A,B,C,D},u,z);
%[sys, s] = fdsid({w,y,u}, n, 2*n, estTrans, dtype, estimd, CT, T, w)
sys = fdsid({w,y,u}, n, 2*n, false);
ye = fdsim(sys(1:4),u,z);
norm(ye-y)/norm(y)
% Estimate ss with without d
sys  = fdsid({w,y,u}, n, 2*n,false,[],false);
ye = fdsim(sys(1:4),u,z);
norm(ye(:)-y(:))/norm(y(:))
%% Set D to zero and generate new data

D = zeros(p,m);
y = fdsim({A,B,C,D},u,z);
sys  = fdsid({w,y,u}, n, 2*n,false,[],false);
ye = fdsim(sys(1:4),u,z);
norm(ye-y)/norm(y)
%% Estimte model from DT fddata w noise + weighting
% noise free data

y = fdsim({A,B,C,D},u,z);
% colored noise model
ns = 0.01; %noise gain
An = 0.99*ones(1,1); Bn = 1.0*ones(1,1); Cn = 1.0*ones(1,1); Dn=1.0*ones(1,1);
F = fresp(z, An,Bn,Cn,Dn);
yn = zeros(N,p);
W = zeros(N,p,p);
for i = 1:N
    Rf = randn(p,p) * abs(F(i,1,1));
    yn(i,:) = y(i,:).' + ns*Rf*(randn(p,1) + 1i * randn(p,1));
    Cf = chol(Rf*Rf');
    W(i,:,:) = inv(Cf');
end
% Estimate without frequency weighting
sys  = fdsid({w,yn,u}, n, 2*n, [], [], [], [], [], []);
ye1 = fdsim(sys(1:4),u,z);
norm(ye1-y)/norm(y)
%% Estimate with frequency weighting

sys  = fdsid({w,yn,u}, n, 2*n, [], [], [], [], [], W);
ye = fdsim(sys(1:4),u,z);
norm(ye-y)/norm(y)
%% Complex ss-matrices
% Model sizes

n = 2; % model order
m = 1; % number inputs
p = 1; % number outputs
A = randn(n, n) + 1i*randn(n, n);
B = randn(n, m) + 1i*randn(n, m);
C = randn(p, n) + 1i*randn(p, n);
D = randn(p, m) + 1i*randn(p, m);
% Generate fd-data
u = randn(N,m)+1i*randn(N,m);
y = fdsim({A,B,C,D},u,z);
%[sys, s] = fdsid({w,y,u}, n, q, estTrans, dtype, estimd, CT, T, w)
sys = fdsid({w,y,u}, n, 2*n, false, 'Complex' );
ye = fdsim(sys(1:4),u,z);
norm(ye-y)/norm(y)
%% Estimate from DT time domain data

A = randn(n, n);
B = randn(n, m);
C = randn(p, n);
D = randn(p, m);
rho = max(abs(eig(A)));
A = A/rho/1.01; % Make stable
% Calulate frequency responses
ff = fresp(z, A, B, C, D);
% create data samples
u = randn(N,m);
y = dlsim(A,B,C,D,u);
uf = fft(u);
yf = fft(y);
%% Estimate without transient Xt

sys = fdsid({w,yf,uf}, n, 2*n, false);
[Ae,Be,Ce,De] = sys{1:4};
ffe = fresp(z, Ae, Be, Ce, De);
norm(ff(:)-ffe(:))/norm(ff(:))
%% Estimate with transient Xt

sys  = fdsid({w,yf,uf}, n, 2*n, true);
% Calulate frequency responses
[Ae,Be,Ce,De] = sys{1:4};
ffe = fresp(z, Ae, Be, Ce, De);
norm(ff(:)-ffe(:))/norm(ff(:))
%% 
%%