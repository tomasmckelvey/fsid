clear all
close all

Nset = [100,200, 400, 800];
nset= [2,5,10,20];
m = 2;
p = 8;
MC = 100;

tim_res = zeros(length(Nset),length(nset));
for N_idx=1:length(Nset)
    for n_idx=1:length(nset)
        for mc = 1:MC
            N = Nset(N_idx);
            n = nset(n_idx);
            A = randn(n,n); %+ 1i*randn(n,n);
            rho = max(abs(eig(A)));
            A = A/rho/1.01;
            B = randn(n,m); %+ 1i*randn(n,m);  
            C = randn(p,n); %+ 1i*randn(p,n);
            D = randn(p,m); %+ 1i*randn(p,m); 
            Xt = randn(n,1);
            %Xt = zeros(n,1);
            f = (0:N-1)/N;
            zset = exp(1j*2*pi*f);
            U = randn(N,m)+1i*randn(N,m);
            Y = randn(n,p);
            for i = 1:N,
                Y(i,:) = (D*U(i,:).').' + ... 
                         ((C*((zset(i)*eye(n)-A)\B))*U(i,:).').' + ...
                         ((C*((zset(i)*eye(n)-A)\Xt))*zset(i).').';
            end
            fddata = {zset, Y, U};
            w = zeros(N,p,p);
            for i=1:N
                wi = randn(p,p);
                w(i,:,:) = eye(p) + wi*wi';
            end
            tic;
% $$$             [ss,s] = gfdsid( {zset, Y, U}, n, n+1, true, 'Real', ...
% $$$                              true, w);
            [ss,s] = gfdsid( {zset, Y, U}, n, n+1, true, 'Real', ...
                             true);
            tim_res(N_idx,n_idx) = tim_res(N_idx,n_idx) + toc;
% $$$         [a,b,c,d,xt] = ss{:};
% $$$         Ye = zeros(n,p);
% $$$         for i = 1:N,
% $$$             Ye(i,:) = (d*U(i,:).').' + ... 
% $$$                       ((c*((zset(i)*eye(n)-a)\b))*U(i,:).').' + ...
% $$$                       ((c*((zset(i)*eye(n)-a)\xt))*zset(i).').';
% $$$         end
% $$$         enorm = norm(Ye-Y)/norm(Y); 
% $$$         if enorm>1e-6
% $$$             m,p,n
% $$$             error(['Unit test failed: enorm=',num2str(enorm)])
% $$$         end
        end
    end
end
tim_res = tim_res/MC;

% $$$ tim_res =
% $$$ 
% $$$     0.0136    0.0132    0.0194    0.0326
% $$$     0.0188    0.0235    0.0342    0.0571
% $$$     0.0366    0.0455    0.0648    0.1194
% $$$     0.0943    0.1197    0.1930    0.3661

xx={}; for n=1:length(Nset), xx{n} = num2str(Nset(n)); end
Nlabels = categorical(xx,xx)
xx={}; for n=1:length(nset), xx{n} = ['', num2str(nset(n))]; end
nlabels = categorical(xx,xx)

platforms = categorical({'Python','Julia','Matlab'},{'Python','Julia','Matlab'});


ax = bar(Nlabels,tim_res);
ylabel('Execution time [s]')
xlabel('Number of frequency samples (N)')
ax = gca;
set(ax,'fontsize',16);
leg=legend(nlabels,'location','nw')
title(leg,'Model order n')

julia_timing = ...
[0.00549532 0.00886636 0.0168048 0.0279553; 
 0.00942935 0.0149802 0.0259728 0.0523305; 
 0.0175457 0.0265168 0.0446474 0.103948; 
 0.0337763 0.0492217 0.0868943 0.187685];

python_timing_w = ...
[0.04 0.05 0.07 0.11;
 0.07 0.09 0.12 0.22;
 0.14 0.18 0.25 0.41
 0.28 0.35 0.52 0.85];

python_timing = ...
[0.03 0.03 0.04 0.07;
 0.05 0.06 0.1  0.16;
 0.09 0.13 0.18 0.29;
 0.19 0.25 0.36 0.58];

% Graph1 over model orders N=400 nidx=3
figure(2);
ax = bar(nlabels,[python_timing(3,:); julia_timing(3,:); tim_res(3,:) ]) ;
title('Timing comparison for data length N=400')
ylabel('Execution time [s]')
xlabel('Model order (n)')
ax = gca;
set(ax,'fontsize',16);
leg=legend(platforms,'location','nw')
%title(leg,'Model order n')
print -depsc timing_order.eps

figure(3);
ax = bar(Nlabels,[python_timing(:,3)'; julia_timing(:,3)'; tim_res(:,3)' ]) ;
title('Timing comparison for model order n=10')
ylabel('Execution time [s]')
xlabel('Data length (N)')
ax = gca;
set(ax,'fontsize',16);
leg=legend(platforms,'location','nw')
%title(leg,'Model order n')
print -depsc timing_N.eps
