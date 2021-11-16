cd '~/Box Sync/Work/fsid/code/fsid/matlab/'
N = 101;
mset= [1 4 ,2];
pset= [1, 2 , 5];
nset= [2,1,5,7];

%fdestim_bd without transient
for m=mset
    for p=pset
        for n=nset
            A = randn(n,n);
            B = randn(n,m);
            C = randn(p,n);
            D = randn(p,m);
            D = zeros(p,m);
            f = (0:N-1)/N;
            zset = exp(1j*2*pi*f);
            U = randn(N,m)+1i*randn(N,m);
            Y = zeros(n,p);
            for i = 1:N,
                Y(i,:) = ((D+C*((zset(i)*eye(n)-A)\B))*U(i,:).').';
            end
            fddata = {zset, Y, U};
            [b,d,xt] = fdestim_bd(zset, Y, U, A, C, false, 'Real', ...
                                      true);
            Ye = zeros(n,p);
            for i = 1:N,
                Ye(i,:) = ((d+C*((zset(i)*eye(n)-A)\b))*U(i,:).').';
            end
            if norm(Ye-Y)/norm(Ye)>1e-6
                error('Unit test failed')
            end
        end
    end
end


%fdestim_bd with transient
for m=mset
    for p=pset
        for n=nset
            A = randn(n,n);
            B = randn(n,m);
            C = randn(p,n);
            D = randn(p,m);
            %D = zeros(p,m);
            Xt = randn(n,1);
            f = (0:N-1)/N;
            zset = exp(1j*2*pi*f);
            U = randn(N,m)+1i*randn(N,m);
            Y = zeros(n,p);
            for i = 1:N,
                Y(i,:) = (D*U(i,:).').' + ... 
                         ((C*((zset(i)*eye(n)-A)\B))*U(i,:).').' + ...
                         ((C*((zset(i)*eye(n)-A)\Xt))*zset(i).').';
            end
            fddata = {zset, Y, U};
            [b,d,xt] = fdestim_bd(zset, Y, U, A, C, true, 'Real', ...
                                      true);
            Ye = zeros(n,p);
            for i = 1:N,
                Ye(i,:) = (d*U(i,:).').' + ... 
                         ((C*((zset(i)*eye(n)-A)\b))*U(i,:).').' + ...
                         ((C*((zset(i)*eye(n)-A)\xt))*zset(i).').';
            end
            if norm(Ye-Y)/norm(Ye)>1e-6
                error('Unit test failed')
            end
        end
    end
end

%fdestim_bd with transient and weighting
for m=mset
    for p=pset
        for n=nset
            A = randn(n,n);
            B = randn(n,m);
            C = randn(p,n);
            D = randn(p,m);
            %D = zeros(p,m);
            Xt = randn(n,1);
            f = (0:N-1)/N;
            zset = exp(1j*2*pi*f);
            U = randn(N,m)+1i*randn(N,m);
            Y = zeros(n,p);
            for i = 1:N,
                Y(i,:) = (D*U(i,:).').' + ... 
                         ((C*((zset(i)*eye(n)-A)\B))*U(i,:).').' + ...
                         ((C*((zset(i)*eye(n)-A)\Xt))*zset(i).').';
            end
            fddata = {zset, Y, U};
            w = randn(N,p,p);
            [b,d,xt] = fdestim_bd(zset, Y, U, A, C, true, 'Real', ...
                                      true,w);
            Ye = zeros(n,p);
            for i = 1:N,
                Ye(i,:) = (d*U(i,:).').' + ... 
                         ((C*((zset(i)*eye(n)-A)\b))*U(i,:).').' + ...
                         ((C*((zset(i)*eye(n)-A)\xt))*zset(i).').';
            end
            if norm(Ye-Y)/norm(Ye)>1e-6
                error('Unit test failed')
            end
        end
    end
end

%fdestim_bd without transient
for m=mset
    for p=pset
        for n=nset
            A = randn(n,n);
            B = randn(n,m);
            C = randn(p,n);
            D = randn(p,m);
            D = zeros(p,m);
            f = (0:N-1)/N;
            zset = exp(1j*2*pi*f);
            U = randn(N,m)+1i*randn(N,m);
            Y = zeros(n,p);
            for i = 1:N,
                Y(i,:) = ((D+C*((zset(i)*eye(n)-A)\B))*U(i,:).').';
            end
            fddata = {zset, Y, U};
            [b,d,xt] = fdestim_bd(zset, Y, U, A, C, false, 'Real', ...
                                      false);
            Ye = zeros(n,p);
            for i = 1:N,
                Ye(i,:) = ((d+C*((zset(i)*eye(n)-A)\b))*U(i,:).').';
            end
            if norm(Ye-Y)/norm(Ye)>1e-6
                error('Unit test failed')
            end
        end
    end
end

%fdestim_bd with transient
for m=mset
    for p=pset
        for n=nset
            A = randn(n,n);
            B = randn(n,m);
            C = randn(p,n);
            D = randn(p,m);
            D = zeros(p,m);
            Xt = randn(n,1);
            f = (0:N-1)/N;
            zset = exp(1j*2*pi*f);
            U = randn(N,m)+1i*randn(N,m);
            Y = zeros(n,p);
            for i = 1:N,
                Y(i,:) = (D*U(i,:).').' + ... 
                         ((C*((zset(i)*eye(n)-A)\B))*U(i,:).').' + ...
                         ((C*((zset(i)*eye(n)-A)\Xt))*zset(i).').';
            end
            fddata = {zset, Y, U};
            [b,d,xt] = fdestim_bd(zset, Y, U, A, C, true, 'Real', ...
                                      false);
            Ye = zeros(n,p);
            for i = 1:N,
                Ye(i,:) = (d*U(i,:).').' + ... 
                         ((C*((zset(i)*eye(n)-A)\b))*U(i,:).').' + ...
                         ((C*((zset(i)*eye(n)-A)\xt))*zset(i).').';
            end
            if norm(Ye-Y)/norm(Ye)>1e-6
                error('Unit test failed')
            end
        end
    end
end

%fdestim_cd without transient, d and weighting
for m=mset
    for p=pset
        for n=nset
            A = randn(n,n) + 1i*randn(n,n);
            B = randn(n,m) + 1i*randn(n,m);  
            C = randn(p,n) + 1i*randn(p,n);
            D = randn(p,m) + 1i*randn(p,m); 
            A = real(A);
            B = real(B);
            C = real(C);
            D = real(D);
            %D = zeros(p,m);
            Xt = randn(n,1);
            Xt = zeros(n,1);
            f = (0:N-1)/N;
            zset = exp(1j*2*pi*f);
            U = randn(N,m)+1i*randn(N,m);
            Y = zeros(n,p);
            for i = 1:N,
                Y(i,:) = (D*U(i,:).').' + ... 
                         ((C*((zset(i)*eye(n)-A)\B))*U(i,:).').' + ...
                         ((C*((zset(i)*eye(n)-A)\Xt))*zset(i).').';
            end
            fddata = {zset, Y, U};
            w = randn(N,p,p);
            [c,d] = fdestim_cd(zset, Y, U, A, B, Xt, 'Real', ...
                                      true, w);
            Ye = zeros(n,p);
            for i = 1:N,
                Ye(i,:) = (d*U(i,:).').' + ... 
                         ((c*((zset(i)*eye(n)-A)\B))*U(i,:).').' + ...
                         ((c*((zset(i)*eye(n)-A)\Xt))*zset(i).').';
            end
            if norm(Ye-Y)/norm(Ye)>1e-6
                error('Unit test failed')
            end
        end
    end
end

%fdestim_cd with transient, d and weighting
for m=mset
    for p=pset
        for n=nset
            A = randn(n,n) + 1i*randn(n,n);
            B = randn(n,m) + 1i*randn(n,m);  
            C = randn(p,n) + 1i*randn(p,n);
            D = randn(p,m) + 1i*randn(p,m); 
            A = real(A);
            B = real(B);
            C = real(C);
            D = real(D);
            %D = zeros(p,m);
            Xt = randn(n,1);
            %Xt = zeros(n,1);
            f = (0:N-1)/N;
            zset = exp(1j*2*pi*f);
            U = randn(N,m)+1i*randn(N,m);
            Y = zeros(n,p);
            for i = 1:N,
                Y(i,:) = (D*U(i,:).').' + ... 
                         ((C*((zset(i)*eye(n)-A)\B))*U(i,:).').' + ...
                         ((C*((zset(i)*eye(n)-A)\Xt))*zset(i).').';
            end
            fddata = {zset, Y, U};
            w = randn(N,p,p);
            [c,d] = fdestim_cd(zset, Y, U, A, B, Xt, 'Real', ...
                                      true);
            Ye = zeros(n,p);
            for i = 1:N,
                Ye(i,:) = (d*U(i,:).').' + ... 
                         ((c*((zset(i)*eye(n)-A)\B))*U(i,:).').' + ...
                         ((c*((zset(i)*eye(n)-A)\Xt))*zset(i).').';
            end
            if norm(Ye-Y)/norm(Ye)>1e-6
                error('Unit test failed')
            end
        end
    end
end


for m=mset
    for p=pset
        for n=nset
            A = randn(n,n);
            B = randn(n,m);
            C = randn(p,n);
            D = randn(p,m);
            %D = zeros(p,m);
            Xt = randn(n,1);
            f = (0:N-1)/N;
            zset = exp(1j*2*pi*f);
            U = randn(N,m)+1i*randn(N,m);
            Y = zeros(n,p);
            for i = 1:N,
                Y(i,:) = (D*U(i,:).').' + ... 
                         ((C*((zset(i)*eye(n)-A)\B))*U(i,:).').' + ...
                         ((C*((zset(i)*eye(n)-A)\Xt))*zset(i).').';
            end
            fddata = {zset, Y, U};
            w = randn(N,p,p);
            [c,d] = fdestim_cd(zset, Y, U, A, B, Xt, 'Real', ...
                                      true, w);
            Ye = zeros(n,p);
            for i = 1:N,
                Ye(i,:) = (d*U(i,:).').' + ... 
                         ((c*((zset(i)*eye(n)-A)\B))*U(i,:).').' + ...
                         ((c*((zset(i)*eye(n)-A)\Xt))*zset(i).').';
            end
            if norm(Ye-Y)/norm(Ye)>1e-6
                error('Unit test failed')
            end
        end
    end
end


%gfdsid with transient, d and weighting
for m=mset
    for p=pset
        for n=nset
            A = randn(n,n) + 1i*randn(n,n);
            B = randn(n,m) + 1i*randn(n,m);  
            C = randn(p,n) + 1i*randn(p,n);
            D = randn(p,m) + 1i*randn(p,m); 
            A = real(A);
            B = real(B);
            C = real(C);
            D = real(D);
            %D = zeros(p,m);
            Xt = randn(n,1);
            %Xt = zeros(n,1);
            f = (0:N-1)/N;
            zset = exp(1j*2*pi*f);
            U = randn(N,m)+1i*randn(N,m);
            Y = zeros(n,p);
            for i = 1:N,
                Y(i,:) = (D*U(i,:).').' + ... 
                         ((C*((zset(i)*eye(n)-A)\B))*U(i,:).').' + ...
                         ((C*((zset(i)*eye(n)-A)\Xt))*zset(i).').';
            end
            fddata = {zset, Y, U};
            w = randn(N,p,p);
            for i=1:N
                wi = randn(p,p);
                w(i,:,:) = eye(p) + wi*wi';
            end
            [ss,s] = gfdsid( {zset, Y, U}, n, 2*n, true, 'Real', ...
                             true, w);
            [a,b,c,d,xt] = ss{:};
            Ye = zeros(n,p);
            for i = 1:N,
                Ye(i,:) = (d*U(i,:).').' + ... 
                         ((c*((zset(i)*eye(n)-a)\b))*U(i,:).').' + ...
                         ((c*((zset(i)*eye(n)-a)\xt))*zset(i).').';
            end
            if norm(Ye-Y)/norm(Ye)>1e-6
                error('Unit test failed')
            end
        end
    end
end
disp('All tests passed')


% $$$ for m=mset
% $$$     for p=pset
% $$$         for n=nset
% $$$             A = randn(n,n);
% $$$             B = randn(n,m);
% $$$             C = randn(p,n);
% $$$             D = randn(p,m);
% $$$             f = (0:N-1)/N;
% $$$             zset = exp(1j*2*pi*f);
% $$$             U = randn(N,m)+1i*randn(N,m);
% $$$             Y = zeros(n,p);
% $$$             for i = 1:N,
% $$$                 Y(i,:) = ((D+C*((zset(i)*eye(n)-A)\B))*U(i,:).').';
% $$$             end
% $$$             fddata = {zset, Y, U};
% $$$             sys = gfdsid(fddata,n,n+1);
% $$$             [a,b,c,d] = sys{:};
% $$$             Ye = zeros(n,p);
% $$$             for i = 1:N,
% $$$                 Ye(i,:) = ((d+C*((zset(i)*eye(n)-a)\b))*u(i,:).').';
% $$$             end
% $$$             if norm(Ye-Y)/norm(Ye)>1e-6
% $$$                 error('Unit test failed')
% $$$             end
% $$$         end
% $$$     end
% $$$ end
