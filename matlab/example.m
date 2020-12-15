N = 101;
mset= [1 4 ,2];
pset= [1, 2 , 5];
nset= [2,1,5,7];

%Example illustrating use of gfdsid 
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
