% test gpuSparse class
clear all
reset(gpuDevice(1))

M = 121401;
N = 113331;
P = 5e-4;

%M = 4;
%N = 4;
%P = 1;

disp('---SETUP---')

rand('state',0);
randn('state',0);

tic; fprintf('Making sparse... ')
A = sprandn(M,N,P);
toc

% remove unwanted precision
[i j v] = find(A);
v = double(single(v));
A = sparse(i,j,v,M,N);

tic; fprintf('Converting to gpuSparse... ')
a = gpuSparse(A); validate(a)
toc

fprintf('Sorted index conversion to gpuSparse: ')
[i j v] = find(a);
tic;
b = gpuSparse(i,j,v,M,N); validate(b)
fprintf('errors = [%i %i %i]. ',any(a.row~=b.row),any(a.col~=b.col),any(a.val~=b.val))
toc

fprintf('Unsorted index conversion to gpuSparse: ')
k = randperm(numel(v));
i = i(k);
j = j(k);
v = v(k);
tic; 
b = gpuSparse(i,j,v,M,N); validate(b)
fprintf('errors = [%i %i %i]. ',any(a.row~=b.row),any(a.col~=b.col),any(a.val~=b.val))
toc

x = randn(N,1,'gpuArray');
y = randn(M,1,'gpuArray');

% remove unwanted precision
x = double(single(x));
y = double(single(y));

%% Expected failues (bounds etc)
disp('---CATCH ERRORS---')

try; gpuSparse('test'); warning('failed'); end
try; gpuSparse(rand(3,3,3)); warning('failed'); end
try; gpuSparse(1,-1); warning('failed'); end
try; gpuSparse(-1,1); warning('failed'); end
try; gpuSparse(1,Inf); warning('failed'); end
try; gpuSparse(1,NaN); warning('failed'); end
try; gpuSparse(Inf,1); warning('failed'); end
try; gpuSparse(NaN,1); warning('failed'); end
try; gpuSparse(intmax('int32'),1); warning('failed'); end
try; gpuSparse(1,intmax('int32')); warning('failed'); end
try; gpuSparse(1,-1,0); warning('failed'); end
try; gpuSparse(-1,1,0); warning('failed'); end
try; gpuSparse(1,Inf,0); warning('failed'); end
try; gpuSparse(1,NaN,0); warning('failed'); end
try; gpuSparse(Inf,1,0); warning('failed'); end
try; gpuSparse(NaN,1,0); warning('failed'); end
try; gpuSparse(intmax('int32'),1,0); warning('failed'); end
try; gpuSparse(1,intmax('int32'),0); warning('failed'); end
try; gpuSparse(1,1,'test'); warning('failed'); end
try; gpuSparse(1:2,1:1,1:2); warning('failed'); end
try; gpuSparse(1:1,1:2,1:2); warning('failed'); end
%try; gpuSparse(1:2,1:2,1:1); warning('failed'); end % this works... why is it here?!
try; gpuSparse(1:1,1:1,1:2); warning('failed'); end
try; gpuSparse(1,1,1,10,0); warning('failed'); end
try; gpuSparse(1,1,1,0,10); warning('failed'); end
try; gpuSparse(1,1,1,10,intmax('int32')); warning('failed'); end
try; gpuSparse(1,1,1,intmax('int32'),10); warning('failed'); end
try; gpuSparse(1,1,10,10,'test'); warning('failed'); end
try; gpuSparse(1,1,10,'test',10); warning('failed'); end
try; gpuSparse(10,10,1,10,9); warning('failed'); end
try; gpuSparse(10,10,1,9,10); warning('failed'); end
try; gpuSparse(10,10,1,10,10,-1); warning('failed'); end
try; gpuSparse(10,10,1,10,10,Inf); warning('failed'); end
try; gpuSparse(1.5,1,1,10,10,1); warning('failed'); end
try; gpuSparse(1,1.5,1,10,10,1); warning('failed'); end
try; gpuSparse(1,1,1,10.5,10,1); warning('failed'); end
try; gpuSparse(1,1,1,10,10.5,1); warning('failed'); end
try; gpuSparse(1,1,1,10,10,1.5); warning('failed'); end
try; gpuSparse(1,1,1,10:11,10,1); warning('failed'); end
try; gpuSparse(1,1,1,10,10:11,1); warning('failed'); end
try; gpuSparse(1,1,1,10,10,1:2); warning('failed'); end

%% accuracy
disp('---ACCURACY---')

disp(['  Ax            ' num2str(norm(A*x-a*single(x),Inf))])
disp(['  A''*y          ' num2str(norm(A'*y-a'*single(y),Inf))])

B = sprandn(M,N,P);

% remove unwanted precision
[i j v] = find(B);
v = double(single(v));
B = sparse(i,j,v,M,N);

b = gpuSparse(B); validate(b)

C=A+B;
c=a+b; validate(c)
disp(['(A+B)x          ' num2str(norm(C*x-c*single(x),Inf))])
disp(['(A+B)''*y        ' num2str(norm(C'*y-c'*single(y),Inf))])

C=A-B;
c=a-b; validate(c)
disp(['(A-B)x          ' num2str(norm(C*x-c*single(x),Inf))])
disp(['(A-B)''*y        ' num2str(norm(C'*y-c'*single(y),Inf))])

d = a - (a')'; validate(d)
disp(['max(a-a'''')      ' num2str(max(d.val))])
disp(['min(a-a'''')      ' num2str(min(d.val))])

d = a - full_transpose(full_transpose(a)); validate(d)
disp(['max(a-a'''')      ' num2str(max(d.val)) '  (full_transpose)'])
disp(['min(a-a'''')      ' num2str(min(d.val)) '  (full_transpose)'])

B = double(single(randn(N,3)));
b = gpuArray(B);

C = A*B;
c = a*single(b);
disp(['(A*B-a*b)       ' num2str([norm([C-c],Inf)])])

B = double(single(randn(M,4)));
b = gpuArray(B);

C = A'*B;
c = a'*single(b);
disp(['(A''*B-a''*b)     ' num2str([norm([C-c],Inf)])])

%% miscellaneous operations

disp('---MISCELLANEOUS---')

% mixed real/complex multiplies

A = A + 1i*sprandn(A);

% remove unwanted precision
[i j v] = find(A);
v = double(single(v));
A = sparse(i,j,v,M,N);

a = gpuSparse(A); validate(a)

x = single(randn(N,1) + 1i*randn(N,1,'gpuArray'));
y = single(randn(M,1) + 1i*randn(M,1,'gpuArray'));

disp('real multiply')
disp(norm(real(A)*double(real(x)) - real(a)*real(x),Inf))
disp(norm(real(A')*double(real(y)) - real(a')*real(y),Inf))
disp(norm(real(A.')*double(real(y)) - real(a.')*real(y),Inf))

disp('complex multiply')
disp(norm(A*double(x) - a*x,Inf))
disp(norm(A'*double(y) - a'*y,Inf))
disp(norm(A.'*double(y) - a.'*y,Inf))

disp('mixed real/complex multiply')
disp(norm(A*real(double(x)) - a*real(x),Inf))
disp(norm(real(A)*double(x) - real(a)*x,Inf))
disp(norm(A'*real(double(y)) - a'*real(y),Inf))
disp(norm(A.'*real(double(y)) - a.'*real(y),Inf))
disp(norm(real(A')*double(y) - real(a')*y,Inf))
disp(norm(real(A.')*double(y) - real(a.')*y,Inf))

disp('max')
disp(norm(full(max(A,[],2)) - max(a,[],2)))

disp('sum')
disp(norm(sum(A,1) - sum(a,1),inf))
disp(norm(sum(A,2) - sum(a,2),inf))

disp('norm')
disp(norm(A,1) - norm(a,1))
disp(norm(A,inf) - norm(a,inf))
disp(norm(A,'fro') - norm(a,'fro'))

disp('sparse');
disp(norm(sparse(a)-A,inf));
disp(norm(sparse(a')-A',inf));
disp(norm(sparse(a.')-A.',inf));
disp('full_transpose(a)')
at = full_transpose(a); validate(at);
disp(norm(sparse(at)-A.',inf))
disp('full_ctranspose(a)')
at = full_ctranspose(a); validate(at);
disp(norm(sparse(at)-A',inf))
disp('full_transpose(a.'')')
att = full_transpose(a.'); validate(att);
disp(norm(sparse(att)-(A.').',inf))
disp('full_transpose(a'')')
att = full_transpose(a'); validate(att);
disp(norm(sparse(att)-(A').',inf))
disp('full_ctranspose(a.'')')
att = full_ctranspose(a.'); validate(att);
disp(norm(sparse(att)-(A.')',inf))
disp('full_ctranspose(a'')')
att = full_ctranspose(a'); validate(att);
disp(norm(sparse(att)-(A')',inf))

disp('find')
[i j v] = find(A); [i2 j2 v2] = find(a);
fprintf('   %i %i %g\n',norm(i-i2),norm(j-j2),norm(single(v)-v2))
[i j v] = find(A'); [i2 j2 v2] = find(a');
fprintf('   %i %i %g\n',norm(i-i2),norm(j-j2),norm(single(v)-v2))
[i j v] = find(A.'); [i2 j2 v2] = find(a.');
fprintf('   %i %i %g\n',norm(i-i2),norm(j-j2),norm(single(v)-v2))

% these fail - values are in different order
%disp('nonzeros')
%disp(norm(nonzeros(A)-nonzeros(a),inf))
%disp(norm(nonzeros(A')-nonzeros(a'),inf))
%disp(norm(nonzeros(A.')-nonzeros(a.'),inf))

disp('addition')
B = sprandn(M,N,P);

[i j v] = find(B); % remove unwanted precision
v = double(single(v));
B = sparse(i,j,v,M,N);

b = gpuSparse(B); validate(b)

A = real(A); B = real(B);
a = real(a); validate(a);
b = real(b); validate(b);
c = a+b; validate(c);

disp(norm((A+B) - sparse(a+b),Inf))
disp(norm((A'+B') - sparse(a'+b'),Inf))
disp(norm((A.'+B.') - sparse(a.'+b.'),Inf))
disp(norm((A+B)' - sparse((a+b)'),Inf))

disp('cat')
C = [A;B];
c = [a;b];
disp(norm(C-c,'fro'))
C = [A B];
c = [a b];
disp(norm(C-c,'fro'))

%% timings
disp('---TIMINGS---')

for j = 1:2
    
    A = gather(A);
    x = gather(x);
    y = gather(y);
    
    x = double(x);
    y = double(y);
    
    % to test mm as well as mv multiply
    if j==1
        fprintf('\n============= Matrix-vector multiply =============\n');
    else
        x = repmat(x,1,5);
        y = repmat(y,1,5);
        fprintf('\n========= Matrix-matrix multiply (cols %i) =========\n',size(x,2));
    end
    
    tic; fprintf('A*x  (sparse)   : ')
    for k = 1:20
        z = A*x; wait(gpuDevice);
    end
    toc;
    
    AT = A';
    tic; fprintf('AT*y (sparse)   : ')
    for k = 1:20
        z = AT*y; wait(gpuDevice);
    end
    toc;
    
    tic; fprintf('A''*y (sparse)   : ')
    for k = 1:20
        z = A'*y; wait(gpuDevice);
    end
    toc;
    
    A = gpuArray(A);
    x = gpuArray(x);
    y = gpuArray(y);
    
    tic; fprintf('\nA*x  (gpuArray) : ')
    for k = 1:20
        z = A*x; wait(gpuDevice);
    end
    toc;
    
    AT = A';
    tic; fprintf('AT*y (gpuArray) : ')
    for k = 1:20
        z = AT*y; wait(gpuDevice);
    end
    toc;
    
    tic; fprintf('A''*y (gpuArray) : ')
    for k = 1:20
        z = A'*y; wait(gpuDevice);
    end
    toc;
    
    a = gpuSparse(A); validate(a)
    x = single(x);
    y = single(y);
    
    tic; fprintf('\nA*x  (gpuSparse): ')
    for k = 1:20
        z = a*x; wait(gpuDevice);
    end
    toc;
    
    at = full_transpose(a); validate(at)
    tic; fprintf('At*y (gpuSparse): ')
    for k = 1:20
        z = at*y; wait(gpuDevice);
    end
    toc;
    
    tic; fprintf('A''*y (gpuSparse): ')
    for k = 1:20
        z = a'*y; wait(gpuDevice);
    end
    toc;
    
end