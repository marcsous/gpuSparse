% test gpuSparse class
clear all

M = 121401;
N = 113331;
P = 0.0005;
TOL = 1e-4; % floating point tolerance (sparse is double and gpuSparse is single)

disp('---SETUP---')
fprintf('TOL = %e\n',TOL);

tic; fprintf('Making sparse... ')
A = sprand(M,N,P);
toc

tic; fprintf('Converting to gpuSparse... ')
a = gpuSparse(A); validate(a)
toc

x = rand(N,1);
y = rand(M,1);

%% accuracy
disp('---ACCURACY---')

disp(['  Ax            ' num2str(norm(A*x-a*single(x),Inf) > TOL)])
disp(['  A''*y          ' num2str(norm(A'*y-a'*single(y),Inf) > TOL)])

B = sprand(M,N,P);
b = gpuSparse(B); validate(b)

C=A+B;
c=a+b; validate(c)
disp(['(A+B)x          ' num2str(norm(C*x-c*single(x),Inf) > TOL)])
disp(['(A+B)''*y        ' num2str(norm(C'*y-c'*single(y),Inf) > TOL)])

C=A-B;
c=a-b; validate(c)
disp(['(A-B)x          ' num2str(norm(C*x-c*single(x),Inf) > TOL)])
disp(['(A-B)''*y        ' num2str(norm(C'*y-c'*single(y),Inf) > TOL)])

d = a - (a')'; validate(d)
disp(['max(a-a'''')      ' num2str(max(d.val) > TOL)])
disp(['min(a-a'''')      ' num2str(min(d.val) > TOL)])

d = a - full_transpose(full_transpose(a)); validate(d)
disp(['max(a-a'''')      ' num2str(max(d.val) > TOL) '  (full_transpose)'])
disp(['min(a-a'''')      ' num2str(min(d.val) > TOL) '  (full_transpose)'])

B = randn(N,3);
b = gpuArray(B);

C = A*B;
c = a*single(b);
disp(['(A*B-a*b)       ' num2str([norm([C-c],Inf)] > TOL)])

B = randn(M,4);
b = gpuArray(B);

C = A'*B;
c = a'*single(b);
disp(['(A''*B-a''*b)     ' num2str([norm([C-c],Inf)] > TOL)])


%% miscellaneous operations

disp('---MISCELLANEOUS---')

% mixed real/complex multiplies

A = A + i*sprand(A);
a = gpuSparse(A); validate(a)

x = single(rand(N,1) + i*rand(N,1));
y = single(rand(M,1) + i*rand(M,1));

disp('complex multiplies')
disp(norm(A*double(x) - a*x,Inf) > TOL)
disp(norm(A'*double(y) - a'*y,Inf) > TOL)
disp(norm(A.'*double(y) - a.'*y,Inf) > TOL)

disp('mixed real/complex multiplies')
disp(norm(A*real(double(x)) - a*real(x),Inf) > TOL)
disp(norm(real(A)*double(x) - real(a)*x,Inf) > TOL)
disp(norm(A'*real(double(y)) - a'*real(y),Inf) > TOL)
disp(norm(A.'*real(double(y)) - a.'*real(y),Inf) > TOL)
disp(norm(real(A')*double(y) - real(a')*y,Inf) > TOL)
disp(norm(real(A.')*double(y) - real(a.')*y,Inf) > TOL)
disp(norm(real(A)*real(double(x)) - real(a)*real(x),Inf) > TOL)

disp('max')
disp(norm(full(max(A,[],2)) - max(a,[],2)) > TOL)

disp('sum')
disp(norm(sum(A,1) - sum(a,1),inf) > TOL)
disp(norm(sum(A,2) - sum(a,2),inf) > TOL)

disp('norm')
disp(norm(A,1) - norm(a,1) > TOL)
disp(norm(A,inf) - norm(a,inf) > TOL)
disp(norm(A,'fro') - norm(a,'fro') > TOL)

disp('full_transpose')
at = full_transpose(a); validate(at);
disp(norm(sparse(at)-A.',inf) > TOL)
disp('full_ctranspose')
at = full_ctranspose(a); validate(at);
disp(norm(sparse(at)-A',inf) > TOL)

disp('find')
[i j v] = find(A); [i2 j2 v2] = find(a);
disp(any([norm(i-i2) norm(j-j2) norm(v-v2)] > TOL))
[i j v] = find(A'); [i2 j2 v2] = find(a');
disp(any([norm(i-i2) norm(j-j2) norm(v-v2)] > TOL))
[i j v] = find(A.'); [i2 j2 v2] = find(a.');
disp(any([norm(i-i2) norm(j-j2) norm(v-v2)] > TOL))

disp('nonzeros')
disp(norm(nonzeros(A)-nonzeros(a),inf) > TOL)
disp(norm(nonzeros(A')-nonzeros(a'),inf) > TOL)
disp(norm(nonzeros(A.')-nonzeros(a.'),inf) > TOL)

disp('addition')
B = sprandn(M,N,P);
b = gpuSparse(B); validate(b)

A = real(A); B = real(B);
a = real(a); validate(a);
b = real(b); validate(b);
c = a+b; validate(c);

disp(norm((A+B) - sparse(a+b),Inf) > TOL)
disp(norm((A'+B') - sparse(a'+b'),Inf) > TOL)
disp(norm((A.'+B.') - sparse(a.'+b.'),Inf) > TOL)
disp(norm((A+B)' - sparse((a+b)'),Inf) > TOL)


%% timings
disp('---TIMINGS---')

A = gpuArray(A);
x = gpuArray(x);
y = gpuArray(y);

x = double(x);
y = double(y);

tic; fprintf('A*x (native)    : ')
for k = 1:10
    z = A*x; wait(gpuDevice);
end
toc;

AT = A';
tic; fprintf('AT*y (native)   : ')
for k = 1:10
    z = AT*y; wait(gpuDevice);
end
toc;

tic; fprintf('A''*y (native)   : ')
for k = 1:10
    z = A'*y; wait(gpuDevice);
end
toc;

x = single(x);
y = single(y);

tic; fprintf('a*x (gpuSparse) : ')
for k = 1:10
    z = a*x; wait(gpuDevice);
end
toc;

at = full_transpose(a); validate(at)
tic; fprintf('at*y (gpuSparse): ')
for k = 1:10
    z = at*y; wait(gpuDevice);
end
toc;

tic; fprintf('a''*y (gpuSparse): ')
for k = 1:10
    z = a'*y; wait(gpuDevice);
end
toc;

