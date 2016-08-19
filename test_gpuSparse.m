% test gpuSparse class
clear all

M = 121401;
N = 113331;
P = 0.0005;

disp('---SETUP---')

tic; fprintf('Making sparse... ')
A = sprandn(M,N,P);
toc

tic; fprintf('Converting to gpuSparse... ')
a = gpuSparse(A); validate(a)
toc

fprintf('Sorted index conversion to gpuSparse: ')
[i j v] = find(A);
tic; 
b = gpuSparse(i,j,v,M,N); validate(b)
fprintf('errors = (%i %i %i). ',any(a.row~=b.row),any(a.col~=b.col),any(a.val~=b.val))
toc

fprintf('Unsorted index conversion to gpuSparse: ')
k = randperm(numel(v));
i = i(k);
j = j(k);
v = v(k);
tic; 
b = gpuSparse(i,j,v,M,N); validate(b)
fprintf('errors = (%i %i %i). ',any(a.row~=b.row),any(a.col~=b.col),any(a.val~=b.val))
toc

x = randn(N,1);
y = randn(M,1);

%% accuracy
disp('---ACCURACY---')

disp(['  Ax            ' num2str(norm(A*x-a*single(x),Inf))])
disp(['  A''*y          ' num2str(norm(A'*y-a'*single(y),Inf))])

B = sprandn(M,N,P);
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

B = randn(N,3);
b = gpuArray(B);

C = A*B;
c = a*single(b);
disp(['(A*B-a*b)       ' num2str([norm([C-c],Inf)])])

B = randn(M,4);
b = gpuArray(B);

C = A'*B;
c = a'*single(b);
disp(['(A''*B-a''*b)     ' num2str([norm([C-c],Inf)])])


%% miscellaneous operations

disp('---MISCELLANEOUS---')

% mixed real/complex multiplies

A = A + 1i*sprandn(A);
a = gpuSparse(A); validate(a)

x = single(randn(N,1) + 1i*randn(N,1));
y = single(randn(M,1) + 1i*randn(M,1));

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
disp(norm(real(A)*real(double(x)) - real(a)*real(x),Inf))

disp('max')
disp(norm(full(max(A,[],2)) - max(a,[],2)))

disp('sum')
disp(norm(sum(A,1) - sum(a,1),inf))
disp(norm(sum(A,2) - sum(a,2),inf))

disp('norm')
disp(norm(A,1) - norm(a,1))
disp(norm(A,inf) - norm(a,inf))
disp(norm(A,'fro') - norm(a,'fro'))

disp('full_transpose')
at = full_transpose(a); validate(at);
disp(norm(sparse(at)-A.',inf))
disp('full_ctranspose')
at = full_ctranspose(a); validate(at);
disp(norm(sparse(at)-A',inf))

disp('find')
[i j v] = find(A); [i2 j2 v2] = find(a);
fprintf('   %i %i %g\n',norm(i-i2),norm(j-j2),norm(v-v2))
[i j v] = find(A'); [i2 j2 v2] = find(a');
fprintf('   %i %i %g\n',norm(i-i2),norm(j-j2),norm(v-v2))
[i j v] = find(A.'); [i2 j2 v2] = find(a.');
fprintf('   %i %i %g\n',norm(i-i2),norm(j-j2),norm(v-v2))

disp('nonzeros')
disp(norm(nonzeros(A)-nonzeros(a),inf))
disp(norm(nonzeros(A')-nonzeros(a'),inf))
disp(norm(nonzeros(A.')-nonzeros(a.'),inf))

disp('addition')
B = sprandn(M,N,P);
b = gpuSparse(B); validate(b)

A = real(A); B = real(B);
a = real(a); validate(a);
b = real(b); validate(b);
c = a+b; validate(c);

disp(norm((A+B) - sparse(a+b),Inf))
disp(norm((A'+B') - sparse(a'+b'),Inf))
disp(norm((A.'+B.') - sparse(a.'+b.'),Inf))
disp(norm((A+B)' - sparse((a+b)'),Inf))


%% timings
disp('---TIMINGS---')

A = gather(A);
x = gather(x);
y = gather(y);

x = double(x);
y = double(y);

tic; fprintf('A*x  (native)   : ')
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

A = gpuArray(A);
x = gpuArray(x);
y = gpuArray(y);

tic; fprintf('\nA*x  (gpuArray) : ')
for k = 1:10
    z = A*x; wait(gpuDevice);
end
toc;

AT = A';
tic; fprintf('AT*y (gpuArray) : ')
for k = 1:10
    z = AT*y; wait(gpuDevice);
end
toc;

tic; fprintf('A''*y (gpuArray) : ')
for k = 1:10
    z = A'*y; wait(gpuDevice);
end
toc;


x = single(x);
y = single(y);

tic; fprintf('\na*x  (gpuSparse): ')
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

