# gpuSparse

Matlab mex wrappers to NVIDIA cuSPARSE (https://developer.nvidia.com/cusparse).


Uses int32 and single precision to save memory (Matlab uses int64 and double).


## Installation


1. Save in a folder called @gpuSparse on the Matlab path

2. Inside Matlab, cd to the private directory

3. Type mex_all to compile the .cu files

4. Tested with Matlab R2016a and CUDA 7.5



## Timings

size(A) = 121401 x 113331
nnz(A)  = 6877563 (0.05%)
AT      = precomputed transpose of A

A*x  (sparse)   : Elapsed time is 1.056144 seconds.
AT*y (sparse)   : Elapsed time is 1.216099 seconds.
A'*y (sparse)   : Elapsed time is 0.117227 seconds.

A*x  (gpuArray) : Elapsed time is 0.134681 seconds.
AT*y (gpuArray) : Elapsed time is 0.138336 seconds.
A'*y (gpuArray) : Elapsed time is 4.140603 seconds.

a*x  (gpuSparse): Elapsed time is 0.095376 seconds.
at*y (gpuSparse): Elapsed time is 0.093627 seconds.
a'*y (gpuSparse): Elapsed time is 4.872948 seconds.
