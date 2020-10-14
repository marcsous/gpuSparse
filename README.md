# gpuSparse

Matlab mex wrappers to NVIDIA cuSPARSE (https://developer.nvidia.com/cusparse).


Uses int32 and single precision to save memory (Matlab uses int64 and double).


## Installation


1. Save in a folder called @gpuSparse on the Matlab path

2. ```A = gpuSparse(rand(4))``` to trigger compilation of mex

3. <b>Recommended:</b> since CUDA-11 the transpose-multiply is much faster


## Timings
<pre>
<b>Due to memory layout (row/col-major) multiply and transpose-multiply differ in performance.</b>

size(A) = 121401 x 113331
nnz(A)  = 6877563 (0.05%)
AT      = precomputed transpose of A


A*x  (sparse)   : Elapsed time is 1.056144 seconds.
AT*y (sparse)   : Elapsed time is 1.216099 seconds.
A'*y (sparse)   : Elapsed time is 0.117227 seconds.

A*x  (gpuArray) : Elapsed time is 0.134681 seconds.
AT*y (gpuArray) : Elapsed time is 0.138336 seconds.
<s>A'*y (gpuArray) : Elapsed time is 4.140603 seconds.</s>
A'*y (gpuArray) : Elapsed time is 0.211860 seconds. <i>(CUDA 11)</i>

a*x  (gpuSparse): Elapsed time is 0.095376 seconds.
at*y (gpuSparse): Elapsed time is 0.093627 seconds.
<s>a'*y (gpuSparse): Elapsed time is 4.872948 seconds.</s>
a'*y (gpuSparse): Elapsed time is 0.073915 seconds. <i>(CUDA 11)</i>
</pre>
