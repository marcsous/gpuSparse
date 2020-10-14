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

A*x  (sparse)   : Elapsed time is 0.210246 seconds.
AT*y (sparse)   : Elapsed time is 0.203459 seconds.
A'*y (sparse)   : Elapsed time is 0.042892 seconds.

A*x  (gpuArray) : Elapsed time is 0.015322 seconds.
AT*y (gpuArray) : Elapsed time is 0.014198 seconds.
A'*y (gpuArray) : Elapsed time is 0.025913 seconds. <i>(CUDA 11)</i>
<s>A'*y (gpuArray) : Elapsed time is 2.156371 seconds.</s>

a*x  (gpuSparse): Elapsed time is 0.029609 seconds.
at*y (gpuSparse): Elapsed time is 0.025422 seconds.
a'*y (gpuSparse): Elapsed time is 0.023224 seconds. <i>(CUDA 11)</i>
<s>a'*y (gpuSparse): Elapsed time is 2.708314 seconds.</s>

<b>If anyone knows why gpuArray(double) is faster than gpuSparse(single) please let me know!<\b>
</pre>
