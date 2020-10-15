# gpuSparse

Matlab mex wrappers to NVIDIA cuSPARSE (https://developer.nvidia.com/cusparse).


Uses int32 and single precision to save memory (Matlab sparse uses int64 and double).


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

<b>CPU sparse</b>
A*x  (sparse)   : Elapsed time is 0.702110 seconds.
AT*y (sparse)   : Elapsed time is 0.694601 seconds.
A'*y (sparse)   : Elapsed time is 0.137209 seconds.

<b>GPU sparse</b>
A*x  (gpuArray) : Elapsed time is 0.064114 seconds.
AT*y (gpuArray) : Elapsed time is 0.053819 seconds.
A'*y (gpuArray) : Elapsed time is 0.116298 seconds. <i>(CUDA 11)</i>
<s>A'*y (gpuArray) : Elapsed time is 4.156371 seconds.</s>

<b>GPU gpuSparse</b>
a*x  (gpuSparse): Elapsed time is 0.045038 seconds.
at*y (gpuSparse): Elapsed time is 0.038266 seconds.
a'*y (gpuSparse): Elapsed time is 0.038972 seconds. <i>(CUDA 11)</i>
<s>a'*y (gpuSparse): Elapsed time is 2.908314 seconds.</s>
</pre>
