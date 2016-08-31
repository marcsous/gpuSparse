% tested on Linux 64-bit with Matlab R2016a and CUDA 7.5

% checks
if ~exist('/usr/local/cuda','dir')
    warning('/usr/local/cuda directory not found. Try:\n%s','"sudo ln -s /usr/local/cuda-7.5 /usr/local/cuda"')
end

% clean
delete csrgeam.mex*
delete csrmv.mex*
delete coo2csr.mex*
delete csr2csc.mex*
delete csr2csc_cpu.mex*
delete csrmm.mex*
delete csr2coo.mex*
delete coosortByRow.mex*
delete csrsort.mex*

% make
mexcuda csrgeam.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib64 LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas_static -lcusparse_static -lculibos
mexcuda csrmv.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib64 LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas_static -lcusparse_static -lculibos
mexcuda coo2csr.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib64 LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas_static -lcusparse_static -lculibos
mexcuda csr2csc.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib64 LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas_static -lcusparse_static -lculibos
mexcuda csr2csc_cpu.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib64 LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas_static -lcusparse_static -lculibos
mexcuda csrmm.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib64 LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas_static -lcusparse_static -lculibos
mexcuda csr2coo.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib64 LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas_static -lcusparse_static -lculibos
mexcuda coosortByRow.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib64 LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas_static -lcusparse_static -lculibos

% not used but might as well compile it
mexcuda csrsort.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib64 LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas_static -lcusparse_static -lculibos

% try with cuda 8: doesn't work with Matlab 2016a
if false
    
    delete csrmv_v8.mex*
    
    cmd = ['/usr/local/cuda-8.0/bin/nvcc -c --compiler-options=-D_GNU_SOURCE,-DMATLAB_MEX_FILE' ...
           ' -I"/usr/local/cuda-8.0/include"' ...
           ' -I"/usr/local/MATLAB/R2016a/extern/include"' ...
           ' -I"/usr/local/MATLAB/R2016a/simulink/include"' ...
           ' -I"/usr/local/MATLAB/R2016a/toolbox/distcomp/gpu/extern/include/"' ...
           ' -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_50,code=\"sm_50,compute_50\"' ...
           ' -std=c++11 --compiler-options=-ansi,-fexceptions,-fPIC,-fno-omit-frame-pointer,-pthread -O -DNDEBUG' ...
           ' csrmv.cu -o csrmv_v8.o'];
    system(cmd)
    
    cmd = ['/usr/bin/g++ -pthread -Wl,--no-undefined  -Wl,--no-as-needed -shared -O' ...
           ' -Wl,--version-script,"/usr/local/MATLAB/R2016a/extern/lib/glnxa64/mexFunction.map"' ...
           ' csrmv_v8.o   -ldl  -lcusparse  -lcublas_static  -lcusparse_static  -lculibos' ...
           ' -L/usr/local/cuda-8.0/lib64 -Wl,-rpath-link,/usr/local/MATLAB/R2016a/bin/glnxa64' ...
           ' -L"/usr/local/MATLAB/R2016a/bin/glnxa64" -lmx -lmex -lmat -lm -lstdc++ -lmwgpu' ...
           ' -lcudart -o csrmv_v8.mexa64']; %' /usr/local/MATLAB/R2016a/bin/glnxa64/libcudart.so.7.5 -o csrmv_v8.mexa64'];
    system(cmd)
    
    cmd = 'rm -f csrmv_v8.o';
    system(cmd)

end



