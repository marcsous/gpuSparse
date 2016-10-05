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

%% default mexcuda (cuda 7.5)
mexcuda csrgeam.cu -v -I/usr/local/cuda/include -L/usr/local/cuda/lib64 LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas_static -lcusparse_static -lculibos
mexcuda csrmv.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib64 LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas_static -lcusparse_static -lculibos
mexcuda coo2csr.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib64 LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas_static -lcusparse_static -lculibos
mexcuda csr2csc.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib64 LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas_static -lcusparse_static -lculibos
mexcuda csr2csc_cpu.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib64 LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas_static -lcusparse_static -lculibos
mexcuda csrmm.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib64 LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas_static -lcusparse_static -lculibos
mexcuda csr2coo.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib64 LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas_static -lcusparse_static -lculibos
mexcuda coosortByRow.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib64 LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas_static -lcusparse_static -lculibos

% not used but might as well compile it
mexcuda csrsort.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib64 LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas_static -lcusparse_static -lculibos

%% cuda 8: csrmv.cu (cusparseScsrmv_mp) doesn't work
if false
    
    files = {'csrgeam','csrmv','coo2csr','csr2csc','csr2csc_cpu','csrmm','csr2coo','coosortByRow','csrsort'};
    
    for k = 1:numel(files)
        
        cmd1 = ['/usr/local/cuda-8.0/bin/nvcc -c --compiler-options=-D_GNU_SOURCE,-DMATLAB_MEX_FILE' ...
            ' -I"/usr/local/cuda-8.0/include"' ...
            ' -I"/usr/local/MATLAB/R2016a/extern/include"' ...
            ' -I"/usr/local/MATLAB/R2016a/simulink/include"' ...
            ' -I"/usr/local/MATLAB/R2016a/toolbox/distcomp/gpu/extern/include/"' ...
            ' -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50' ...
            ' -std=c++11 --compiler-options=-ansi,-fexceptions,-fPIC,-fno-omit-frame-pointer,-pthread -O -DNDEBUG' ...
            ' ' files{k} '.cu -o ' files{k} '.o'];
        
        cmd2 = ['/usr/bin/g++ -pthread -Wl,--no-undefined -Wl,--no-as-needed -shared -O' ...
            ' -Wl,--version-script,"/usr/local/MATLAB/R2016a/extern/lib/glnxa64/mexFunction.map"' ...
            ' ' files{k} '.o' ' -ldl' ...
            ' /usr/local/cuda-8.0/targets/x86_64-linux/lib/libcusparse.so' ... % -lcusparse
            ' /usr/local/cuda-8.0/targets/x86_64-linux/lib/libcublas_static.a' ... % -lcublas_static
            ' /usr/local/cuda-8.0/targets/x86_64-linux/lib/libcusparse_static.a' ... % -lcusparse_static
            ' /usr/local/cuda-8.0/targets/x86_64-linux/lib/libculibos.a' ... % -lculibos'
            ' -L/usr/local/cuda-8.0/lib64 -Wl,-rpath-link,/usr/local/MATLAB/R2016a/bin/glnxa64' ...
            ' -L"/usr/local/MATLAB/R2016a/bin/glnxa64" -lmx -lmex -lmat -lm -lstdc++ -lmwgpu' ...
            ' /usr/local/cuda-8.0/targets/x86_64-linux/lib/libcudart.so' ... % /usr/local/MATLAB/R2016a/bin/glnxa64/libcudart.so.7.5
            ' -o ' files{k} '.mexa64'];
        
        cmd3 = ['rm -f ' files{k} '.o'];
        
        disp([files{k} '.cu'])
        if system(cmd1); error('%s failed step 1',files{k}); end
        if system(cmd2); error('%s failed step 2',files{k}); end
        if system(cmd3); error('%s failed step 3',files{k}); end
        disp('MEX completed successfully.')
        
    end
end
