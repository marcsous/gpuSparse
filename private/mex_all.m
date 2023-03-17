function mex_all()

% checks
if ~exist('/usr/local/cuda','dir')
    warning('/usr/local/cuda directory not found. Try:\n%s','"sudo ln -s /usr/local/cuda-11 /usr/local/cuda"')
end

% override MATLAB's supplied version of nvcc - not sure what difference this makes
setenv('MW_ALLOW_ANY_CUDA','1')
setenv('MW_NVCC_PATH', '/usr/local/cuda/bin')

% need to be in the current directory for mexcuda
oldpath = pwd;
newpath = fileparts(mfilename('fullpath'));
cd(newpath);

% if the mexcuda fails, we are stuck - rethrow error
try
    mex_all_compile();
    cd(oldpath)
catch ME
    cd(oldpath)
    rethrow(ME)
end

%% call mexcuda
function mex_all_compile()

% works with either R2017b or R2018a

mexcuda csrgeam.cu      -R2018a -I/usr/local/cuda/include -L/usr/local/cuda/lib64 NVCCFLAGS='"$NVCCFLAGS -Wno-deprecated-gpu-targets"' LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas -lculibos -dynamic -v
mexcuda csrmv.cu        -R2018a -I/usr/local/cuda/include -L/usr/local/cuda/lib64 NVCCFLAGS='"$NVCCFLAGS -Wno-deprecated-gpu-targets"' LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas -lculibos -dynamic
mexcuda coo2csr.cu      -R2018a -I/usr/local/cuda/include -L/usr/local/cuda/lib64 NVCCFLAGS='"$NVCCFLAGS -Wno-deprecated-gpu-targets"' LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas -lculibos -dynamic
mexcuda csr2csc.cu      -R2018a -I/usr/local/cuda/include -L/usr/local/cuda/lib64 NVCCFLAGS='"$NVCCFLAGS -Wno-deprecated-gpu-targets"' LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas -lculibos -dynamic
mexcuda csr2csc_cpu.cu  -R2018a -I/usr/local/cuda/include -L/usr/local/cuda/lib64 NVCCFLAGS='"$NVCCFLAGS -Wno-deprecated-gpu-targets"' LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas -lculibos -dynamic
mexcuda csrmm.cu        -R2018a -I/usr/local/cuda/include -L/usr/local/cuda/lib64 NVCCFLAGS='"$NVCCFLAGS -Wno-deprecated-gpu-targets"' LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas -lculibos -dynamic
mexcuda csr2coo.cu      -R2018a -I/usr/local/cuda/include -L/usr/local/cuda/lib64 NVCCFLAGS='"$NVCCFLAGS -Wno-deprecated-gpu-targets"' LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas -lculibos -dynamic
mexcuda csrsort.cu      -R2018a -I/usr/local/cuda/include -L/usr/local/cuda/lib64 NVCCFLAGS='"$NVCCFLAGS -Wno-deprecated-gpu-targets"' LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas -lculibos -dynamic
mexcuda coosortByRow.cu -R2018a -I/usr/local/cuda/include -L/usr/local/cuda/lib64 NVCCFLAGS='"$NVCCFLAGS -Wno-deprecated-gpu-targets"' LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas -lculibos -dynamic
