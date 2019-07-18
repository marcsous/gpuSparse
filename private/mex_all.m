function mex_all()
% tested on Linux 64-bit with Matlab R2016a/CUDA-7.5, R2017a/CUDA-8, 2019b/CUDA-10.1

% checks
if ~exist('/usr/local/cuda','dir')
    warning('/usr/local/cuda directory not found. Try:\n%s','"sudo ln -s /usr/local/cuda-9 /usr/local/cuda"')
end

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


function mex_all_compile()

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

%% default mexcuda

mexcuda csrgeam.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib64 LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas -lculibos -lcublasLt
mexcuda csrmv.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib64 LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas -lculibos -lcublasLt
mexcuda coo2csr.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib64 LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas -lculibos -lcublasLt
mexcuda csr2csc.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib64 LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas -lculibos -lcublasLt
mexcuda csr2csc_cpu.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib64 LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas -lculibos -lcublasLt
mexcuda csrmm.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib64 LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas -lculibos -lcublasLt
mexcuda csr2coo.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib64 LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas -lculibos -lcublasLt
mexcuda coosortByRow.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib64 LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas -lculibos -lcublasLt

% not used any more
%mexcuda csrsort.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib64 LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas -lculibos -lcublasLt

