classdef gpuSparse
%%
% Sparse GPU array class (mex wrappers to cuSPARSE).
% Arguments are the same as matlab's sparse function.
% Usage: A = gpuSparse(row,col,val,nrows,ncols,nzmax)
%
%%
    properties (SetAccess=private, SetObservable=true)

        nrows @ int32 = int32(0); % number of rows
        ncols @ int32 = int32(0); % number of columns
        
        row @ gpuArray = zeros(0,'int32','gpuArray'); % row index (CSR format)
        col @ gpuArray = zeros(0,'int32','gpuArray'); % column index
        val @ gpuArray = zeros(0,'single','gpuArray'); % nonzero values
        
        % flag to hold the transpose state of the matrix
        %  0 = CUSPARSE_OPERATION_NON_TRANSPOSE
        %  1 = CUSPARSE_OPERATION_TRANSPOSE
        %  2 = CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE
        transp @ int32 = int32(0);

    end
    
%%
    methods
        
        % constructor: accepts 0, 1, 2, 3, 5 or 6 arguments
        function A = gpuSparse(row,col,val,nrows,ncols,nzmax)

            % check arguments
            if nargin==0; return; end % necessary for constructor
            if nargin==4 || nargin>6
                error('Wrong number of arguments.');
            end

            % catch oddball first argument ("row" is the first argument)
            if ~isnumeric(row)
                if nargin==1 && isa(row,class(A)); A = row; return; end
                error('row type %s not supported.',class(row));
            end

            % convert scalar, vector or matrix to gpuSparse format
            if nargin==1
                [nrows ncols] = size(row); % "row" is the first argument
                [row col val] = find(row); % not efficient: row unsorted
            end

            % create empty m x n matrix
            if nargin==2
                nrows = row; ncols = col;
                row = []; col = []; val = [];
            end
            
            % check sizes
            if numel(row)~=numel(val); error('row and val size mismatch.'); end
            if numel(col)~=numel(val); error('col and val size mismatch.'); end

            % check indices are integer
            validateattributes(row,{'numeric','gpuArray'},{'integer'},'','row');
            validateattributes(col,{'numeric','gpuArray'},{'integer'},'','col');

            % rows must be sorted for COO to CSR conversion
            if ~issorted(row)
                [row k] = sort(row);
                col = col(k);
                val = val(k);
            end

            % cast to required classes
            val = single(val(:));
            col = int32(col(:));
            row = int32(row(:)); % COO format

            % check lower bounds
            if numel(row) > 0
                if row(1)<1; error('All row indices must be greater than zero.'); end
                if min(col)<1; error('All col indices must be greater than zero.'); end
                A.nrows = gather(row(end));
                A.ncols = gather(max(col));
            end

            % check and apply user-supplied sizes
            if exist('nrows','var')
                nrows = gather(nrows);
                validateattributes(nrows,{'numeric'},{'scalar','integer','>=',A.nrows},'','nrows');
                A.nrows = int32(nrows);
            end
            if exist('ncols','var')
                ncols = gather(ncols);
                validateattributes(ncols,{'numeric'},{'scalar','integer','>=',A.ncols},'','ncols');
                A.ncols = int32(ncols);
            end

			% check upper bounds
			if A.nrows==intmax('int32')
				error('Number of rows equals or exceeds int32 range (%i).',intmax('int32'))
			end
			if A.ncols==intmax('int32')
				error('Number of columns equals or exceeds int32 range (%i).',intmax('int32'))
            end

             % estimate memory required to create matrix
            info = gpuDevice();
            AvailableMemory = info.AvailableMemory / 1E9;
            if nargin < 6
                nzmax = numel(val);
            else
                nzmax = gather(nzmax);
                validateattributes(nzmax,{'numeric'},{'scalar','integer'},'','nzmax');
                nzmax = max(double(nzmax),numel(val));
            end
            RequiredMemory = 4*double(A.nrows+1)/1E9;
            if ~isa(val,'gpuArray'); RequiredMemory = RequiredMemory+4*nzmax/1E9; end
            if ~isa(col,'gpuArray'); RequiredMemory = RequiredMemory+4*nzmax/1E9; end
            if RequiredMemory > AvailableMemory
                error('Not enough memory (%.1fGb required, %.1fGb available).',RequiredMemory,AvailableMemory);
            end
            
            % finally convert from COO to CSR format
            A.row = coo2csr(row,A.nrows);
            A.col = gpuArray(col);
            A.val = gpuArray(val);
 
        end
        
        % validation - helpful for testing
        function validate(A)

            message = 'Validation failure. Check line for details.';
            
            % fast checks
            if ~isa(A.nrows,'int32'); error(message); end
            if ~isa(A.ncols,'int32'); error(message); end
            if ~isa(A.transp,'int32'); error(message); end
            if ~isa(A.row,'gpuArray'); error(message); end
            if ~isa(A.col,'gpuArray'); error(message);end
            if ~isa(A.val,'gpuArray'); error(message); end
            if ~isequal(classUnderlying(A.row),'int32'); error(message); end
            if ~isequal(classUnderlying(A.col),'int32'); error(message); end
            if ~isequal(classUnderlying(A.val),'single'); error(message); end
            if A.nrows < 0; error(message); end   
            if A.ncols < 0; error(message); end    
            if numel(A.col) ~= numel(A.val); error(message); end
            if numel(A.row) ~= A.nrows+1; error(message); end
            if A.transp<0 || A.transp>2; error(message); end
            if numel(A.row) > 0
                if A.row(1) ~= 1; error(message); end
                if A.row(end) ~= numel(A.val)+1; error(message); end
            end

            % slow checks
            if numel(A.row) > 0
                if ~issorted(A.row); error(message); end
                if min(A.col) < 1; error(message); end
                if max(A.col) > A.ncols; error(message); end
            end

        end
        
        % overload size
        function varargout = size(A,dim)
            if A.transp==0
                m = double(A.nrows);
                n = double(A.ncols);
            else
                n = double(A.nrows);
                m = double(A.ncols);
            end
            if nargin>1
                if nargout>1
                    error('too many output arguments.');
                end
                if dim==1
                    varargout{1} = m;
                elseif dim==2
                    varargout{1} = n;
                else
                    varargout{1} = 1;
                end
            else
                if nargout==0 || nargout==1
                    varargout{1} = [m n];
                else
                    varargout{1} = m;
                    varargout{2} = n;
                    for k = 3:nargout
                        varargout{k} = 1;
                    end
                end
            end
        end
        
        % wrapper to csrgeam: C = a*A + b*B
        function C = geam(A,B,a,b)
            if ~isequal(class(A),class(B))
                error('No method for adding %s and %s.',class(A),class(B))
            end
            if ~isequal(size(A),size(B))
                error('Matrices must be the same size.')
            end
            if A.transp ~= 0 || A.transp ~= 0
                error('Matrix lazy-transpose not supported. Use full_transpose.')
            end
            validateattributes(a,{'numeric'},{'real','scalar','finite'},'','a');
            validateattributes(b,{'numeric'},{'real','scalar','finite'},'','b');
            [m n] = size(A);
            C = gpuSparse(m,n);
            [C.row C.col C.val] = csrgeam(A.row,A.col,A.val,m,n,B.row,B.col,B.val,a,b);
        end
  
        % overload add: C = A+B
        function C = plus(A,B)
            C = geam(A,B,1,1);
        end
 
        % overload minus: C = A-B
        function C = minus(A,B)
            C = geam(A,B,1,-1);
        end
        
        % overload isreal
        function retval = isreal(A)
            retval = isreal(A.val);
        end
        
        % overload mtimes: A * x
        function y = mtimes(A,x)
            if isempty(x)
                error('Argument x is empty.')
            elseif ~isnumeric(x)
                error('Argument x must be numeric.')
            elseif isscalar(x)
                y = A;
                y.val = y.val * x;
            elseif isvector(x)
                if isreal(A) == isreal(x)
                    y = csrmv(A.row,A.col,A.val,A.nrows,A.ncols,A.transp,x);
                elseif isreal(A)
                    y = complex(csrmv(A.row,A.col,A.val,A.nrows,A.ncols,A.transp,real(x)), ...
                                csrmv(A.row,A.col,A.val,A.nrows,A.ncols,A.transp,imag(x)));
                else
                    y = complex(csrmv(A.row,A.col,real(A.val),A.nrows,A.ncols,A.transp,x), ...
                                csrmv(A.row,A.col,imag(A.val),A.nrows,A.ncols,A.transp,x));
                    if A.transp == 2; y = conj(y); end
                end
            elseif ismatrix(x)
                if ~isreal(A)
                    error('Complex A not supported at the moment.')
                end
                if isreal(x)
                    y = csrmm(A.row,A.col,A.val,A.nrows,A.ncols,A.transp,x);
                else
                    y = complex(csrmm(A.row,A.col,A.val,A.nrows,A.ncols,A.transp,real(x)), ...
                                csrmm(A.row,A.col,A.val,A.nrows,A.ncols,A.transp,imag(x)));
                end
            else
                error('Argument x has too many dimensions.')
            end
        end
        
        % overload times: A .* x
        function y = times(A,x)
            if isempty(x)
                error('Argument x is empty.')
            elseif ~isnumeric(x)
                error('Argument x must be numeric.') 
            elseif isscalar(x)
                y = A;
                y.val = y.val .* x;
            else
                error('Argument x has too many dimensions.')
            end
        end
        
        % full transpose: A.'
        function A_t = full_transpose(A) 
            [m n] = size(A);
            A_t = gpuSparse([],[],[],n,m,nnz(A));
            A_t.transp = int32(0);

            % cuSPARSE version uses a lot of memory
            %[A_t.col A_t.row A_t.val] = csr2csc(A.row,A.col,A.val,m,n);

            % CPU version is slower but has no memory problems
            row = gather(A.row);
            col = gather(A.col);
            val = gather(A.val);
            [col row val] = csr2csc_cpu(row,col,val,m,n);
            A_t.col = gpuArray(col);
            A_t.row = gpuArray(row);
            A_t.val = gpuArray(val);
        end
        
        % full ctranspose: A'
        function A_h = full_ctranspose(A)
            A_h = full_transpose(A);
            A_h.val = conj(A_h.val);
        end

        % lazy transpose (flag): A.'
        function A_t = transpose(A) 
            A_t = A; % lazy copy
            switch A.transp
                case 0; A_t.transp = int32(1);
                case 1; A_t.transp = int32(0);
                case 2; A_t.transp = int32(0); A_t.val = conj(A_t.val);
            end
        end
        
        % lazy transpose (flag): A'
        function A_t = ctranspose(A)
            A_t = A; % lazy copy
            switch A_t.transp
                case 0; if isreal(A_t); A_t.transp = int32(1); else A_t.transp = int32(2); end
                case 1; A_t.transp = int32(0); if ~isreal(A_t); A_t.val = conj(A_t.val); end
                case 2; A_t.transp = int32(0);
            end
        end

        % overload nnz
        function retval = nnz(A)
            retval = numel(A.val);
        end
        
        % overload nonzeros
        function val = nonzeros(A)
            val = A.val;
        end
        
        % overload nzmax (basically a placeholder)
        function retval = nzmax(A)
            retval = numel(A.val);
        end
        
        % remove zeros from sparse matrix
        function A = remove_zeros(A,threshold)
            nonzeros = (A.val ~= 0);
            if nargin<2; threshold = 0.9; end
            if sum(nonzeros) < threshold * numel(A.val)
                A.row = csr2coo(A.row,A.nrows);
                A.row = coo2csr(A.row(nonzeros),A.nrows);
                A.col = A.col(nonzeros);
                A.val = A.val(nonzeros);
            end
        end

        % overload classUnderlying
        function str = classUnderlying(A)
            str = classUnderlying(A.val);
        end
        
        % overload real
        function B = real(A);
            B = A;
            B.val = real(A.val);
            B = remove_zeros(B);
        end
        
        % overload imag
        function B = imag(A);
            B = A;
            B.val = imag(A.val);
            B = remove_zeros(B);
        end
        
        % overload find: not efficient, just for debugging
        function varargout = find(A)
            if nargin>1; error('only 1 input argument supported'); end
            if nargout>3; error('too many ouput arguments'); end
 
            % MATLAB style, sorted columns
            j = gather(A.col);
            [j k] = sort(j);
            j = double(j);

            i = csr2coo(A.row,A.nrows);
            i = gather(i);
            i = double(i(k));

            if nargout==0 || nargout==1
                varargout{1} = sub2ind(size(A),i,j);
            else
                varargout{1} = i;
                varargout{2} = j;
            end
            if nargout==3
                v = gather(A.val);
                varargout{3} = v(k);
            end
        end
        
        % overload sparse (returns sparse matrix on CPU): not efficient, for debugging
        function A_sp = sparse(A)
            [m n] = size(A);
            [i j v] = find(A);
            A_sp = sparse(i,j,double(v),m,n); % args must be double
        end

        % overload full (returns full matrix on CPU): not efficient, for debugging
        function A_f = full(A)
            A_sp = sparse(A);
            A_f = cast(full(A_sp),classUnderlying(A));
        end
        
    end
    
end
