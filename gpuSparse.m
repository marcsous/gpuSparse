classdef gpuSparse
    %%
    % Sparse GPU array class (mex wrappers to cuSPARSE)
    % using int32 indices and single precision values.
    %
    % Arguments are the same as matlab's sparse function.
    %
    % If supplied, nzmax is used to check memory before
    % creation: gpuSparse([],[],[],nrows,ncols,nzmax)
    %
    % Usage: A = gpuSparse(row,col,val,nrows,ncols,nzmax)

    % TO DO
    % 1) Test with CUDA 8 and Matlab > R2016a
    % 2) Revisit csr2csc on gpu with CUDA 8
    % 3) Native mixed real/complex operations

    %%
    properties (SetAccess = immutable)
        
        nrows @ int32 scalar; % number of rows
        ncols @ int32 scalar; % number of columns
        
    end
    
    properties (SetAccess = private)
        
        row @ gpuArray; % int32 row index (CSR format)
        col @ gpuArray; % int32 column index
        val @ gpuArray; % single precision values

        trans @ int32 scalar; % lazy transpose flag (passed to cuSPARSE)
                              % 0 = CUSPARSE_OPERATION_NON_TRANSPOSE
                              % 1 = CUSPARSE_OPERATION_TRANSPOSE
                              % 2 = CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE
    end
    
    %%
    methods
        
        %% constructor: same syntax as matlab's sparse
        function A = gpuSparse(row,col,val,nrows,ncols,nzmax)

            % empty gpuSparse matrix
            if nargin==0
                row = []; col = []; val = [];
            end

            % expect a matrix, return gpuSparse ("row" is the first argument)
            if nargin==1
                if isa(row,'gpuSparse'); A = row; return; end % return unchanged
                if ~isnumeric(row); error('Cannot convert ''%s'' to gpuSparse.',class(row)); end
                if ~ismatrix(row); error('Cannot convert ND array to gpuSparse.'); end
                [nrows ncols] = size(row);
                [row col val] = find(row);
            end

            % empty m x n matrix
            if nargin==2
                nrows = row; ncols = col;
                row = []; col = []; val = [];
            end
            
            % catch illegal no. arguments
            if nargin==4 || nargin>6
                error('Wrong number of arguments.');
            end

            % validate argument types
            validateattributes(row,{'numeric','gpuArray'},{'integer'},'','row');
            validateattributes(col,{'numeric','gpuArray'},{'integer'},'','col');
            validateattributes(val,{'numeric','gpuArray'},{},'','val');

            % check vector lengths
            row = reshape(row,[],1);
            col = reshape(col,[],1);
            val = reshape(val,[],1);            
            if numel(row)~=numel(col)
                error('Vectors must be the same length (row=%i col=%i).',numel(row),numel(col));
            end
            if numel(val)==1
                val = repmat(val,numel(row),1);
            elseif numel(val)~=numel(row)
                error('Vectors must be the same length (row=%i val=%i).',numel(row),numel(val));
            end

            % check bounds of indices
            if numel(row) > 0
                A.nrows = gather(max(row));
                if min(row)<1 || A.nrows==intmax('int32')
                    error('row indices must be between 1 and %i.',intmax('int32')-1);
                end
                A.ncols = gather(max(col));
                if min(col)<1 || A.ncols==intmax('int32')
                    error('col indices must be between 1 and %i.',intmax('int32')-1);
                end
            end

            % check and apply user-supplied matrix dims
            if exist('nrows','var')
                nrows = gather(nrows);
                validateattributes(nrows,{'numeric'},{'scalar','integer','>=',A.nrows,'<',intmax('int32')},'','nrows');
                A.nrows = nrows;
            end
            if exist('ncols','var')
                ncols = gather(ncols);
                validateattributes(ncols,{'numeric'},{'scalar','integer','>=',A.ncols,'<',intmax('int32')},'','ncols');
                A.ncols = ncols;
            end

            % simple memory check - needs work
            info = gpuDevice();
            AvailableMemory = info.AvailableMemory / 1E9;
            if ~exist('nzmax','var')
                nzmax = numel(val);
            else
                nzmax = gather(nzmax);
                validateattributes(nzmax,{'numeric'},{'scalar','integer','>=',numel(val)},'','nzmax');
            end
            RequiredMemory = 4*double(A.nrows+1)/1E9;
            RequiredMemory = RequiredMemory+4*double(nzmax)/1E9;
            RequiredMemory = RequiredMemory+4*double(nzmax)/1E9;
            if RequiredMemory > AvailableMemory
                error('Not enough memory (%.1fGb required, %.1fGb available).',RequiredMemory,AvailableMemory);
            end

            % cast to required class
            row = int32(row);
            col = int32(col);
            val = single(val);

            % sort row and col for COO to CSR conversion - attempt to recompile mex files if there's an error
            try

                [A.row A.col A.val] = coosortByRow(row,col,val,A.nrows,A.ncols);
                
            catch ME
                
                warning('%s Attempting to recompile mex files...',ME.message);
                mex_all;
                [A.row A.col A.val] = coosortByRow(row,col,val,A.nrows,A.ncols);

            end
            
            % convert from COO to CSR
            A.row = coo2csr(A.row,A.nrows);

        end
        
        %% enforce some class properties - inexpensive checks only
        function A = set.row(A,row)
            if ~iscolumn(row) || ~isequal(classUnderlying(row),'int32')
                error('Property row must be a column vector of int32s.')
            end
            A.row = row;
        end
        function A = set.col(A,col)
            if ~iscolumn(col) || ~isequal(classUnderlying(col),'int32')
                error('Property col must be a column vector of int32s.')
            end
            A.col = col;
        end
        function A = set.val(A,val)
            if ~iscolumn(val) || ~isequal(classUnderlying(val),'single')
                error('Property val must be a column vector of singles.')
            end
            A.val = val;
        end
        function A = set.trans(A,trans)
            if trans~=0 && trans~=1 && trans~=2
                error('Property trans must be 0, 1 or 2.')
            end
            A.trans = trans;
        end

        %% validation - helpful for testing
        function validate(A)
            
            message = 'Validation failure.';
            
            % fast checks
            if ~isa(A.nrows,'int32'); error(message); end
            if ~isa(A.ncols,'int32'); error(message); end
            if ~isa(A.trans,'int32'); error(message); end
            if ~isa(A.row,'gpuArray'); error(message); end
            if ~isa(A.col,'gpuArray'); error(message);end
            if ~isa(A.val,'gpuArray'); error(message); end
            if ~isequal(classUnderlying(A.row),'int32'); error(message); end
            if ~isequal(classUnderlying(A.col),'int32'); error(message); end
            if ~isequal(classUnderlying(A.val),'single'); error(message); end
            if A.nrows < 0; error(message); end
            if A.ncols < 0; error(message); end
            if A.nrows == intmax('int32'); error(message); end
            if A.ncols == intmax('int32'); error(message); end
            if ~iscolumn(A.row); error(message); end
            if ~iscolumn(A.col); error(message); end
            if ~iscolumn(A.val); error(message); end
            if numel(A.col) ~= numel(A.val); error(message); end
            if numel(A.row) ~= A.nrows+1; error(message); end
            if A.row(1) ~= 1; error(message); end
            if A.row(end) ~= numel(A.val)+1; error(message); end
            if A.trans~=0 && A.trans~=1 && A.trans~=2; error(message); end

            % slow checks
            if numel(A.val) > 0
                if min(A.col) < 1; error(message); end
                if max(A.col) > A.ncols; error(message); end
                rowcol = gather([csr2coo(A.row,A.nrows) A.col]);
                if ~issorted(rowcol,'rows'); error(message); end
            end
            
        end
        
        %% overloaded functions
        
        % isreal
        function retval = isreal(A)
            retval = isreal(A.val);
        end
        
        % real
        function B = real(A);
            B = A;
            B.val = real(A.val);
            %B = drop_zeros(B);
        end
        
        % imag
        function B = imag(A);
            B = A;
            B.val = imag(A.val);
            %B = drop_zeros(B);
        end
        
        % abs
        function B = abs(A);
            B = A;
            B.val = abs(A.val);
        end
        
        % angle
        function B = angle(A);
            B = A;
            B.val = angle(A.val);
        end
        
        % conj
        function B = conj(A);
            B = A;
            B.val = conj(A.val);
        end
        
        % sign
        function B = sign(A);
            B = A;
            B.val = sign(A.val);
        end
        
        % complex
        function B = complex(A);
            B = A;
            B.val = complex(A.val);
        end

        % classUnderlying
        function str = classUnderlying(A)
            str = classUnderlying(A.val);
        end
        
        % nnz
        function retval = nnz(A)
            retval = nnz(A.val);
        end
        
        % numel
        function retval = numel(A)
            retval = prod(size(A));
        end
        
        % length
        function retval = length(A)
            retval = max(size(A));
        end
        
        % nzmax
        function retval = nzmax(A)
            retval = numel(A.val);
        end
        
        % mean
        function retval = mean(A,DIM,varargin)
            if nargin < 2; DIM = 1; end
            if ~isequal(DIM,1) && ~isequal(DIM,2); error('Dimension value not supported.'); end
            retval = sum(A,DIM) / size(A,DIM);
        end
        
        % nonzeros
        function val = nonzeros(A)
            if A.trans==0
                [~,~,val] = find(A);
            else
                val = nonzeros(A.val);
                if A.trans==2
                    val = conj(A.val);
                end
            end
        end

        % sum: only A and DIM args are supported
        function retval = sum(A,DIM)
            if nargin==1
                DIM = 1;
            else
                validateattributes(DIM,{'numeric'},{'integer','positive'},'','DIM')
            end
            if numel(A)==0
                retval = sum(zeros(size(A)),DIM);
                retval = gpuSparse(retval);
            else
                if DIM==1; retval =(A'* ones(size(A,1),1,'single','gpuArray'))'; end
                if DIM==2; retval = A * ones(size(A,2),1,'single','gpuArray'); end
                if DIM>2; retval = A; end
            end
        end
        
        % norm: support same types as sparse
        function retval = norm(A,p);
            if nargin<2; p = 2; end
            if isvector(A)
                retval = norm(A.val,p);
            else
                if isequal(p,2)
                    error('gpuSparse norm(A,2) is not available.');
                elseif isequal(p,1)
                    retval = max(sum(abs(A),1));
                elseif isequal(p,Inf)
                    retval = max(sum(abs(A),2));
                elseif isequal(p,'fro');
                    retval = norm(A.val);
                else
                    error('The only matrix norms available are 1, 2, inf, and ''fro''.');
                end
            end
        end
        
        % max: support for max(A,[],2) only
        function retval = max(A,Y,DIM);
            if nargin ~= 3 || ~isempty(Y) || ~isequal(DIM,2)
                error('Only 3 argument form supported: max(A,[],2).');
            end
            if A.trans
                error('Transpose max not supported - try full_transpose(A).')
            end
            
            % do it on CPU to reduce transfer overhead
            row = gather(A.row);
            val = gather(A.val);
            retval = zeros(A.nrows,1,'like',val);
            
            for j = 1:A.nrows
                k = row(j):row(j+1)-1;
                if ~isempty(k)
                    retval(j) = max(val(k));
                end
            end
        end
        
        % size
        function varargout = size(A,DIM)
            if A.trans==0
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
                if DIM==1
                    varargout{1} = m;
                elseif DIM==2
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
        
        % find: returns indices on the GPU (not efficient, mainly for debugging)
        function varargout = find(A)
            if nargin>1; error('only 1 input argument supported'); end
            if nargout>3; error('too many ouput arguments'); end
            
            % COO format on GPU
            i = csr2coo(A.row,A.nrows);
            j = A.col;
            v = A.val;

            % remove explicit zeros
            nz = (v ~= 0);
            i = i(nz);
            j = j(nz);
            v = v(nz);

            % MATLAB style, double precision, sorted columns
            if A.trans
                [i j] = deal(j,i);
            else
                [~,k] = sortrows([j i]);
                i = i(k);
                j = j(k);
            end
            i = double(i);
            j = double(j);

            if nargout==0 || nargout==1
                varargout{1} = sub2ind(size(A),i,j);
            else
                varargout{1} = i;
                varargout{2} = j;
            end
            if nargout==3
                if A.trans==0; varargout{3} = v(k); end
                if A.trans==1; varargout{3} = v; end
                if A.trans==2; varargout{3} = conj(v); end
            end
        end
        
        % add: C = A+B
        function C = plus(A,B)
            C = geam(A,B,1,1);
        end
        
        % minus: C = A-B
        function C = minus(A,B)
            C = geam(A,B,1,-1);
        end
        
        % csrgeam: C = a*A + b*B
        function C = geam(A,B,a,b)
            if ~isequal(class(A),class(B))
                error('No method for adding %s and %s.',class(A),class(B))
            end
            if ~isequal(size(A),size(B))
                error('Matrices must be the same size.')
            end
            if ~isreal(A) || ~isreal(B)
                error('Complex addition not supported at the moment.')
            end
            if A.trans ~= B.trans
                error('Matrix lazy transpose not fully supported. Use full_transpose.')
            end
            validateattributes(a,{'numeric'},{'real','scalar','finite'},'','a');
            validateattributes(b,{'numeric'},{'real','scalar','finite'},'','b');
            if A.trans
                [n m] = size(A);
            else
                [m n] = size(A);
            end
            C = gpuSparse(m,n);
            C.trans = A.trans;
            [C.row C.col C.val] = csrgeam(A.row,A.col,A.val,m,n,B.row,B.col,B.val,a,b);
        end
        
        % mtimes: A * x
        function y = mtimes(A,x)
            if isempty(x)
                error('Argument x is empty.')
            elseif ~isnumeric(x)
                error('Argument x must be numeric (%s not supported).',class(x))
            elseif isscalar(x) && ~iscolumn(A)
                y = A;
                y.val = y.val * x;
            elseif isvector(x)
                if isreal(A) == isreal(x)
                    y = csrmv(A.row,A.col,A.val,A.nrows,A.ncols,A.trans,x);
                elseif isreal(A)
                    y = complex(csrmv(A.row,A.col,A.val,A.nrows,A.ncols,A.trans,real(x)), ...
                                csrmv(A.row,A.col,A.val,A.nrows,A.ncols,A.trans,imag(x)));
                elseif isreal(x)
                    y = csrmv(A.row,A.col,A.val,A.nrows,A.ncols,A.trans,complex(x));
                else
                    error('Should never get here.')
                end
            elseif ismatrix(x)
                if ~isreal(A)
                    error('Complex A not supported at the moment.')
                end
                if isreal(x)
                    y = csrmm(A.row,A.col,A.val,A.nrows,A.ncols,A.trans,x);
                else
                    y = complex(csrmm(A.row,A.col,A.val,A.nrows,A.ncols,A.trans,real(x)), ...
                                csrmm(A.row,A.col,A.val,A.nrows,A.ncols,A.trans,imag(x)));
                end
            else
                error('Argument x has too many dimensions.')
            end
        end
        
        % times: A .* x
        function y = times(A,x)
            if isempty(x)
                error('Argument x is empty.')
            elseif ~isnumeric(x)
                error('Argument x must be numeric (%s not supported).',class(x))
            elseif isscalar(x)
                y = A;
                y.val = y.val .* x;
            else
                error('Argument x has too many dimensions.')
            end
        end
        
        % full transpose: A.'
        function A_t = full_transpose(A)
            if A.trans
                A_t = A;
                A_t.trans = 0;
                if A.trans==2 && ~isreal(A)
                    A_t.val = conj(A_t.val);
                end
            else
                [m n] = size(A);
                A_t = gpuSparse([],[],[],n,m,nnz(A));
                
                % cuSPARSE version uses excessive memory, prefer CPU version
                if false
                    [A_t.col A_t.row A_t.val] = csr2csc(A.row,A.col,A.val,m,n);
                else
                    row = gather(A.row);
                    col = gather(A.col);
                    val = gather(A.val);
                    [col row val] = csr2csc_cpu(row,col,val,m,n);
                    A_t.col = gpuArray(col);
                    A_t.row = gpuArray(row);
                    A_t.val = gpuArray(val);
                end
            end
        end
        
        % full ctranspose: A'
        function A_t = full_ctranspose(A)
            if A.trans
                A_t = A;
                A_t.trans = 0;
            else
                A_t = full_transpose(A);
            end
            if A.trans~=2 && ~isreal(A)
                A_t.val = conj(A_t.val);
            end
        end

        % lazy transpose (flag): A.'
        function A_t = transpose(A)
            A_t = A; % lazy copy
            switch A.trans
                case 0; A_t.trans = 1;
                case 1; A_t.trans = 0;
                case 2; A_t.trans = 0; A_t.val = conj(A_t.val);
            end
        end
        
        % lazy transpose (flag): A'
        function A_t = ctranspose(A)
            A_t = A; % lazy copy
            switch A_t.trans
                case 0; if isreal(A_t); A_t.trans = 1; else A_t.trans = 2; end
                case 1; A_t.trans = 0; if ~isreal(A_t); A_t.val = conj(A_t.val); end
                case 2; A_t.trans = 0;
            end
        end

        % remove zeros from sparse matrix
        function A = drop_zeros(A)
            nonzeros = (A.val ~= 0);
            if ~all(nonzeros)
                A.row = csr2coo(A.row,A.nrows);
                A.row = A.row(nonzeros);
                A.row = coo2csr(A.row,A.nrows);
                A.col = A.col(nonzeros);
                A.val = A.val(nonzeros);
            end
        end

        % sparse: returns sparse matrix on GPU (not efficient, mainly for debugging)
        function A_sp = sparse(A)
            [m n] = size(A);
            [i j v] = find(A);
            A_sp = sparse(i,j,double(v),m,n);
        end
        
        % full: returns full matrix on GPU (not efficient, mainly for debugging)
        function A_f = full(A)
            A_sp = sparse(A);
            A_f = cast(full(A_sp),classUnderlying(A));
        end
        
        % subsref/assgn not implemented - hard
        function A = subsasgn(A,s,b)
            error('Not implemented.');
        end
        function b = subsref(A,s)
            error('Not implemented.');
        end
        
    end
    
end
