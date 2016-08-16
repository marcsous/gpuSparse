classdef gpuSparse
    %%
    % Sparse GPU array class (mex wrappers to cuSPARSE).
    % Tested on MATLAB R2016a and CUDA 7.5 only.
    % Arguments are the same as matlab's sparse function.
    % Works fastest when row/col are already sorted.
    %
    % Usage: A = gpuSparse(row,col,val,nrows,ncols,nzmax)
    %
    %%
    properties (SetAccess=private, SetObservable=true)
        
        nrows @ int32 scalar = 0; % number of rows
        ncols @ int32 scalar = 0; % number of columns
        
        row @ gpuArray = zeros(0,1,'int32','gpuArray'); % row index (CSR format)
        col @ gpuArray = zeros(0,1,'int32','gpuArray'); % column index
        val @ gpuArray = zeros(0,1,'single','gpuArray'); % nonzero values
        
        % flag to hold the transpose state of the matrix
        %  0 = CUSPARSE_OPERATION_NON_TRANSPOSE
        %  1 = CUSPARSE_OPERATION_TRANSPOSE
        %  2 = CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE
        transp @ int32 scalar = 0;
        
    end
    
    % TO DO (with CUDA > 7.5 and Matlab > R2016a)
    % 1) Revisit sort_row_only
    % 2) Revisit use_gpu_csr2csc
    % 3) Mixed real/complex operations
    
    %%
    methods
        
        % constructor: accepts 0, 1, 2, 3, 5 or 6 arguments
        function A = gpuSparse(row,col,val,nrows,ncols,nzmax)
            
            % return default object, necessary for constructor
            if nargin==0
                return;
            end
            
            % if arg is a gpuSparse, just return it
            if nargin==1 && isa(row,class(A))
                A = row;
                return;
            end
            
            % catch oddball first argument ("row" is the first argument)
            if ~isnumeric(row)
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
            
            % catch illegal no. arguments
            if nargin==4 || nargin>6
                error('Wrong number of arguments.');
            end
            
            % validate arguments - slow to check if integer but adds flexibility
            if numel(row)~=numel(val); error('row and val size mismatch.'); end
            if numel(col)~=numel(val); error('col and val size mismatch.'); end
            
            validateattributes(val,{'numeric','gpuArray'},{},'','val');
            validateattributes(row,{'numeric','gpuArray'},{'integer'},'','row');
            validateattributes(col,{'numeric','gpuArray'},{'integer'},'','col');
            
            % put in required format
            val = single(val(:));
            col = int32(col(:));
            row = int32(row(:)); % still in COO format
            
            % row and probably col should be sorted for COO to CSR conversion
            % 1) sorted row is definitely needed
            % 2) within each row, col is 'presumed to be ascending' (https://people.sc.fsu.edu/~jburkardt/data/cr/cr.html)
            % 3) everything seems to work without the col sort
            % 4) issorted/sortrows with 2D array is not supported on gpu (R2016a)
            sort_row_only = false;
            %
            if sort_row_only
                if ~issorted(row)
                    [row k] = sort(row);
                    col = col(k);
                    val = val(k);
                end
            else
                rowcol = gather([row col]); % need to be on CPU for 2D sort
                if ~issorted(rowcol,'rows')
                    [rowcol k] = sortrows(rowcol);
                    row(:) = rowcol(:,1);
                    col(:) = rowcol(:,2);
                    val = val(k);
                end
            end
            
            % check lower bounds
            if numel(row) > 0
                if row(1)<1; error('All row indices must be greater than zero.'); end
                if min(col)<1; error('All col indices must be greater than zero.'); end
                A.nrows = gather(row(end));
                A.ncols = gather(max(col));
            end
            
            % check and apply user-supplied sizes
            if exist('nrows','var')
                A.nrows = gather(nrows);
            end
            if exist('ncols','var')
                A.ncols = gather(ncols);
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
            if ~exist('nzmax','var')
                nzmax = numel(val);
            else
                nzmax = gather(nzmax);
                validateattributes(nzmax,{'numeric'},{'scalar','integer'},'','nzmax');
                nzmax = max(double(nzmax),numel(val)); % double to prevent overflow
            end
            RequiredMemory = 4*double(A.nrows+1)/1E9;
            if ~isa(val,'gpuArray'); RequiredMemory = RequiredMemory+4*nzmax/1E9; end
            if ~isa(col,'gpuArray'); RequiredMemory = RequiredMemory+4*nzmax/1E9; end
            if RequiredMemory > AvailableMemory
                error('Not enough memory (%.1fGb required, %.1fGb available).',RequiredMemory,AvailableMemory);
            end
            
            % convert row from COO to CSR format
            A.row = coo2csr(row,A.nrows);
            A.col = gpuArray(col);
            A.val = gpuArray(val);

        end
        
        % enforce class properties
        function A = set.nrows(A,nrows)
            validateattributes(nrows,{'numeric'},{'scalar','integer','>=',A.nrows},'','nrows');
            A.nrows = nrows;
        end
        function A = set.ncols(A,ncols)
            validateattributes(ncols,{'numeric'},{'scalar','integer','>=',A.ncols},'','ncols');
            A.ncols = ncols;
        end
        function A = set.row(A,row)
            if ~iscolumn(row) || ~existsOnGPU(row) || ~isequal(classUnderlying(row),'int32')
                error('Property row must be a int32 column gpuArray.')
            end
            A.row = row;
        end
        function A = set.col(A,col)
            if ~iscolumn(col) || ~existsOnGPU(col) || ~isequal(classUnderlying(col),'int32')
                error('Property col must be an int32 column gpuArray.')
            end
            A.col = col;
        end
        function A = set.val(A,val)
            if ~iscolumn(val) || ~existsOnGPU(val) || ~isequal(classUnderlying(val),'single')
                error('Property val must be a single column gpuArray.')
            end
            A.val = val;
        end
        function A = set.transp(A,transp)
            if transp~=0 && transp~=1 && transp~=2
                error('Property transp must be a 0, 1 or 2.')
            end
            A.transp = transp;
        end

        % validation - helpful for testing
        function validate(A)
            
            message = 'Validation failure.';
            
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
            if ~iscolumn(A.row); error(message); end
            if ~iscolumn(A.col); error(message); end
            if ~iscolumn(A.val); error(message); end
            if numel(A.col) ~= numel(A.val); error(message); end
            if numel(A.row) ~= A.nrows+1; error(message); end
            if A.transp~=0 && A.transp~=1 && A.transp~=2; error(message); end
            if numel(A.row) > 0
                if A.row(1) ~= 1; error(message); end
                if A.row(end) ~= numel(A.val)+1; error(message); end
            end
            
            % slow checks
            if numel(A.row) > 0
                if min(A.col) < 1; error(message); end
                if max(A.col) > A.ncols; error(message); end
                rowcol = gather([csr2coo(A.row,A.nrows) A.col]);
                if ~issorted(rowcol,'rows'); error(message); end
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
            if ~isreal(A) || ~isreal(B)
                error('Complex addition not supported at the moment.')
            end
            if A.transp ~= B.transp
                error('Matrix lazy-transpose not fully supported. Use full_transpose.')
            end
            validateattributes(a,{'numeric'},{'real','scalar','finite'},'','a');
            validateattributes(b,{'numeric'},{'real','scalar','finite'},'','b');
            [m n] = size(A);
            if A.transp
                [n m] = size(A);
            else
                [m n] = size(A);
            end
            C = gpuSparse(m,n);
            C.transp = A.transp;
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
                error('Argument x must be numeric (%s not supported).',class(x))
            elseif isscalar(x)
                y = A;
                y.val = y.val * x;
            elseif isvector(x)
                if isreal(A) == isreal(x)
                    y = csrmv(A.row,A.col,A.val,A.nrows,A.ncols,A.transp,x);
                elseif isreal(A)
                    y = complex(csrmv(A.row,A.col,A.val,A.nrows,A.ncols,A.transp,real(x)), ...
                                csrmv(A.row,A.col,A.val,A.nrows,A.ncols,A.transp,imag(x)));
                elseif isreal(x)
                    y = csrmv(A.row,A.col,A.val,A.nrows,A.ncols,A.transp,complex(x));
                else
                    error('Should never get here.')
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
            [m n] = size(A);
            A_t = gpuSparse([],[],[],n,m,nnz(A));
            
            % cuSPARSE version uses excessive memory, prefer CPU version
            use_gpu_csr2csc = false;
            
            if use_gpu_csr2csc
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
        
        % full ctranspose: A'
        function A_t = full_ctranspose(A)
            A_t = full_transpose(A);
            if ~isreal(A_t); A_t.val = conj(A_t.val); end
        end
        
        % lazy transpose (flag): A.'
        function A_t = transpose(A)
            A_t = A; % lazy copy
            switch A.transp
                case 0; A_t.transp = 1;
                case 1; A_t.transp = 0;
                case 2; A_t.transp = 0; A_t.val = conj(A_t.val);
            end
        end
        
        % lazy transpose (flag): A'
        function A_t = ctranspose(A)
            A_t = A; % lazy copy
            switch A_t.transp
                case 0; if isreal(A_t); A_t.transp = 1; else A_t.transp = 2; end
                case 1; A_t.transp = 0; if ~isreal(A_t); A_t.val = conj(A_t.val); end
                case 2; A_t.transp = 0;
            end
        end
        
        % overload nnz
        function retval = nnz(A)
            retval = nnz(A.val);
        end
        
        % overload numel
        function retval = numel(A)
            retval = prod(size(A));
        end
        
        % overload length
        function retval = length(A)
            retval = max(size(A));
        end
        
        % overload nonzeros
        function val = nonzeros(A)
            if A.transp==0
                [~,~,val] = find(A);
                val = gpuArray(val);
            else
                val = nonzeros(A.val);
                if A.transp==2
                    val = conj(A.val);
                end
            end
        end
        
        % overload nzmax
        function retval = nzmax(A)
            retval = numel(A.val);
        end
        
        % overload sum: only A and dim args are supported
        function retval = sum(A,dim,varargin)
            if nargin==1; dim = 1; end
            validateattributes(dim,{'numeric'},{'integer','nonzero'},'','dim')
            if dim==1; retval = (A' * ones(size(A,1),1,'single','gpuArray'))'; end
            if dim==2; retval = A * ones(size(A,2),1,'single','gpuArray'); end
            if dim>2; retval = A; end
        end
        
        % remove zeros from sparse matrix
        function A = remove_zeros(A)
            nonzeros = (A.val ~= 0);
            if ~all(nonzeros)
                A.row = csr2coo(A.row,A.nrows);
                A.row = A.row(nonzeros);
                A.row = coo2csr(A.row,A.nrows);
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
            %B = remove_zeros(B);
        end
        
        % overload imag
        function B = imag(A);
            B = A;
            B.val = imag(A.val);
            %B = remove_zeros(B);
        end
        
        % overload abs
        function B = abs(A);
            B = A;
            B.val = abs(A.val);
        end
        
        % overload angle
        function B = angle(A);
            B = A;
            B.val = angle(A.val);
        end
        
        % overload conj
        function B = conj(A);
            B = A;
            B.val = conj(A.val);
        end
        
        % overload sign
        function B = sign(A);
            B = A;
            B.val = sign(A.val);
        end
        
        % overload complex
        function B = complex(A);
            B = A;
            B.val = complex(A.val);
        end

        % overload norm
        function retval = norm(A,p);
            if nargin<2; p = 2; end
            if isequal(p,1)
                retval = max(sum(abs(A),1));
            elseif isequal(p,2)
                error('gpuSparse norm(S,2) is not available.');
            elseif isequal(p,Inf)
                retval = max(sum(abs(A),2));
            elseif isequal(p,'fro');
                retval = sqrt(real(dot(A.val,A.val)));
            else
                error('The only matrix norms available are 1, 2, inf, and ''fro''.');
            end
        end
        
        % overload max: support for max(A,[],2) only
        function retval = max(A,Y,DIM);
            if nargin ~= 3 || ~isempty(Y) || ~isequal(DIM,2)
                error('Only 3 argument form supported: max(A,[],2).');
            end
            if A.transp
                error('Transpose max not supported - try full_transpose(A).')
            end
            
            % do it on CPU to avoid transfer overhead
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
        
        % overload mean
        function retval = mean(A,DIM,varargin)
            if nargin < 2; DIM = 1; end
            if ~isequal(DIM,1) && ~isequal(DIM,2); error('Dimension value not supported.'); end
            retval = sum(A,DIM) / size(A,DIM);
        end

        % overload find: not efficient, for debugging
        function varargout = find(A)
            if nargin>1; error('only 1 input argument supported'); end
            if nargout>3; error('too many ouput arguments'); end
            
            % COO format on the CPU
            i = gather(csr2coo(A.row,A.nrows));
            j = gather(A.col);
            v = gather(A.val);
            
            % remove explicit zeros
            nonzeros = (v ~= 0);
            if ~all(nonzeros)
                i = i(nonzeros);
                j = j(nonzeros);
                v = v(nonzeros);
            end
            
            % MATLAB style, double precision, sorted columns
            if A.transp
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
                if A.transp==0; varargout{3} = v(k); end
                if A.transp==1; varargout{3} = v; end
                if A.transp==2; varargout{3} = conj(v); end
            end
        end
        
        % overload sparse (returns sparse matrix on CPU): not efficient, for debugging
        function A_sp = sparse(A)
            [m n] = size(A);
            [i j v] = find(A);
            A_sp = sparse(i,j,double(v),m,n);
        end
        
        % overload full (returns full matrix on CPU): not efficient, for debugging
        function A_f = full(A)
            A_sp = sparse(A);
            A_f = cast(full(A_sp),classUnderlying(A));
        end
        
    end
    
end
