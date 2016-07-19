classdef gpuSparse
%
% Sparse GPU array class (mex wrappers to cuSPARSE).
% Arguments are the same as matlab's sparse function.
%
% Usage: A = gpuSparse(row,col,val,nrows,ncols,nzmax)
%
    %% properties

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
    
    %% methods
    
    methods
        
        % constructor: accepts 0, 2, 3, 5 or 6 arguments
        function A = gpuSparse(row,col,val,nrows,ncols,nzmax)

            % check arguments
            if nargin==0; return; end % necessary for constructor
            if nargin==4 || nargin>6
                error('Wrong number of arguments.');
            end

            % catch oddball arguments
            if ~isnumeric(row)
                if isa(row, class(A)); A = row; return; end
                error('Argument type %s not supported.',class(row));
            end

            % convert scalar, vector or matrix to sparse format
            if nargin==1
                [nrows ncols] = size(row);
                [row col val] = find(row); % not efficient: row unsorted
            end

            % create empty m x n matrix
            if nargin==2
                nrows = row; ncols = col;
                row = []; col = []; val = [];
            end

            % check sizes
            if numel(row)~=numel(val); error('row vector is wrong size.'); end
            if numel(col)~=numel(val); error('col vector is wrong size.'); end

            % convert to required format for cuSPARSE
            A.val = gpuArray(single(val(:)));
            A.col = gpuArray(int32(col(:)));
            A.row = gpuArray(int32(row(:))); % COO format

            % rows must be sorted for COO to CSR conversion
            if ~issorted(A.row)
                [A.row k] = sort(A.row);
                A.col = A.col(k);
                A.val = A.val(k);
            end
            
            % check bounds
            if numel(A.row) > 0
                if A.row(1)<1; error('row indices must be >0.'); end
                A.nrows = gather(A.row(end));
                if min(A.col)<1; error('col indices must be >0.'); end
                A.ncols = gather(max(A.col));
            end

            % check and apply user-supplied sizes
            if exist('nrows','var') && exist('ncols','var')
                if nrows>=A.nrows
                    A.nrows = int32(nrows);
                else
                    error('Maximum row index exceeds nrows.');
                end
                if ncols>=A.ncols
                    A.ncols = int32(ncols);
                else
                    error('Maximum col index exceeds ncols.');
                end
            end

            % don't use nzmax except to check if size is feasible
            if nargin > 5
                info = gpuDevice();
                factor = 1; % empirical factor - not tuned yet
                Required = 4 * (2 * nzmax + A.nrows + 1) * factor / 1E9;
                Available = info.AvailableMemory / 1E9;
                
                if Required > Available
                    error('Memory required too high (%.1fGb versus %.1fGb)',Required,Available);
                end
            end

            % finally convert from COO to CSR format
            A.row = coo2csr(A.row,A.nrows);
           
        end
        
        %% local functions
        
        % validation - reduces performance but is helpful for testing
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
                if A.row(1) < 1; error(message); end
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
                error('no method for adding %s and %s.',class(A),class(B))
            end
            if ~isequal(size(A),size(B))
                error('matrices must have same size.')
            end
            validateattributes(a, {'numeric'}, {'real', 'finite'});
            validateattributes(b, {'numeric'}, {'real', 'finite'});
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
                if ~isreal(A)
                    error('Complex A not supported at the moment.')
                end
                if isreal(x)
                    y = csrmv(A.row,A.col,A.val,A.nrows,A.ncols,A.transp,x);
                else
                    y = complex(csrmv(A.row,A.col,A.val,A.nrows,A.ncols,A.transp,real(x)), ...
                                csrmv(A.row,A.col,A.val,A.nrows,A.ncols,A.transp,imag(x)));
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
            A_t = gpuSparse(n,m);
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
            switch A2.transp
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
        
        % overload classUnderlying
        function str = classUnderlying(A)
            str = classUnderlying(A.val);
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
