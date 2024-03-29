classdef gpuSparse
    %%
    % Sparse GPU array class (mex wrappers to cuSPARSE)
    % using int32 indices and single precision values.
    %
    % Usage: A = gpuSparse(row,col,val,nrows,ncols,nzmax)
    %
    % To recompile mex call gpuSparse('recompile')
    %
    % The nzmax argument can be used to check sufficient
    % memory: gpuSparse([],[],[],nrows,ncols,nzmax)
    %
    %%
    properties (SetAccess = private) %immutable)
        
        nrows(1,1) int32 % number of rows
        ncols(1,1) int32 % number of columns

    end
    
    properties (SetAccess = private, Hidden = true)
        
        row(:,1) gpuArray % int32 row index (CSR format)
        col(:,1) gpuArray % int32 column index
        val(:,1) gpuArray % single precision values
        trans(1,1) int32  % lazy transpose flag (passed to cuSPARSE)
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

            % expecting a matrix, return gpuSparse ("row" is the first argument)
            if nargin==1
                if isa(row,'gpuSparse'); A = row; return; end % return unchanged
                if isequal(row,'recompile'); mex_all; return; end % recompile mex
                if ~isnumeric(row) && ~islogical(row); error('Cannot convert ''%s'' to gpuSparse.',class(row)); end
                if ~ismatrix(row); error('Cannot convert ND array to gpuSparse.'); end
                [nrows ncols] = size(row);
                [row col val] = find(row); % if sparse, could grab the CSR vectors directly but needs mex = hassle
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
            validateattributes(val,{'numeric','gpuArray','logical'},{},'','val');

            % check vector lengths
            row = reshape(row,[],1);
            col = reshape(col,[],1);
            val = reshape(val,[],1);            
            if numel(row)~=numel(col)
                error('Vectors must be the same length (row=%i col=%i).',numel(row),numel(col));
            end
            if numel(val)~=numel(row)
                if numel(val)==1
                    val = repmat(val,numel(row),1);
                else
                    error('Vectors must be the same length (row=%i val=%i).',numel(row),numel(val));
                end
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
            if ~exist('nzmax','var')
                nzmax = numel(val);
            else
                nzmax = gather(nzmax);
                validateattributes(nzmax,{'numeric'},{'scalar','integer','>=',numel(val)},'','nzmax');
            end
            RequiredMemory = 4*double(A.nrows+1)/1E9;
            RequiredMemory = RequiredMemory+4*double(nzmax)/1E9;
            RequiredMemory = RequiredMemory+4*double(nzmax)/1E9;
            AvailableMemory = getfield(gpuDevice(),'AvailableMemory') / 1E9;            
            if RequiredMemory > AvailableMemory
                error('Not enough memory (%.1fGb required, %.1fGb available).',RequiredMemory,AvailableMemory);
            end

            % cast to required class
            row = int32(row);
            col = int32(col);
            val = single(val);

            % sort row and col for COO to CSR conversion (MATLAB version)
            %[B I] = sortrows([row col]);
            %A.row = B(:,1);
            %A.col = B(:,2);
            %A.val = val(I);
            %clear B I row col val 
            
            % sort row and col for COO to CSR conversion (CUDA version)
            try
                [A.row A.col A.val] = coosortByRow(row,col,val,A.nrows,A.ncols);
            catch ME
                error('%s Try gpuSparse(''recompile'') to recompile mex.',ME.message);
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
            if isreal(A) && trans==2
                error('Real matrix trans flag must be 0 or 1');
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
            if isreal(A) && A.trans==2; error(message); end

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
        function A = real(A)
            A.val = real(A.val);
            if A.trans==2; A.trans = 1; end
            A = drop_zeros(A);
        end
        
        % imag
        function A = imag(A)
            A.val = imag(A.val);
            if A.trans==2; A.trans = 1; end
            A = drop_zeros(A);
        end
        
        % abs
        function A = abs(A)
            A.val = abs(A.val);
            if A.trans==2; A.trans = 1; end
        end
        
        % angle
        function A = angle(A)
            A.val = angle(A.val);
            if A.trans==2; A.trans = 1; end
            A = drop_zeros(A);
        end
        
        % conj
        function A = conj(A)
            A.val = conj(A.val);
        end
        
        % sign
        function A = sign(A)
            A.val = sign(A.val);
            if A.trans==2; A.trans = 1; end
        end
        
        % complex
        function A = complex(A)
            A.val = complex(A.val);
        end

        % classUnderlying
        function str = classUnderlying(A)
            str = classUnderlying(A.val);
        end
        
        % gt (only support scalar)
        function A = gt(A,tol);
            if ~isscalar(tol)
                error('Non-scalar argument not supported.');
            end
            A.val = cast(A.val > tol,classUnderlying(A));
            if A.trans==2; A.trans = 1; end
            A = drop_zeros(A);
        end
        
        % lt (only support scalar)
        function A = lt(A,tol);
            if ~isscalar(tol)
                error('Non-scalar argument not supported.');
            end
            A.val = cast(A.val < tol,classUnderlying(A));
            if A.trans==2; A.trans = 1; end
            A = drop_zeros(A);
        end   
        
        % eq (only support scalar)
        function A = eq(A,tol);
            if ~isscalar(tol)
                error('Non-scalar argument not supported.');
            end
            A.val = cast(A.val == tol,classUnderlying(A));
            if A.trans==2; A.trans = 1; end
            A = drop_zeros(A);
        end  
        
        % nnz
        function retval = nnz(A)
            retval = nnz(A.val);
        end
        
        % length
        function retval = length(A)
            retval = max(size(A));
        end
        
        % nzmax
        function retval = nzmax(A)
            retval = numel(A.val);
        end
        
        % mean: only A and DIM args are supported
        function retval = mean(A,DIM)
            if nargin==1; DIM = 1; end
            retval = sum(A,DIM) / size(A,DIM);
        end
        
        % nonzeros
        function val = nonzeros(A)
            val = nonzeros(A.val);
            if A.trans==2
                val = conj(val);
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
                switch DIM
                    case 1; retval =(A'* ones(size(A,1),1,'like',A.val))';
                    case 2; retval = A * ones(size(A,2),1,'like',A.val);
                    otherwise; retval = A;
                end
            end
        end
        
        % norm: support same types as sparse
        function retval = norm(A,p);
            if nargin<2; p = 2; end
            if isvector(A)
                retval = norm(A.val,p);
            else
                if isequal(p,2)
                    error('gpuSparse norm(A,2) is not supported.');
                elseif isequal(p,1)
                    retval = max(sum(abs(A),1));
                elseif isequal(p,Inf)
                    retval = max(sum(abs(A),2));
                elseif isequal(p,'fro');
                    retval = norm(A.val);
                else
                    error('The only matrix norms supported are 1, 2, inf, and ''fro''.');
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
                if ~isscalar(DIM) || DIM<=0 || mod(DIM,1)
                    error('Dimension argument must be a positive integer scalar.')
                elseif DIM==1
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
            A = gpuSparse(A);
            B = gpuSparse(B);
            if ~isequal(size(A),size(B))
                error('Matrices must be the same size.')
            end
            if ~isreal(A) || ~isreal(B)
                error('Complex addition not supported at the moment.')
            end
            if A.trans ~= B.trans
                error('Matrix addition with lazy transpose not fully supported.')
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
        
        % mtimes: A*x (or x*A for scalar x)
        function y = mtimes(A,x)
            if isa(x,'gpuSparse') && ~isa(A,'gpuSparse')
                [A x] = deal(x,A);
            end
            if ~isnumeric(x) && islogical(x)
                error('Argument x must be numeric (%s not supported).',class(x))
            elseif isscalar(x) && ~iscolumn(A)
                y = A;
                y.val = y.val * x;
            elseif isvector(x)
                if isreal(A)
                    y = csrmv(A.row,A.col,A.val,A.nrows,A.ncols,A.trans,x);
                else
                    y = csrmv(A.row,A.col,A.val,A.nrows,A.ncols,A.trans,complex(x));
                end
            elseif ismatrix(x)
                if isreal(A)
                    y = csrmm(A.row,A.col,A.val,A.nrows,A.ncols,A.trans,x);
                else
                    y = csrmm(A.row,A.col,A.val,A.nrows,A.ncols,A.trans,complex(x));
                end
            end
        end
        
        % times: A.*x or x.*A (scalar x only)
        function A = times(A,x)
            if isa(x,'gpuSparse') && ~isa(A,'gpuSparse')
                [A x] = deal(x,A);
            end
            if ~isnumeric(x) && ~islogical(x) && ~isempty(x)
                error('Argument x must be numeric (%s not supported).',class(x))
            elseif isscalar(x) && isfinite(x)
                A.val = A.val .* x;
            else
                error('Multiplication only supported for finite scalars.')
            end
        end
        
        % divide: A./x
        function A = rdivide(A,x)
            if isa(x,'gpuSparse')
                error('Division by gpuSparse array not supported.');
            end
            A = times(A,1./x);
        end
        
        % divide: A/x (scalar x only)
        function A = mrdivide(A,x)
            A = A./x;
        end
        
        % power: A.^x
        function A = power(A,x)
            if isa(x,'gpuSparse') || ~isscalar(x)
                error('Power A.^x only supported for gpuSparse A and scalar x.');
            end
            A.val = A.val.^x;
        end
        
        % full transpose: A.'
        function AT = full_transpose(A)
            if A.trans
                AT = A;
                AT.trans = 0;
                if ~isreal(A) && A.trans==2
                    AT.val = conj(AT.val);
                end
            else
                [m n] = size(A);
                AT = gpuSparse([],[],[],n,m,nnz(A));
                
                if nnz(A) % cuSPARSE breaks if nnz==0 so avoid call
                    if 1 % older cuSPARSE used excessive memory - seems OK now
                        [AT.col AT.row AT.val] = csr2csc(A.row,A.col,A.val,m,n);
                    else % cpu version
                        row = gather(A.row);
                        col = gather(A.col);
                        val = gather(A.val);
                        [col row val] = csr2csc_cpu(row,col,val,m,n);
                        AT.col = gpuArray(col);
                        AT.row = gpuArray(row);
                        AT.val = gpuArray(val);
                    end
                end
            end
        end
        
        % full ctranspose: A'
        function AT = full_ctranspose(A)
            if A.trans
                AT = A;
                AT.trans = 0;
            else
                AT = full_transpose(A);
            end
            if ~isreal(A) && A.trans~=2
                AT.val = conj(AT.val);
            end
        end

        % lazy transpose (flag): A.'
        function AT = transpose(A)
            AT = A; % lazy copy
            switch A.trans
                case 0; AT.trans = 1;
                case 1; AT.trans = 0;
                case 2; AT.trans = 0; AT.val = conj(AT.val);
            end
        end
        
        % lazy transpose (flag): A'
        function AT = ctranspose(A)
            AT = A; % lazy copy
            switch A.trans
                case 0; if isreal(A); AT.trans = 1; else; AT.trans = 2; end
                case 1; AT.trans = 0; if ~isreal(A); AT.val = conj(AT.val); end
                case 2; AT.trans = 0;
            end
        end

        % remove zeros from sparse matrix
        function A = drop_zeros(A,tol)
            if nargin<2
                nz = (A.val ~= 0);
            else
                validateattributes(tol,{'numeric'},{'nonnegative','scalar'},'','tol');
                nz = abs(A.val) < tol;
            end
            if any(nz)
                A.row = csr2coo(A.row,A.nrows);
                A.row = A.row(nz);
                A.row = coo2csr(A.row,A.nrows);
                A.col = A.col(nz);
                A.val = A.val(nz);
            end
        end
        
        % sparse: returns sparse matrix on GPU
        function A_sp = sparse(A)
            [m n] = size(A);
            i = csr2coo(A.row,A.nrows);
            j = A.col;
            v = double(A.val);
            switch A.trans
                % int32 indices ok (2020a)
                case 0; A_sp = sparse(i,j,v,m,n);
                case 1; A_sp = sparse(j,i,v,m,n);
                case 2; A_sp = sparse(j,i,conj(v),m,n);
            end
        end

        % gather: returns sparse matrix on CPU - gather(sparse(A)) is faster but memory intensive 
        function A_sp = gather(A)
            [m n] = size(A);
            i = gather(csr2coo(A.row,A.nrows));
            j = gather(A.col);
            v = gather(double(A.val)); % double for sparse
            switch A.trans
                % sparse int32 indices ok (2020a)
                case 0; A_sp = sparse(i,j,v,m,n);
                case 1; A_sp = sparse(j,i,v,m,n);
                case 2; A_sp = sparse(j,i,conj(v),m,n);
            end
        end
        
        % full: returns full matrix on CPU (not efficient, mainly for debugging)
        function A_f = full(A)
            i = gather(csr2coo(A.row,A.nrows));
            j = gather(A.col);
            v = gather(A.val);
            switch A.trans
                % sparse int32 indices ok (2020a)
                case 0; k = sub2ind(size(A),i,j);
                case 1; k = sub2ind(size(A),j,i);
                case 2; k = sub2ind(size(A),j,i); v = conj(v);
            end
            A_f = zeros(size(A),'like',v);
            A_f(k) = v;
        end

        % numel - should it be 1 object or prod(size(A)) elements?
        function retval = numel(A)
            retval = prod(size(A));
        end

        % cat
        function C = cat(dim,A,B)
            switch dim
                case 1; C = vertcat(A,B);
                case 2; C = horzcat(A,B);
                otherwise; error('Concatenation only supported for dim=1 or 2.');
            end
        end

        % vertcat
        function C = vertcat(A,B)
            if ~isa(B,'gpuSparse')
                error('Concatenation only supported for gpuSparse.');
            end
            if A.trans || B.trans
                error('Concatenation not supported with transpose.');
            end
            if size(A,2)~=size(B,2)
                error('Concatenation requires number of cols be equal.');
            end
            C = gpuSparse(size(A,1)+size(B,1),size(A,2));
            C.row = [A.row;B.row(2:end)+numel(A.val)];
            C.col = [A.col;B.col];
            C.val = [A.val;B.val];
        end

        % horzcat - possible to avoid csr2coo calls?
        function C = horzcat(A,B)
            if ~isa(B,'gpuSparse') || A.trans || B.trans
                error('Concatenation only supported for non-tranposed gpuSparse.');
            end
            if A.trans || B.trans
                error('Concatenation not supported with transpose.');
            end
            if size(A,1)~=size(B,1)
                error('Concatenation requires number of rows be equal.');
            end
            i = [csr2coo(A.row,A.nrows);csr2coo(B.row,B.nrows)];
            j = [A.col;B.col+size(A,2)];
            v = [A.val;B.val];
            C = gpuSparse(i,j,v,size(A,1),size(A,2)+size(B,2));
        end

        % Mathworks suggested this to help fix . indexing
        function retval = numArgumentsFromSubscript(A, s, ic)
            retval = builtin('numArgumentsFromSubscript', A, s, ic);
        end

        % the following are hard - don't implement
        function retval = subsref(A,s)
            if isequal(s.type,'.')
                retval = A.(s.subs);
            else
                error('subsref not implemented.');
            end
        end        
        function retval = subsasgn(A,s,b)
            error('subsasgn not implemented.');
        end
        function A = reshape(A,m,n)
            error('reshape not implemented.');
        end
    end
end
