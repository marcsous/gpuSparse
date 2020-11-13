// 
// Mex wrapper to CUSPARSE matrix-vector multiply (csrmv).
//
// Inspired by cusparse samples (conugateGradient) and Matlab gcsparse.
//  http://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-csrmv
//  http://www.mathworks.com/matlabcentral/fileexchange/44423-gpu-sparse--accumarray--non-uniform-grid
//

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>

/* Using updated (v2) interfaces to cublas */
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

#if CUDART_VERSION >= 11000
#include "wrappers_to_cuda_11.h"
#endif
        
// MATLAB related
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "mxShowCriticalErrorMessage.c"

// Input Arguments
#define	ROW_CSR prhs[0] // this in CSR format (returned from coo2csr.cu)
#define	COL     prhs[1]
#define	VAL     prhs[2]
#define	NROWS   prhs[3]
#define	NCOLS   prhs[4]
#define	TRANS   prhs[5]
#define	X       prhs[6]

// Output Arguments
#define	Y	plhs[0]

void mexFunction(int nlhs, mxArray * plhs[], int nrhs, const mxArray * prhs[])
{
    // Checks
    if (nlhs > 1) mxShowCriticalErrorMessage("wrong number of output arguments",nlhs);
    if (nrhs != 7) mxShowCriticalErrorMessage("wrong number of input arguments",nrhs);

    if(!mxIsGPUArray(ROW_CSR)) mxShowCriticalErrorMessage("ROW_CSR argument is not on GPU");
    if(!mxIsGPUArray(COL)) mxShowCriticalErrorMessage("COL argument is not on GPU");
    if(!mxIsGPUArray(VAL)) mxShowCriticalErrorMessage("VAL argument is not on GPU");
    if(!mxIsGPUArray(X)) mxShowCriticalErrorMessage("X argument is not on GPU");

    if (!mxIsScalar(NROWS)) mxShowCriticalErrorMessage("NROWS argument must be a scalar");
    if (!mxIsScalar(NCOLS)) mxShowCriticalErrorMessage("NCOLS argument must be a scalar");
    if (!mxIsScalar(TRANS)) mxShowCriticalErrorMessage("TRANS argument must be a scalar");

    // Initialize the MathWorks GPU API
    mxInitGPU();

    // Create Matlab pointers on the GPU
    mxGPUArray const *row_csr = mxGPUCreateFromMxArray(ROW_CSR);
    mxGPUArray const *col = mxGPUCreateFromMxArray(COL);
    mxGPUArray const *val = mxGPUCreateFromMxArray(VAL);
    mxGPUArray const *x = mxGPUCreateFromMxArray(X);

    // Check sizes - note rows are in CSR (compressed row) format
    mwSize nnz = mxGPUGetNumberOfElements(val);
    mwSize nrows = mxGetScalar(NROWS);
    mwSize ncols = mxGetScalar(NCOLS);

    mwSize *xdims = (mwSize*)mxGPUGetDimensions(x); // xdims always has >= 2 elements
    if (mxGPUGetNumberOfDimensions(x) > 2) mxShowCriticalErrorMessage("X argument has too many dimensions",mxGPUGetNumberOfDimensions(x));
    if (xdims[1] != 1) mxShowCriticalErrorMessage("X argument is not a column vector");

    int nx = xdims[0];

    if (mxGPUGetNumberOfElements(row_csr) != nrows+1) mxShowCriticalErrorMessage("ROW_CSR argument wrong size",mxGPUGetNumberOfElements(row_csr));
    if (mxGPUGetNumberOfElements(col) != nnz) mxShowCriticalErrorMessage("COL argument wrong size",mxGPUGetNumberOfElements(col));

    cusparseOperation_t trans = (cusparseOperation_t)mxGetScalar(TRANS);
    if (trans == CUSPARSE_OPERATION_NON_TRANSPOSE)
    {
	if (nx != ncols) mxShowCriticalErrorMessage("X argument wrong size for multiply",nx);
    }
    else
    {
	if (nx != nrows) mxShowCriticalErrorMessage("X argument wrong size for transpose multiply",nx);
    }

    // Check types
    if (mxGPUGetClassID(row_csr) != mxINT32_CLASS) mxShowCriticalErrorMessage("ROW_CSR argument is not int32");
    if (mxGPUGetClassID(col) != mxINT32_CLASS) mxShowCriticalErrorMessage("COL argument is not int32");
    if (mxGPUGetClassID(val) != mxSINGLE_CLASS) mxShowCriticalErrorMessage("VAL argument is not single");
    if (mxGPUGetClassID(x) != mxSINGLE_CLASS) mxShowCriticalErrorMessage("X argument is not single");

    // Check real/complex - mixed is not supported except special case (real A / complex x)
    mxComplexity ccx = mxGPUGetComplexity(x);
    mxComplexity ccv = mxGPUGetComplexity(val);
    mxComplexity ccy = (ccx==mxCOMPLEX || ccv==mxCOMPLEX) ? mxCOMPLEX : mxREAL;
    if(ccx==mxREAL && ccv==mxCOMPLEX) mxShowCriticalErrorMessage("Complex matrix and real vector not supported");
            
    // Create space for output vector
    const mwSize ndim = 1;
    mwSize dims[ndim] = {trans == CUSPARSE_OPERATION_NON_TRANSPOSE ? nrows : ncols};
    mxClassID cid = mxGPUGetClassID(x);
    mxGPUArray *y;

    // Get handle to the CUBLAS context
    cublasHandle_t cublasHandle = 0;
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&cublasHandle);
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) mxShowCriticalErrorMessage(cublasStatus);

    // Get handle to the CUSPARSE context
    cusparseHandle_t cusparseHandle = 0;
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&cusparseHandle);
    if (cusparseStatus != CUSPARSE_STATUS_SUCCESS) mxShowCriticalErrorMessage(cusparseStatus);
    cusparseMatDescr_t descr = 0;
    cusparseStatus = cusparseCreateMatDescr(&descr);
    if (cusparseStatus != CUSPARSE_STATUS_SUCCESS) mxShowCriticalErrorMessage(cusparseStatus);
    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ONE);

    // Convert from matlab pointers to native pointers
    const int* const d_row_csr = (int*)mxGPUGetDataReadOnly(row_csr);
    const int* const d_col = (int*)mxGPUGetDataReadOnly(col);

    // Now we can access the arrays, we can do some checks
    int base;
    cudaMemcpy(&base, d_row_csr, sizeof(int), cudaMemcpyDeviceToHost);
    if (base != CUSPARSE_INDEX_BASE_ONE) mxShowCriticalErrorMessage("ROW_CSR not using 1-based indexing");

    int nnz_check;
    cudaMemcpy(&nnz_check, d_row_csr+nrows, sizeof(int), cudaMemcpyDeviceToHost);
    nnz_check -= CUSPARSE_INDEX_BASE_ONE;
    if (nnz_check != nnz) mxShowCriticalErrorMessage("ROW_CSR argument last element != nnz",nnz_check);

    // Call cusparse multiply function in (S)ingle precision
    if (ccv==mxREAL && ccx==mxREAL)
    {
        const float alpha = 1.0; 
        const float beta = 0.0;
        y = mxGPUCreateGPUArray(ndim, dims, cid, ccy, MX_GPU_INITIALIZE_VALUES);
        if (y==NULL) mxShowCriticalErrorMessage("mxGPUCreateGPUArray failed.");
        float* d_y = (float*)mxGPUGetData(y);
    	const float* const d_val = (float*)mxGPUGetDataReadOnly(val);
    	const float* const d_x = (float*)mxGPUGetDataReadOnly(x);
#if CUDART_VERSION >= 11000
        cusparseStatus = cusparseXcsrmv_wrapper(cusparseHandle, trans, nrows, ncols, nnz, &alpha, descr, d_val, d_row_csr, d_col, d_x, &beta, d_y);
#else
        cusparseStatus = cusparseScsrmv(cusparseHandle, trans, nrows, ncols, nnz, &alpha, descr, d_val, d_row_csr, d_col, d_x, &beta, d_y);
#endif
    }
    else if (ccv==mxREAL && ccx==mxCOMPLEX)
    {
        const float alpha = 1.0; 
        const float beta = 0.0;
        const float* const d_val = (float*)mxGPUGetDataReadOnly(val);

        mxGPUArray* y_real = mxGPUCreateGPUArray(ndim, dims, cid, mxREAL, MX_GPU_INITIALIZE_VALUES);
        mxGPUArray* y_imag = mxGPUCreateGPUArray(ndim, dims, cid, mxREAL, MX_GPU_INITIALIZE_VALUES);
        if(!y_real || !y_imag) mxShowCriticalErrorMessage("mxGPUCreateGPUArray failed.");
        float* d_y_real = (float*)mxGPUGetDataReadOnly(y_real);
        float* d_y_imag = (float*)mxGPUGetDataReadOnly(y_imag);

        for(int i = 0; i<2; i++)
        {
            mxGPUArray const *x_tmp;
            if(i==0) x_tmp = mxGPUCopyReal(x);
            if(i==1) x_tmp = mxGPUCopyImag(x);
            const float* const d_x = (float*)mxGPUGetDataReadOnly(x_tmp);
#if CUDART_VERSION >= 11000	
            if(i==0) cusparseStatus = cusparseXcsrmv_wrapper(cusparseHandle, trans, nrows, ncols, nnz, &alpha, descr, d_val, d_row_csr, d_col, d_x, &beta, d_y_real);
            if(i==1) cusparseStatus = cusparseXcsrmv_wrapper(cusparseHandle, trans, nrows, ncols, nnz, &alpha, descr, d_val, d_row_csr, d_col, d_x, &beta, d_y_imag);
#else
            if(i==0) cusparseStatus = cusparseScsrmv(cusparseHandle, trans, nrows, ncols, nnz, &alpha, descr, d_val, d_row_csr, d_col, d_x, &beta, d_y_real);
            if(i==1) cusparseStatus = cusparseScsrmv(cusparseHandle, trans, nrows, ncols, nnz, &alpha, descr, d_val, d_row_csr, d_col, d_x, &beta, d_y_imag);
#endif
            mxGPUDestroyGPUArray(x_tmp);
            if(cusparseStatus != CUSPARSE_STATUS_SUCCESS) mxShowCriticalErrorMessage("csrmv failed.");
        }
        y = mxGPUCreateComplexGPUArray(y_real,y_imag);
        if (y==NULL) mxShowCriticalErrorMessage("mxGPUCreateComplexGPUArray failed.");
        mxGPUDestroyGPUArray(y_real);
        mxGPUDestroyGPUArray(y_imag);
    }
    else
    {
        const cuFloatComplex alpha = make_cuFloatComplex(1.0, 0.0);
        const cuFloatComplex beta = make_cuFloatComplex(0.0, 0.0);
        y = mxGPUCreateGPUArray(ndim, dims, cid, ccy, MX_GPU_INITIALIZE_VALUES);
        if (y==NULL) mxShowCriticalErrorMessage("mxGPUCreateGPUArray failed.");
        cuFloatComplex* d_y = (cuFloatComplex*)mxGPUGetData(y);
    	const cuFloatComplex* const d_val = (cuFloatComplex*)mxGPUGetDataReadOnly(val);
    	const cuFloatComplex* const d_x = (cuFloatComplex*)mxGPUGetDataReadOnly(x);
#if CUDART_VERSION >= 11000	
        cusparseStatus = cusparseXcsrmv_wrapper(cusparseHandle, trans, nrows, ncols, nnz, &alpha, descr, d_val, d_row_csr, d_col, d_x, &beta, d_y);
#else
        cusparseStatus = cusparseCcsrmv(cusparseHandle, trans, nrows, ncols, nnz, &alpha, descr, d_val, d_row_csr, d_col, d_x, &beta, d_y);
#endif
    }
    
	// Return result
    if (cusparseStatus == CUSPARSE_STATUS_SUCCESS)
    {
    	Y = mxGPUCreateMxArrayOnGPU(y);
    }
    else
    {
        mxShowCriticalErrorMessage("unknown failure",cusparseStatus);
    }

    // Clean up
    cusparseDestroyMatDescr(descr);
    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);
    mxGPUDestroyGPUArray(row_csr);
    mxGPUDestroyGPUArray(col);
    mxGPUDestroyGPUArray(val);
    mxGPUDestroyGPUArray(x);
    mxGPUDestroyGPUArray(y);
    mxFree(xdims);

    return;
}

