// 
// Mex wrapper to C code format converter (csr2csc) to do transpose.
//
// Inspired by:
//    http://www.dgate.org/~brg/files/dis/smvm/frontend/matrix_io.c
//

void csr2csc(const int nrows, const int ncols, const int *row_csr, const int *col, const float *val_real, const float *val_imag,
             int *row, int *col_csc, float *val_csc_real, float *val_csc_imag)
{
  int i, j, k, l;

  // Base index (0 or 1) and number of nonzeros
  const int base = row_csr[0];
  const int nnz = row_csr[nrows]-base;

  // Determine column lengths
  for (i=0; i<=ncols; i++) col_csc[i] = 0;
  for (i=0; i<nnz; i++) col_csc[col[i]+1-base]++;
  for (i=0; i<ncols; i++) col_csc[i+1] += col_csc[i];

  // Fill in output arrays
  for (i=0; i<nrows; i++)
  {
    for (j=row_csr[i]-base; j<row_csr[i+1]-base; j++)
    {
      k = col[j]-base;
      l = col_csc[k]++;
      row[l] = i+base;
      if (val_real) val_csc_real[l] = val_real[j];
      if (val_imag) val_csc_imag[l] = val_imag[j];
    }
  }

  // Shift col_csc by 1 place
  for (i=ncols; i>0; i--) col_csc[i] = col_csc[i-1]+base;

  col_csc[0] = base;
}

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// MATLAB related
#include "mex.h"
#include "mxShowCriticalErrorMessage.c"

// Input Arguments
#define	ROW_CSR prhs[0] // CSR format
#define	COL     prhs[1]
#define	VAL     prhs[2]
#define	NROWS   prhs[3]
#define	NCOLS   prhs[4]

// Output Arguments
#define	ROW     plhs[0]
#define	COL_CSC plhs[1] // CSC format
#define	VAL_CSC plhs[2]

void mexFunction(int nlhs, mxArray * plhs[], int nrhs, const mxArray * prhs[])
{
    // Checks
    if (nlhs > 3) mxShowCriticalErrorMessage("wrong number of output arguments",nlhs);
    if (nrhs != 5) mxShowCriticalErrorMessage("wrong number of input arguments",nrhs);

    // Checks - note rows must be in CSR format
    int nnz = mxGetNumberOfElements(VAL);

    if (!mxIsScalar(NROWS)) mxShowCriticalErrorMessage("NROWS argument must be a scalar");
    if (!mxIsScalar(NCOLS)) mxShowCriticalErrorMessage("NCOLS argument must be a scalar");

    int nrows = (int)mxGetScalar(NROWS);
    int ncols = (int)mxGetScalar(NCOLS);

    if (mxGetClassID(ROW_CSR) != mxINT32_CLASS) mxShowCriticalErrorMessage("ROW_CSR argument is not int32");
    if (mxGetClassID(COL) != mxINT32_CLASS) mxShowCriticalErrorMessage("COL argument is not int32");
    if (mxGetClassID(VAL) != mxSINGLE_CLASS) mxShowCriticalErrorMessage("VAL argument is not single");

    // Create space for output vectors
    const mwSize ndim = 1;
    mwSize dims[ndim];

    dims[0] = ncols+1;
    COL_CSC = mxCreateUninitNumericArray(ndim, dims, mxINT32_CLASS, mxREAL);
    if (COL_CSC==NULL) mxShowCriticalErrorMessage("mxGPUCreateGPUArray failed");

    dims[0] = nnz;
    ROW = mxCreateUninitNumericArray(ndim, dims, mxINT32_CLASS, mxREAL);
    if (ROW==NULL) mxShowCriticalErrorMessage("mxGPUCreateGPUArray failed");

    mxComplexity ccx = mxGetPi(VAL) ? mxCOMPLEX : mxREAL;
    VAL_CSC = mxCreateUninitNumericArray(ndim, dims, mxSINGLE_CLASS, ccx);
    if (VAL_CSC==NULL) mxShowCriticalErrorMessage("mxGPUCreateGPUArray failed");
 
    // Pointers to the raw data
    const int * const row_csr = (int *)mxGetData(ROW_CSR);
    const int * const col = (int *)mxGetData(COL);
    const float * const val_real = (float *)mxGetData(VAL);
    const float * const val_imag = (float *)mxGetImagData(VAL);
    	
    int *row = (int *)mxGetData(ROW);
    int *col_csc = (int *)mxGetData(COL_CSC);
    float *val_csc_real = (float *)mxGetData(VAL_CSC);
    float *val_csc_imag = (float *)mxGetImagData(VAL_CSC);
    
    // Now we can access the arrays, we can do some checks
    const int base = row_csr[0];
    if (base != 1) mxShowCriticalErrorMessage("ROW_CSR not using 1-based indexing");

    int nnz_check = row_csr[nrows];
    nnz_check -= 1;
    if (nnz_check != nnz) mxShowCriticalErrorMessage("ROW_CSR argument last element != nnz",nnz_check);

    // Convert from CSR to CSC
    csr2csc(nrows, ncols, row_csr, col, val_real, val_imag, row, col_csc, val_csc_real, val_csc_imag);

    return;
}

