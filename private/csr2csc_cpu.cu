// 
// Mex wrapper to C code format converter (csr2csc) to do transpose.
//
// Inspired by:
//    http://www.dgate.org/~brg/files/dis/smvm/frontend/matrix_io.c
//

void csr2csc(const int nrows, const int ncols, const float *val, const int *col, const int *row_csr,
             float *val_csc, int *row, int *col_csc)
{
  int i, j, k, l;

  // Base index (0 or 1) and no. nonzeros
  const int base = row_csr[0];
  const int nnz = row_csr[nrows]-base;

  for (i=0; i<=ncols; i++) col_csc[i] = 0;

  // Determine column lengths
  for (i=0; i<nnz; i++) col_csc[col[i]+1-base]++;

  for (i=0; i<ncols; i++) col_csc[i+1] += col_csc[i];

  // Fill in output array
  for (i=0; i<nrows; i++)
  {
    for (j=row_csr[i]-base; j<row_csr[i+1]-base; j++)
    {
      k = col[j]-base;
      l = col_csc[k]++;
      row[l] = i+base;
      val_csc[l] = val[j];
    }
  }

  // Shift col_csc by 1
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
    if (nlhs > 3) mxShowCriticalErrorMessage("wrong number of output arguments");
    if (nrhs != 5) mxShowCriticalErrorMessage("wrong number of input arguments");

    // Checks - note rows must be in CSR format
    int nnz = mxGetNumberOfElements(VAL);
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

    VAL_CSC = mxCreateUninitNumericArray(ndim, dims, mxSINGLE_CLASS, mxREAL);
    if (VAL_CSC==NULL) mxShowCriticalErrorMessage("mxGPUCreateGPUArray failed");
 
    // Pointers to the raw data
    int *row_csr = (int *)mxGetData(ROW_CSR);
    int *col = (int *)mxGetData(COL);
    float *val = (float *)mxGetData(VAL);

    int *row = (int *)mxGetData(ROW);
    int *col_csc = (int *)mxGetData(COL_CSC);
    float *val_csc = (float *)mxGetData(VAL_CSC);

    // Now we can access the arrays, we can do some checks
    int base = row_csr[0];
    if (base != 1) mxShowCriticalErrorMessage("ROW_CSR not using 1-based indexing");

    int nnz_check = row_csr[nrows];
    nnz_check -= 1; // MATLAB unit offset
    if (nnz_check != nnz) mxShowCriticalErrorMessage("ROW_CSR argument last element != nnz");

    // Convert from CSR to CSC
    csr2csc(nrows, ncols, val, col, row_csr, val_csc, row, col_csc);

    return;
}

