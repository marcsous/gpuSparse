// Beautified mex error message function inspired by the following program.
// http://www.advanpix.com/2016/02/14/short-and-informative-error-messages-from-mex/
//
#define mxShowCriticalErrorMessage(error_message) \
{\
    mxArray *error_args[4];\
    error_args[0] = mxCreateString("%s(%i): %s.");\
    error_args[1] = mxCreateString(basename(__FILE__));\
    error_args[2] = mxCreateDoubleMatrix(1,1,mxREAL);\
    *mxGetPr(error_args[2]) = __LINE__;\
    error_args[3] = mxCreateString(error_message);\
    mexCallMATLAB(0,0,4,error_args,"error");\
}

