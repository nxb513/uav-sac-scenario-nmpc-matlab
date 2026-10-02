/* d1_set_ftz(mode) -- DIAGNOSTIC MEX (not part of the method).
 * Reads / changes the floating-point control register MXCSR of the calling thread (the
 * MATLAB interpreter thread, which also runs the acados MEX):
 *   mode =  1 : set FTZ (bit 15, flush-to-zero) and DAZ (bit 6, denormals-are-zero)
 *   mode =  0 : clear FTZ and DAZ
 *   mode = -1 : read only (default)
 * Returns the MXCSR value BEFORE the change. Build: mex d1_set_ftz.c
 */
#include "mex.h"
#include <xmmintrin.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    unsigned int csr = _mm_getcsr();
    int mode = (nrhs > 0) ? (int) mxGetScalar(prhs[0]) : -1;
    if (mode == 1) {
        _mm_setcsr(csr | 0x8040u);
    } else if (mode == 0) {
        _mm_setcsr(csr & ~0x8040u);
    }
    plhs[0] = mxCreateDoubleScalar((double) csr);
}
