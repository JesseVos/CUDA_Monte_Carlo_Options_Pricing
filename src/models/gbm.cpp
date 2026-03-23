// GBM is a header-only template class.
// This translation unit exists to ensure the header compiles correctly
// and to provide an explicit instantiation for the common case (double).

#include "models/gbm.h"

// Explicit template instantiation for double precision.
template class GBM<double>;

// Explicit template instantiation for single precision (for GPU use later).
template class GBM<float>;
