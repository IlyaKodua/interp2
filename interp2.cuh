#ifndef INT2_INTERP2_CUH
#define INT2_INTERP2_CUH

#include <cuda_runtime_api.h>
#include <cuda.h>
#include "device_launch_parameters.h"
#include <cuComplex.h>

typedef float xt;
typedef float yt;
typedef cuComplex zt;



template<typename T>
__device__ int search_index(const T *data, const int size_data,
                            const T dataq)
{
    int idx(-1);
    for(int i = 0; i < size_data; i++)
    {
        if(data[i] > dataq)
        {
            break;
        }
        idx++;
    }
    return idx;
}


__global__ void interp2(const xt* X,  const int n_X,
                        const yt *Y,  const int n_Y,
                        const zt *Z,
                        const xt *Xq,
                        const yt *Yq,
                        zt *Zq, const int points_count)
{

    zt z1;
    zt z2;

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int numThreads = blockDim.x * gridDim.x;

    for (int ii = idx; ii < points_count; ii += numThreads)
    {
        if( Xq[ii] >= X[0] && Xq[ii] <= X[n_X - 1] &&
                Yq[ii] >= Y[0] && Yq[ii] <= Y[n_Y - 1])
        {

            int ix = search_index(X, n_X, Xq[ii]);
            int iy = search_index(Y, n_Y, Yq[ii]);

            yt d_y = Y[iy+1] - Y[iy];
            xt d_x = X[ix+1] - X[ix];

            int id = iy + n_Y*ix;

            z1 = (Z[id + n_Y] - Z[id]) * Xq[ii] / d_x;

            z1 += (Z[id] * X[ix + 1]- Z[id + n_Y]* X[ix]) / d_x;

            z2 = (Z[id + n_Y + 1] - Z[id + 1]) * Xq[ii] / d_x;

            z2 += (Z[id + 1] * X[ix + 1]- Z[id + n_Y + 1]* X[ix]) / d_x;

            Zq[ii] = (z2 - z1) * Yq[ii] / d_y;

            Zq[ii] += (z1 * Y[iy+1]- z2 * Y[iy]) / d_y;
        }
        else
        {
            Zq[ii] = {0,0};
        }
        }
        else
        {
            Zq[ii] = {};
        }
    }
}

#endif //INT2_INTERP2_CUH
