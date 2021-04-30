#include <iostream>
#include "interp2.cuh"



int main() {


    int nX = 2;
    int nY = 2;
    int points = 5;

    int sizeX = sizeof(xt) * nX;
    int sizeY = sizeof(yt) * nY;
    int sizeZ = sizeof(zt) * nX * nY;
    int sizeXq= sizeof(xt) * points;
    int sizeYq= sizeof(yt) * points;
    int sizeZq= sizeof(zt) * points;

    xt *X = (xt *) malloc(sizeX);
    yt *Y = (yt *) malloc(sizeY);;
    zt *Z = (zt *) malloc(sizeZ);;
    xt *Xq = (xt *) malloc(sizeXq);
    yt *Yq = (yt *) malloc(sizeYq);
    zt *Zq = (zt *) malloc(sizeZq);

    float *dX;
    float *dY;
    float *dZ;
    float *dXq;
    float *dYq;
    float *dZq;

    X[0] = 0;
    X[1] = 1;

    Y[0] = 0;
    Y[1] = 1;

    Z[0] = 1;
    Z[1] = 1;
    Z[2] = 2;
    Z[3] = 2;


    cudaMalloc((void **)&dX, sizeX);
    cudaMalloc((void **)&dY, sizeY);
    cudaMalloc((void **)&dZ, sizeZ);
    cudaMalloc((void **)&dXq, sizeXq);
    cudaMalloc((void **)&dYq, sizeYq);
    cudaMalloc((void **)&dZq, sizeZq);

    for( int i = 0; i < 5; i++)
    {
        Xq[i] = (xt)i/4;
        Yq[i] = (yt)i/4;
    }

    cudaMemcpy(dX, X, sizeX, cudaMemcpyHostToDevice);
    cudaMemcpy(dY, Y, sizeY, cudaMemcpyHostToDevice);
    cudaMemcpy(dZ, Z, sizeZ, cudaMemcpyHostToDevice);
    cudaMemcpy(dXq, Xq, sizeXq, cudaMemcpyHostToDevice);
    cudaMemcpy(dYq, Yq, sizeYq, cudaMemcpyHostToDevice);


    int threadsPerBlock = 256;
    int blocksPerGrid =
            (points + threadsPerBlock - 1) / threadsPerBlock;


    interp2<<<blocksPerGrid, threadsPerBlock>>>(dX, nX,
                                                        dY, nY,
                                                        dZ,
                                                        dXq, dYq,
                                                        dZq, points);


    cudaMemcpy(Zq, dZq, sizeZq, cudaMemcpyDeviceToHost);

    for (int i =0; i<5; i++)
    {
        std::cout<<Xq[i]<<" "<<Yq[i]<<" "<<Zq[i]<<std::endl;
    }

    cudaFree(dX);
    cudaFree(dY);
    cudaFree(dZ);
    cudaFree(dXq);
    cudaFree(dYq);
    cudaFree(dZq);


    return 0;
}
