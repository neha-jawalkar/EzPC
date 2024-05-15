#pragma once

#include "gpu_dpf.h"
#include "gpu_select.h"

typedef void (*tabGen)(int N, int scaleIn, int scaleOut, u8 *tab);
typedef double (*inlineFunc)(double inp);


template <typename T>
struct GPULUTKey
{
    int bout;
    GPUDPFKey k;
    u32 *maskU;
    GPUSelectKey<T> s;
};

template <typename T>
GPULUTKey<T> readGPULUTKey(uint8_t **key_as_bytes)
{
    GPULUTKey<T> l;
    l.bout = (int)**key_as_bytes;
    *key_as_bytes += sizeof(int);
    l.k = readGPUDPFKey(key_as_bytes);
    l.maskU = (u32 *)*key_as_bytes;
    *key_as_bytes += l.k.memSzOut;
    l.s = readGPUSelectKey<T>(key_as_bytes, l.k.M);
    return l;
}

template <typename T>
__global__ void reluSubGelu(int N, int scaleIn, int scaleOut, u8 *tab)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        double x = double(i) / (1LL << scaleIn);
        double g = x * 0.5 * (1 + erf(x * rsqrt(2.0)));
        ((T *)tab)[i] = T((max(0.0, x) - g) * (1LL << scaleOut));
        // printf("Tab[%d]=%ld\n", i, ((T *)tab)[i]);
    }
}

template <typename T>
__global__ void reluSubSilu(int N, int scaleIn, int scaleOut, u8 *tab)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        double x = double(i) / (1LL << scaleIn);
        double s = x / (1 + exp(-x));
        ((T *)tab)[i] = T((max(0.0, x) - s) * (1LL << scaleOut));
        // printf("Tab[%d]=%ld\n", i, ((T *)tab)[i]);
    }
}

template <typename T>
__global__ void nExpLsb(int N, int scaleIn, int scaleOut, u8 *tab)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        ((T *)tab)[i] = T(std::exp(-i / double(1LL << scaleIn)) * (1LL << scaleOut));
    }
}

template <typename T>
__global__ void nExpMsb(int N, int scaleIn, int scaleOut, u8 *tab)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        ((T *)tab)[i] = T(std::exp(-i / double(1LL << scaleIn)) * (1LL << scaleOut));
    }
}

template <typename T>
__global__ void inv(int N, int scaleIn, int scaleOut, u8 *tab)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0)
    {
        ((T *)tab)[i] = T((1ULL << (scaleIn + scaleOut)) / double(N));
    }
    else if (i > 0 && i < N)
    {
        ((T *)tab)[i] = T((1ULL << (scaleIn + scaleOut)) / double(i));
    }
}

template <typename T>
__global__ void invSqrt(int N, int extradiv, int scale, u8 *tab)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        u64 k = i % (1LL << 6);
        u64 m = i >> 6;
        double val = double(m + 128) * pow(2.0, k - 7);
        ((T *)tab)[i] = T(double(1LL << (2 * scale)) / sqrt(val / extradiv));
    }
}

template <typename T>
__global__ void identity(int N, int scaleIn, int scaleOut, u8 *tab)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        ((T *)tab)[i] = T(i);
    }
}

template <typename T, tabGen t>
T *genLUT(int bin, int scaleIn, int scaleOut)
{
    assert(bin > 7);
    assert(bin < 32);
    int N = 1 << bin;
    T *d_tab = (T *)gpuMalloc(N * sizeof(T));
    t<<<(N - 1) / 128 + 1, 128>>>(N, scaleIn, scaleOut, (u8 *)d_tab);
    return d_tab;
}

#include "gpu_lut.cu"