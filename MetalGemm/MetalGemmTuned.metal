//
//  MetalGemmTuned.metal
//  MetalGemm
//
//  Created by Lun on 17/11/2017.
//  Copyright © 2017 Lun. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

typedef struct {
    bool trans_a, trans_b;
    ushort m, n, k, rmd_a, rmd_b;
    float alpha, beta;
} MetalMatrixDim;

#define BLOCK_SIZE_V1 16

kernel void
MetalGemmTunedV1(const device float        *A   [[ buffer(0) ]],
                 const device float        *B   [[ buffer(1) ]],
                 device float              *C   [[ buffer(2) ]],
                 constant MetalMatrixDim&  dims [[ buffer(3) ]],
                 uint2                     gid  [[ threadgroup_position_in_grid ]],
                 uint2                     tid  [[ thread_position_in_threadgroup ]]) {
    const uint m = dims.m;
    const uint n = dims.n;
    const uint k = dims.k;
    const float alpha = dims.alpha;
    const float beta  = dims.beta;
    const uint2 gidIn = gid * BLOCK_SIZE_V1;
    
    if (n - gidIn.x < BLOCK_SIZE_V1 || m - gidIn.y < BLOCK_SIZE_V1) return;
    else {
        threadgroup float As[BLOCK_SIZE_V1][BLOCK_SIZE_V1];
        threadgroup float Bs[BLOCK_SIZE_V1][BLOCK_SIZE_V1];
        threadgroup float4 *As4 = (threadgroup float4 *)As;
        threadgroup float4 *Bs4 = (threadgroup float4 *)Bs;
        float Cval = 0.0f;
        
        for (uchar i = 0; i < k / BLOCK_SIZE_V1; ++i) {
            As[tid.y][tid.x] = A[(gidIn.y + tid.y) * k + i * BLOCK_SIZE_V1 + tid.x];
            Bs[tid.y][tid.x] = B[(i * BLOCK_SIZE_V1 + tid.x) * n + gidIn.x + tid.y];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float4 tmp = float4(0.0f);
            for (uchar e = 0; e < BLOCK_SIZE_V1 / 4; ++e)
                tmp += As4[BLOCK_SIZE_V1 / 4 * tid.y + e] * Bs4[BLOCK_SIZE_V1 / 4 * tid.x + e];
            Cval += tmp.x + tmp.y + tmp.z + tmp.w;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        const uint Cidx = (gidIn.y + tid.y) * n + gidIn.x + tid.x;
        C[Cidx] = Cval * alpha + C[Cidx] * beta;
    }
}

#define BLOCK_SIZE_V2 16

kernel void
MetalGemmTunedV2(const device float        *A   [[ buffer(0) ]],
                 const device float        *B   [[ buffer(1) ]],
                 device float              *C   [[ buffer(2) ]],
                 constant MetalMatrixDim&  dims [[ buffer(3) ]],
                 uint2                     gid  [[ threadgroup_position_in_grid ]],
                 uint2                     tid  [[ thread_position_in_threadgroup ]]) {
    const uint m = dims.m;
    const uint n = dims.n;
    const uint k = dims.k;
    const float alpha = dims.alpha;
    const float beta  = dims.beta;
    const uint2 gidIn = gid * BLOCK_SIZE_V2;
    
    if (n - gidIn.x < BLOCK_SIZE_V2 || m - gidIn.y < BLOCK_SIZE_V2) return;
    else {
        threadgroup float4 As4[BLOCK_SIZE_V2][BLOCK_SIZE_V2 / 4];
        threadgroup float4 Bs4[BLOCK_SIZE_V2 / 4][BLOCK_SIZE_V2];
        threadgroup float *As = (threadgroup float *)As4;
        float4 Cval = float4(0.0f);
        
        for (uchar i = 0; i < k / BLOCK_SIZE_V2; ++i) {
            As4[tid.y][tid.x] = *((const device float4 *)(A + (gidIn.y + tid.y) * k + BLOCK_SIZE_V2 * i + tid.x * 4));
            Bs4[tid.x][tid.y] = *((const device float4 *)(B + (BLOCK_SIZE_V2 * i + tid.y) * n + gidIn.x + tid.x * 4));
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            for (uchar e = 0; e < BLOCK_SIZE_V2; ++e)
                Cval += *(As + BLOCK_SIZE_V2 * tid.y + e) * Bs4[tid.x][e];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        device float4 *c = (device float4 *)(C + (gidIn.y + tid.y) * n + gidIn.x + tid.x * 4);
        *c = Cval * alpha + *c * beta;
    }
}

#define BLOCK_SIZE_V3 16

kernel void
MetalGemmTunedV3(const device float        *A   [[ buffer(0) ]],
                 const device float        *B   [[ buffer(1) ]],
                 device float              *C   [[ buffer(2) ]],
                 constant MetalMatrixDim&  dims [[ buffer(3) ]],
                 uint2                     gid  [[ threadgroup_position_in_grid ]],
                 uint2                     tid  [[ thread_position_in_threadgroup ]]) {
    const uint m = dims.m;
    const uint n = dims.n;
    const uint k = dims.k;
    const float alpha = dims.alpha;
    const float beta  = dims.beta;
    const uint2 gidIn = gid * BLOCK_SIZE_V3;
    
    if (n - gidIn.x < BLOCK_SIZE_V3 || m - gidIn.y < BLOCK_SIZE_V3) return;
    else {
        threadgroup float4 As4[BLOCK_SIZE_V3][BLOCK_SIZE_V3 / 4];
        threadgroup float4 Bs4[BLOCK_SIZE_V3 / 4][BLOCK_SIZE_V3];
        float4x4 Cval = float4x4(0.0f);
        
        for (uchar i = 0; i < k / BLOCK_SIZE_V3; ++i) {
            As4[tid.y][tid.x] = *((const device float4 *)(A + (gidIn.y + tid.y) * k + BLOCK_SIZE_V3 * i + tid.x * 4));
            const device float *b = B + (BLOCK_SIZE_V3 * i + tid.x * 4) * n + gidIn.x + tid.y;
            Bs4[tid.y / 4][tid.x * 4 + tid.y % 4] = float4(*(b + n * 0),
                                                           *(b + n * 1),
                                                           *(b + n * 2),
                                                           *(b + n * 3));
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            for (uchar e = 0; e < BLOCK_SIZE_V3 / 4; ++e) {
                Cval[0] += As4[tid.y][e] * Bs4[tid.x][e * 4 + 0];
                Cval[1] += As4[tid.y][e] * Bs4[tid.x][e * 4 + 1];
                Cval[2] += As4[tid.y][e] * Bs4[tid.x][e * 4 + 2];
                Cval[3] += As4[tid.y][e] * Bs4[tid.x][e * 4 + 3];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        Cval = transpose(Cval);
        
        device float4 *c = (device float4 *)(C + (gidIn.y + tid.y) * n + gidIn.x + tid.x * 4);
        *c = (Cval[0] + Cval[1] + Cval[2] + Cval[3]) * alpha + *c * beta;
    }
}
