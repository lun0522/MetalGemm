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

#define BLOCK_SIZE_V1 8

kernel void MetalGemmTunedV1(const device float        *A   [[ buffer(0) ]],
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
        thread float Cval = 0.0f;
        
        for (uint i = 0; i < k / BLOCK_SIZE_V1; ++i) {
            As[tid.y][tid.x] = A[(gidIn.y + tid.y) * k + i * BLOCK_SIZE_V1 + tid.x];
            Bs[tid.y][tid.x] = B[(i * BLOCK_SIZE_V1 + tid.x) * n + gidIn.x + tid.y];
            threadgroup_barrier(mem_flags::mem_none);

            thread float4 tmp = As4[BLOCK_SIZE_V1 / 4 * tid.y] * Bs4[BLOCK_SIZE_V1 / 4 * tid.x] + As4[BLOCK_SIZE_V1 / 4 * tid.y + 1] * Bs4[BLOCK_SIZE_V1 / 4 * tid.x + 1];
            Cval += (tmp.x + tmp.y + tmp.z + tmp.w);
            threadgroup_barrier(mem_flags::mem_none);
        }
        
        const uint Cidx = (gidIn.y + tid.y) * n + gidIn.x + tid.x;
        C[Cidx] = Cval * alpha + C[Cidx] * beta;
    }
}

#define BLOCK_SIZE_V2_X 16
#define BLOCK_SIZE_V2_Y 8
#define BLOCK_SIZE_V2_K 16

kernel void MetalGemmTunedV2(const device float        *A   [[ buffer(0) ]],
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
    const uint2 gidIn = uint2(gid.x * BLOCK_SIZE_V2_X, gid.y * BLOCK_SIZE_V2_Y);
    
    if (n - gidIn.x < BLOCK_SIZE_V1 || m - gidIn.y < BLOCK_SIZE_V1) return;
    else {
        threadgroup float4 As4[BLOCK_SIZE_V2_Y][BLOCK_SIZE_V2_K / 4];
        threadgroup float4 Bs4[BLOCK_SIZE_V2_K / 4][BLOCK_SIZE_V2_X];
        threadgroup float *As = (threadgroup float *)As4;
        thread float4 Cval = float4(0.0f);
        
        for (uint i = 0; i < k / BLOCK_SIZE_V2_K; ++i) {
            As4[tid.y][tid.x] = *((const device float4 *)(A + (gidIn.y + tid.y) * k + i * BLOCK_SIZE_V2_K + tid.x * 4));
            Bs4[tid.x][tid.y * 2] = *((const device float4 *)(B + (i * BLOCK_SIZE_V2_K + tid.y * 2) * n + gidIn.x + tid.x * 4));
            Bs4[tid.x][tid.y * 2 + 1] = *((const device float4 *)(B + (i * BLOCK_SIZE_V2_K + tid.y * 2 + 1) * n + gidIn.x + tid.x * 4));
            threadgroup_barrier(mem_flags::mem_none);
            
            for (uint e = 0; e < BLOCK_SIZE_V2_K; ++e)
                Cval += (*(As + BLOCK_SIZE_V2_K * tid.y + e) * Bs4[tid.x][e]);
            threadgroup_barrier(mem_flags::mem_none);
        }
        
        device float4 *c = (device float4 *)(C + (gidIn.y + tid.y) * n + gidIn.x + tid.x * 4);
        *c = Cval * alpha + *c * beta;
    }
}
