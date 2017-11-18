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

constant uint BLOCK_SIZE = 8;

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
    const uint2 gidIn = gid << 3;
    
    if (n - gidIn.x < BLOCK_SIZE || m - gidIn.y < BLOCK_SIZE) return;
    else {
        threadgroup float As[BLOCK_SIZE][BLOCK_SIZE];
        threadgroup float Bs[BLOCK_SIZE][BLOCK_SIZE];
        threadgroup float4 *As4 = (threadgroup float4 *)As;
        threadgroup float4 *Bs4 = (threadgroup float4 *)Bs;
        thread float Cval = 0.0f;
        
        for (uint i = 0; i < k / BLOCK_SIZE; ++i) {
            As[tid.y][tid.x] = A[(gidIn.y + tid.y) * k + i * BLOCK_SIZE + tid.x];
            Bs[tid.y][tid.x] = B[(i * BLOCK_SIZE + tid.x) * n + gidIn.x + tid.y];
            threadgroup_barrier(mem_flags::mem_none);

            thread float4 tmp = As4[BLOCK_SIZE / 4 * tid.y] * Bs4[BLOCK_SIZE / 4 * tid.x] + As4[BLOCK_SIZE / 4 * tid.y + 1] * Bs4[BLOCK_SIZE / 4 * tid.x + 1];
            Cval += (tmp.x + tmp.y + tmp.z + tmp.w);
            threadgroup_barrier(mem_flags::mem_none);
        }
        
        const uint Cidx = (gidIn.y + tid.y) * n + gidIn.x + tid.x;
        C[Cidx] = Cval * alpha + C[Cidx] * beta;
    }
}
