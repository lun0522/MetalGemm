//
//  MetalGemmTunedV1.metal
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

constant uint BLOCK_SIZE = 1 << 3;

kernel void MetalGemmTunedV1(const device float        *A   [[ buffer(0) ]],
                             const device float        *B   [[ buffer(1) ]],
                             device float              *C   [[ buffer(2) ]],
                             constant MetalMatrixDim&  dims [[ buffer(3) ]],
                             uint2                     gid  [[ threadgroup_position_in_grid ]],
                             uint2                     tid  [[ thread_position_in_threadgroup ]]) {
    const uint m = dims.m;
    const uint n = dims.n;
    const uint k = dims.k;
    const uint2 gidIn = gid << 3;
    
    if (n - gidIn.x < BLOCK_SIZE || m - gidIn.y < BLOCK_SIZE) return;
    else {
        threadgroup float As[BLOCK_SIZE][BLOCK_SIZE];
        threadgroup float Bs[BLOCK_SIZE][BLOCK_SIZE];
        thread float Cval = 0.0f;
        
        for (uint idx = 0; idx < k / BLOCK_SIZE; ++idx) {
            As[tid.y][tid.x] = A[(gidIn.y + tid.y) * k + idx * BLOCK_SIZE + tid.x];
            Bs[tid.x][tid.y] = B[(idx * BLOCK_SIZE + tid.y) * n + gidIn.x + tid.x];
            threadgroup_barrier(mem_flags::mem_none);

            for (uint e = 0; e < BLOCK_SIZE; ++e)
                Cval += As[tid.y][e] * Bs[tid.x][e];
            threadgroup_barrier(mem_flags::mem_none);
        }
        
        C[(gidIn.y + tid.y) * n + gidIn.x + tid.x] = Cval;
    }
}
