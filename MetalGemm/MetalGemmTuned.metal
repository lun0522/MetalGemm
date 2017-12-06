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

            float4 tmp = (As4[BLOCK_SIZE_V1 / 4 * tid.y + 0] * Bs4[BLOCK_SIZE_V1 / 4 * tid.x + 0] +
                          As4[BLOCK_SIZE_V1 / 4 * tid.y + 1] * Bs4[BLOCK_SIZE_V1 / 4 * tid.x + 1] +
                          As4[BLOCK_SIZE_V1 / 4 * tid.y + 2] * Bs4[BLOCK_SIZE_V1 / 4 * tid.x + 2] +
                          As4[BLOCK_SIZE_V1 / 4 * tid.y + 3] * Bs4[BLOCK_SIZE_V1 / 4 * tid.x + 3]);
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
            
            Cval += (*(As + BLOCK_SIZE_V2 * tid.y + 0) * Bs4[tid.x][0] +
                     *(As + BLOCK_SIZE_V2 * tid.y + 1) * Bs4[tid.x][1] +
                     *(As + BLOCK_SIZE_V2 * tid.y + 2) * Bs4[tid.x][2] +
                     *(As + BLOCK_SIZE_V2 * tid.y + 3) * Bs4[tid.x][3] +
                     *(As + BLOCK_SIZE_V2 * tid.y + 4) * Bs4[tid.x][4] +
                     *(As + BLOCK_SIZE_V2 * tid.y + 5) * Bs4[tid.x][5] +
                     *(As + BLOCK_SIZE_V2 * tid.y + 6) * Bs4[tid.x][6] +
                     *(As + BLOCK_SIZE_V2 * tid.y + 7) * Bs4[tid.x][7] +
                     *(As + BLOCK_SIZE_V2 * tid.y + 8) * Bs4[tid.x][8] +
                     *(As + BLOCK_SIZE_V2 * tid.y + 9) * Bs4[tid.x][9] +
                     *(As + BLOCK_SIZE_V2 * tid.y + 10) * Bs4[tid.x][10] +
                     *(As + BLOCK_SIZE_V2 * tid.y + 11) * Bs4[tid.x][11] +
                     *(As + BLOCK_SIZE_V2 * tid.y + 12) * Bs4[tid.x][12] +
                     *(As + BLOCK_SIZE_V2 * tid.y + 13) * Bs4[tid.x][13] +
                     *(As + BLOCK_SIZE_V2 * tid.y + 14) * Bs4[tid.x][14] +
                     *(As + BLOCK_SIZE_V2 * tid.y + 15) * Bs4[tid.x][15]);
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
        threadgroup float *As = (threadgroup float *)As4;
        float4 Cval_1 = float4(0.0f), Cval_2 = float4(0.0f);
        
        for (uchar i = 0; i < k / BLOCK_SIZE_V2; ++i) {
            As4[tid.y][tid.x * 2] = *((const device float4 *)(A + (gidIn.y + tid.y) * k + BLOCK_SIZE_V2 * i + tid.x * 2 * 4));
            As4[tid.y][tid.x * 2 + 1] = *((const device float4 *)(A + (gidIn.y + tid.y) * k + BLOCK_SIZE_V2 * i + tid.x * 2 * 4 + 4));
            Bs4[tid.x * 2][tid.y] = *((const device float4 *)(B + (BLOCK_SIZE_V2 * i + tid.y) * n + gidIn.x + tid.x * 2 * 4));
            Bs4[tid.x * 2 + 1][tid.y] = *((const device float4 *)(B + (BLOCK_SIZE_V2 * i + tid.y) * n + gidIn.x + tid.x * 2 * 4 + 4));
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            Cval_1 += *(As + BLOCK_SIZE_V2 * tid.y + 0) * Bs4[tid.x * 2][0];
            Cval_1 += *(As + BLOCK_SIZE_V2 * tid.y + 1) * Bs4[tid.x * 2][1];
            Cval_1 += *(As + BLOCK_SIZE_V2 * tid.y + 2) * Bs4[tid.x * 2][2];
            Cval_1 += *(As + BLOCK_SIZE_V2 * tid.y + 3) * Bs4[tid.x * 2][3];
            Cval_1 += *(As + BLOCK_SIZE_V2 * tid.y + 4) * Bs4[tid.x * 2][4];
            Cval_1 += *(As + BLOCK_SIZE_V2 * tid.y + 5) * Bs4[tid.x * 2][5];
            Cval_1 += *(As + BLOCK_SIZE_V2 * tid.y + 6) * Bs4[tid.x * 2][6];
            Cval_1 += *(As + BLOCK_SIZE_V2 * tid.y + 7) * Bs4[tid.x * 2][7];
            Cval_1 += *(As + BLOCK_SIZE_V2 * tid.y + 8) * Bs4[tid.x * 2][8];
            Cval_1 += *(As + BLOCK_SIZE_V2 * tid.y + 9) * Bs4[tid.x * 2][9];
            Cval_1 += *(As + BLOCK_SIZE_V2 * tid.y + 10) * Bs4[tid.x * 2][10];
            Cval_1 += *(As + BLOCK_SIZE_V2 * tid.y + 11) * Bs4[tid.x * 2][11];
            Cval_1 += *(As + BLOCK_SIZE_V2 * tid.y + 12) * Bs4[tid.x * 2][12];
            Cval_1 += *(As + BLOCK_SIZE_V2 * tid.y + 13) * Bs4[tid.x * 2][13];
            Cval_1 += *(As + BLOCK_SIZE_V2 * tid.y + 14) * Bs4[tid.x * 2][14];
            Cval_1 += *(As + BLOCK_SIZE_V2 * tid.y + 15) * Bs4[tid.x * 2][15];
            
            Cval_2 += *(As + BLOCK_SIZE_V2 * tid.y + 0) * Bs4[tid.x * 2 + 1][0];
            Cval_2 += *(As + BLOCK_SIZE_V2 * tid.y + 1) * Bs4[tid.x * 2 + 1][1];
            Cval_2 += *(As + BLOCK_SIZE_V2 * tid.y + 2) * Bs4[tid.x * 2 + 1][2];
            Cval_2 += *(As + BLOCK_SIZE_V2 * tid.y + 3) * Bs4[tid.x * 2 + 1][3];
            Cval_2 += *(As + BLOCK_SIZE_V2 * tid.y + 4) * Bs4[tid.x * 2 + 1][4];
            Cval_2 += *(As + BLOCK_SIZE_V2 * tid.y + 5) * Bs4[tid.x * 2 + 1][5];
            Cval_2 += *(As + BLOCK_SIZE_V2 * tid.y + 6) * Bs4[tid.x * 2 + 1][6];
            Cval_2 += *(As + BLOCK_SIZE_V2 * tid.y + 7) * Bs4[tid.x * 2 + 1][7];
            Cval_2 += *(As + BLOCK_SIZE_V2 * tid.y + 8) * Bs4[tid.x * 2 + 1][8];
            Cval_2 += *(As + BLOCK_SIZE_V2 * tid.y + 9) * Bs4[tid.x * 2 + 1][9];
            Cval_2 += *(As + BLOCK_SIZE_V2 * tid.y + 10) * Bs4[tid.x * 2 + 1][10];
            Cval_2 += *(As + BLOCK_SIZE_V2 * tid.y + 11) * Bs4[tid.x * 2 + 1][11];
            Cval_2 += *(As + BLOCK_SIZE_V2 * tid.y + 12) * Bs4[tid.x * 2 + 1][12];
            Cval_2 += *(As + BLOCK_SIZE_V2 * tid.y + 13) * Bs4[tid.x * 2 + 1][13];
            Cval_2 += *(As + BLOCK_SIZE_V2 * tid.y + 14) * Bs4[tid.x * 2 + 1][14];
            Cval_2 += *(As + BLOCK_SIZE_V2 * tid.y + 15) * Bs4[tid.x * 2 + 1][15];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        device float4 *c = (device float4 *)(C + (gidIn.y + tid.y) * n + gidIn.x + tid.x * 8);
        *c = Cval_1 * alpha + *c * beta; c++;
        *c = Cval_2 * alpha + *c * beta;
    }
}

#define BLOCK_SIZE_V4 16

kernel void
MetalGemmTunedV4(const device float        *A   [[ buffer(0) ]],
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
    const uint2 gidIn = gid * BLOCK_SIZE_V4;
    
    if (n - gidIn.x < BLOCK_SIZE_V4 || m - gidIn.y < BLOCK_SIZE_V4) return;
    else {
        threadgroup float4 As4[BLOCK_SIZE_V4][BLOCK_SIZE_V4 / 4];
        threadgroup float4 Bs4[BLOCK_SIZE_V4 / 4][BLOCK_SIZE_V4];
        float4x4 Cval = float4x4(0.0f);
        
        for (uchar i = 0; i < k / BLOCK_SIZE_V4; ++i) {
            As4[tid.y][tid.x] = *((const device float4 *)(A + (gidIn.y + tid.y) * k + BLOCK_SIZE_V4 * i + tid.x * 4));
            const device float *b = B + (BLOCK_SIZE_V4 * i + tid.x * 4) * n + gidIn.x + tid.y;
            Bs4[tid.y / 4][tid.x * 4 + tid.y % 4] = float4(*(b + n * 0),
                                                           *(b + n * 1),
                                                           *(b + n * 2),
                                                           *(b + n * 3));
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            Cval[0] += As4[tid.y][0] * Bs4[tid.x][0 * 4 + 0];
            Cval[1] += As4[tid.y][0] * Bs4[tid.x][0 * 4 + 1];
            Cval[2] += As4[tid.y][0] * Bs4[tid.x][0 * 4 + 2];
            Cval[3] += As4[tid.y][0] * Bs4[tid.x][0 * 4 + 3];
            Cval[0] += As4[tid.y][1] * Bs4[tid.x][1 * 4 + 0];
            Cval[1] += As4[tid.y][1] * Bs4[tid.x][1 * 4 + 1];
            Cval[2] += As4[tid.y][1] * Bs4[tid.x][1 * 4 + 2];
            Cval[3] += As4[tid.y][1] * Bs4[tid.x][1 * 4 + 3];
            Cval[0] += As4[tid.y][2] * Bs4[tid.x][2 * 4 + 0];
            Cval[1] += As4[tid.y][2] * Bs4[tid.x][2 * 4 + 1];
            Cval[2] += As4[tid.y][2] * Bs4[tid.x][2 * 4 + 2];
            Cval[3] += As4[tid.y][2] * Bs4[tid.x][2 * 4 + 3];
            Cval[0] += As4[tid.y][3] * Bs4[tid.x][3 * 4 + 0];
            Cval[1] += As4[tid.y][3] * Bs4[tid.x][3 * 4 + 1];
            Cval[2] += As4[tid.y][3] * Bs4[tid.x][3 * 4 + 2];
            Cval[3] += As4[tid.y][3] * Bs4[tid.x][3 * 4 + 3];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        Cval = transpose(Cval);
        
        device float4 *c = (device float4 *)(C + (gidIn.y + tid.y) * n + gidIn.x + tid.x * 4);
        *c = (Cval[0] + Cval[1] + Cval[2] + Cval[3]) * alpha + *c * beta;
    }
}
