#include <metal_stdlib>

using namespace metal;

typedef struct {
    bool trans_a, trans_b;
    ushort m, n, k, rmd_a, rmd_b;
    float alpha, beta;
} MetalMatrixDim;

kernel void MetalGemmNaive(const device float        *A   [[ buffer(0) ]],
                           const device float        *B   [[ buffer(1) ]],
                           device float              *C   [[ buffer(2) ]],
                           constant MetalMatrixDim&  dims [[ buffer(3) ]],
                           ushort2                   gid  [[ thread_position_in_grid ]]) {
    const ushort m = dims.m;
    const ushort n = dims.n;
    const ushort k = dims.k;
    
    const ushort rmd_a = dims.rmd_a;
    const ushort rmd_b = dims.rmd_b;
    
    const bool trans_a = dims.trans_a;
    const bool trans_b = dims.trans_b;
    
    const float alpha = dims.alpha;
    const float beta = dims.beta;
    
    const ushort2 gidIn = ushort2(gid.x << 3, gid.y << 3);
    
    if (gidIn.x >= m || gidIn.y >= n) return;
    
    const device float4 *a = trans_a ? (const device float4 *)(A + gidIn.x) :
                                       (const device float4 *)(A + k * gidIn.x);
    const device float4 *b = trans_b ? (const device float4 *)(B + k * gidIn.y) :
                                       (const device float4 *)(B + gidIn.y);
    
    C += n * gidIn.x;
    
    const device float4 *Bend = trans_b ? (const device float4 *)((const device float *)b + k) :
                                          (const device float4 *)((const device float *)b + k * n);
    
    float4 s0  = 0.0f, s1  = 0.0f;
    float4 s2  = 0.0f, s3  = 0.0f;
    float4 s4  = 0.0f, s5  = 0.0f;
    float4 s6  = 0.0f, s7  = 0.0f;
    float4 s8  = 0.0f, s9  = 0.0f;
    float4 s10 = 0.0f, s11 = 0.0f;
    float4 s12 = 0.0f, s13 = 0.0f;
    float4 s14 = 0.0f, s15 = 0.0f;
    
    if (gidIn.x + 8 > m || gidIn.y + 8 > n) {
        ushort remainderA = (gidIn.x + 8 > m) ? rmd_a: 8;
        ushort remainderB = (gidIn.y + 8 > n) ? rmd_b: 8;
        
        do {
            float4 aCurr0 = 0.0f;
            float4 aCurr1 = 0.0f;
            float4 bCurr0 = 0.0f;
            float4 bCurr1 = 0.0f;
            
            switch (remainderA) {
                case 1:
                    aCurr0.x = trans_a ? a[0].x : *((device float *)a + k * 0);
                    break;
                case 2:
                    aCurr0.xy = trans_a ? a[0].xy : float2(*((device float *)a + k * 0),
                                                           *((device float *)a + k * 1));
                    break;
                case 3:
                    aCurr0.xyz = trans_a ? a[0].xyz : float3(*((device float *)a + k * 0),
                                                             *((device float *)a + k * 1),
                                                             *((device float *)a + k * 2));
                    break;
                case 4:
                    aCurr0 = trans_a ? a[0] : float4(*((device float *)a + k * 0),
                                                     *((device float *)a + k * 1),
                                                     *((device float *)a + k * 2),
                                                     *((device float *)a + k * 3));
                    break;
                case 5:
                    aCurr0 = trans_a ? a[0] : float4(*((device float *)a + k * 0),
                                                     *((device float *)a + k * 1),
                                                     *((device float *)a + k * 2),
                                                     *((device float *)a + k * 3));
                    aCurr1.x = trans_a ? a[1].x : *((device float *)a + k * 4);
                    break;
                case 6:
                    aCurr0 = trans_a ? a[0] : float4(*((device float *)a + k * 0),
                                                     *((device float *)a + k * 1),
                                                     *((device float *)a + k * 2),
                                                     *((device float *)a + k * 3));
                    aCurr1.xy = trans_a ? a[1].xy : float2(*((device float *)a + k * 4),
                                                           *((device float *)a + k * 5));
                    break;
                case 7:
                    aCurr0 = trans_a ? a[0] : float4(*((device float *)a + k * 0),
                                                     *((device float *)a + k * 1),
                                                     *((device float *)a + k * 2),
                                                     *((device float *)a + k * 3));
                    aCurr1.xyz = trans_a ? a[1].xyz : float3(*((device float *)a + k * 4),
                                                             *((device float *)a + k * 5),
                                                             *((device float *)a + k * 6));
                    break;
                case 8:
                    aCurr0 = trans_a ? a[0] : float4(*((device float *)a + k * 0),
                                                     *((device float *)a + k * 1),
                                                     *((device float *)a + k * 2),
                                                     *((device float *)a + k * 3));
                    aCurr1 = trans_a ? a[1] : float4(*((device float *)a + k * 4),
                                                     *((device float *)a + k * 5),
                                                     *((device float *)a + k * 6),
                                                     *((device float *)a + k * 7));
                    break;
                default:
                    break;
            }
            
            switch (remainderB) {
                case 1:
                    bCurr0.x = trans_b ? *((device float *)b + k * 0) : b[0].x;
                    break;
                case 2:
                    bCurr0.xy = trans_b ? float2(*((device float *)b + k * 0),
                                                 *((device float *)b + k * 1)) : b[0].xy;
                    break;
                case 3:
                    bCurr0.xyz = trans_b ? float3(*((device float *)b + k * 0),
                                                  *((device float *)b + k * 1),
                                                  *((device float *)b + k * 2)) : b[0].xyz;
                    break;
                case 4:
                    bCurr0 = trans_b ? float4(*((device float *)b + k * 0),
                                              *((device float *)b + k * 1),
                                              *((device float *)b + k * 2),
                                              *((device float *)b + k * 3)) : b[0];
                    break;
                case 5:
                    bCurr0 = trans_b ? float4(*((device float *)b + k * 0),
                                              *((device float *)b + k * 1),
                                              *((device float *)b + k * 2),
                                              *((device float *)b + k * 3)) : b[0];
                    bCurr1.x = trans_b ? *((device float *)b + k * 4) : b[1].x;
                    break;
                case 6:
                    bCurr0 = trans_b ? float4(*((device float *)b + k * 0),
                                              *((device float *)b + k * 1),
                                              *((device float *)b + k * 2),
                                              *((device float *)b + k * 3)) : b[0];
                    bCurr1.xy = trans_b ? float2(*((device float *)b + k * 4),
                                                 *((device float *)b + k * 5)) : b[1].xy;
                    break;
                case 7:
                    bCurr0 = trans_b ? float4(*((device float *)b + k * 0),
                                              *((device float *)b + k * 1),
                                              *((device float *)b + k * 2),
                                              *((device float *)b + k * 3)) : b[0];
                    bCurr1.xyz = trans_b ? float3(*((device float *)b + k * 4),
                                                  *((device float *)b + k * 5),
                                                  *((device float *)b + k * 6)) : b[1].xyz;
                    break;
                case 8:
                    bCurr0 = trans_b ? float4(*((device float *)b + k * 0),
                                              *((device float *)b + k * 1),
                                              *((device float *)b + k * 2),
                                              *((device float *)b + k * 3)) : b[0];
                    bCurr1 = trans_b ? float4(*((device float *)b + k * 4),
                                              *((device float *)b + k * 5),
                                              *((device float *)b + k * 6),
                                              *((device float *)b + k * 7)) : b[1];
                    break;
                default:
                    break;
            }
            
            s0   += (aCurr0.x * bCurr0);
            s2   += (aCurr0.y * bCurr0);
            s4   += (aCurr0.z * bCurr0);
            s6   += (aCurr0.w * bCurr0);
            
            s1   += (aCurr0.x * bCurr1);
            s3   += (aCurr0.y * bCurr1);
            s5   += (aCurr0.z * bCurr1);
            s7   += (aCurr0.w * bCurr1);
            
            s8   += (aCurr1.x * bCurr0);
            s10  += (aCurr1.y * bCurr0);
            s12  += (aCurr1.z * bCurr0);
            s14  += (aCurr1.w * bCurr0);
            
            s9   += (aCurr1.x * bCurr1);
            s11  += (aCurr1.y * bCurr1);
            s13  += (aCurr1.z * bCurr1);
            s15  += (aCurr1.w * bCurr1);
            
            a = trans_a ? (device float4 *)((device float *)a + m) : (device float4 *)((device float *)a + 1);
            b = trans_b ? (device float4 *)((device float *)b + 1) : (device float4 *)((device float *)b + n);
            
        } while (b < Bend);
        
        /*
                           |
         Quadrant 2 (s_11) | Quadrant 1 (s_12)
                           |
           s0 (x,y,z,w)    |   s1 (x,y,z,w)
           s2 (x,y,z,w)    |   s3 (x,y,z,w)
           s4 (x,y,z,w)    |   s5 (x,y,z,w)
           s6 (x,y,z,w)    |   s7 (x,y,z,w)
                           |
         ------------------+------------------
                           |
         Quadrant 3 (s_21) | Quadrant 4 (s_22)
                           |
           s8 (x,y,z,w)    |   s9 (x,y,z,w)
           s10(x,y,z,w)    |   s11(x,y,z,w)
           s12(x,y,z,w)    |   s13(x,y,z,w)
           s14(x,y,z,w)    |   s15(x,y,z,w)
                           |
         */
        
        float4x4 s_11 = float4x4(s0 ,s2 ,s4 ,s6 );
        float4x4 s_12 = float4x4(s1 ,s3 ,s5 ,s7 );
        float4x4 s_21 = float4x4(s8 ,s10,s12,s14);
        float4x4 s_22 = float4x4(s9 ,s11,s13,s15);
        
        if (remainderA <= 4 && remainderB <= 4) {
            // Quadrant 2
            device float4 *c = (device float4 *)(C + gidIn.y);
            switch (remainderB) {
                case 1:
                    for (ushort i = 0;  i < remainderA; i++) {
                        c[0].x = s_11[i].x * alpha + c[0].x * beta;
                        c = (device float4 *)((device float *)c + n);
                    }
                    break;
                case 2:
                    for (ushort i = 0;  i < remainderA; i++) {
                        c[0].xy = s_11[i].xy * alpha + c[0].xy * beta;
                        c = (device float4 *)((device float *)c + n);
                    }
                    break;
                case 3:
                    for (ushort i = 0;  i < remainderA; i++) {
                        c[0].xyz = s_11[i].xyz * alpha + c[0].xyz * beta;
                        c = (device float4 *)((device float *)c + n);
                    }
                    break;
                case 4:
                    for (ushort i = 0;  i < remainderA; i++) {
                        c[0] = s_11[i] * alpha + c[0] * beta;
                        c = (device float4 *)((device float *)c + n);
                    }
                    break;
                default:
                    break;
            }
        } else if (remainderA <= 4 && remainderB > 4) {
            // Quadrant 2
            device float4 *c = (device float4 *)(C + gidIn.y);
            for (ushort i = 0;  i < remainderA; i++) {
                c[0] = s_11[i] * alpha + c[0] * beta;
                c = (device float4 *)((device float *)c + n);
            }
            // Quadrant 1
            c = (device float4 *)(C + gidIn.y + 4);
            switch (remainderB - 4) {
                case 1:
                    for (ushort i = 0;  i < remainderA; i++) {
                        c[0].x = s_12[i].x * alpha + c[0].x * beta;
                        c = (device float4 *)((device float *)c + n);
                    }
                    break;
                case 2:
                    for (ushort i = 0;  i < remainderA; i++) {
                        c[0].xy = s_12[i].xy * alpha + c[0].xy * beta;
                        c = (device float4 *)((device float *)c + n);
                    }
                    break;
                case 3:
                    for (ushort i = 0;  i < remainderA; i++) {
                        c[0].xyz = s_12[i].xyz * alpha + c[0].xyz * beta;
                        c = (device float4 *)((device float *)c + n);
                    }
                    break;
                case 4:
                    for (ushort i = 0;  i < remainderA; i++) {
                        c[0] = s_12[i] * alpha + c[0] * beta;
                        c = (device float4 *)((device float *)c + n);
                    }
                    break;
                default:
                    break;
            }
        } else if (remainderA > 4 && remainderB <= 4) {
            // Quadrant 2
            device float4 *c = (device float4 *)(C + gidIn.y);
            switch (remainderB) {
                case 1:
                    for (ushort i = 0;  i < 4; i++) {
                        c[0].x = s_11[i].x * alpha + c[0].x * beta;
                        c = (device float4 *)((device float *)c + n);
                    }
                    break;
                case 2:
                    for (ushort i = 0;  i < 4; i++) {
                        c[0].xy = s_11[i].xy * alpha + c[0].xy * beta;
                        c = (device float4 *)((device float *)c + n);
                    }
                    break;
                case 3:
                    for (ushort i = 0;  i < 4; i++) {
                        c[0].xyz = s_11[i].xyz * alpha + c[0].xyz * beta;
                        c = (device float4 *)((device float *)c + n);
                    }
                    break;
                case 4:
                    for (ushort i = 0;  i < 4; i++) {
                        c[0] = s_11[i] * alpha + c[0] * beta;
                        c = (device float4 *)((device float *)c + n);
                    }
                    break;
                default:
                    break;
            }
            // Quadrant 3
            c = (device float4 *)(C + gidIn.y + n * 4);
            switch (remainderB) {
                case 1:
                    for (ushort i = 0;  i < remainderA - 4; i++) {
                        c[0].x = s_21[i].x * alpha + c[0].x * beta;
                        c = (device float4 *)((device float *)c + n);
                    }
                    break;
                case 2:
                    for (ushort i = 0;  i < remainderA - 4; i++) {
                        c[0].xy = s_21[i].xy * alpha + c[0].xy * beta;
                        c = (device float4 *)((device float *)c + n);
                    }
                    break;
                case 3:
                    for (ushort i = 0;  i < remainderA - 4; i++) {
                        c[0].xyz = s_21[i].xyz * alpha + c[0].xyz * beta;
                        c = (device float4 *)((device float *)c + n);
                    }
                    break;
                case 4:
                    for (ushort i = 0;  i < remainderA - 4; i++) {
                        c[0] = s_21[i] * alpha + c[0] * beta;
                        c = (device float4 *)((device float *)c + n);
                    }
                    break;
                default:
                    break;
            }
        } else if (remainderA > 4 && remainderB > 4) {
            // Quadrant 2
            device float4 *c = (device float4 *)(C + gidIn.y);
            for (ushort i = 0;  i < 4; i++) {
                c[0] = s_11[i] * alpha + c[0] * beta;
                c = (device float4 *)((device float *)c + n);
            }
            // Quadrant 1
            c = (device float4 *)(C + gidIn.y + 4);
            switch (remainderB - 4) {
                case 1:
                    for (ushort i = 0;  i < 4; i++) {
                        c[0].x = s_12[i].x * alpha + c[0].x * beta;
                        c = (device float4 *)((device float *)c + n);
                    }
                    break;
                case 2:
                    for (ushort i = 0;  i < 4; i++) {
                        c[0].xy = s_12[i].xy * alpha + c[0].xy * beta;
                        c = (device float4 *)((device float *)c + n);
                    }
                    break;
                case 3:
                    for (ushort i = 0;  i < 4; i++) {
                        c[0].xyz = s_12[i].xyz * alpha + c[0].xyz * beta;
                        c = (device float4 *)((device float *)c + n);
                    }
                    break;
                case 4:
                    for (ushort i = 0;  i < 4; i++) {
                        c[0] = s_12[i] * alpha + c[0] * beta;
                        c = (device float4 *)((device float *)c + n);
                    }
                    break;
                default:
                    break;
            }
            // Quadrant 3
            c = (device float4 *)(C + gidIn.y + n * 4);
            for (ushort i = 0;  i < remainderA - 4; i++) {
                c[0] = s_21[i] * alpha + c[0] * beta;
                c = (device float4 *)((device float *)c + n);
            }
            // Quadrant 4
            c = (device float4 *)(C + gidIn.y + n * 4 + 4);
            switch (remainderB - 4) {
                case 1:
                    for (ushort i = 0;  i < remainderA - 4; i++) {
                        c[0].x = s_22[i].x * alpha + c[0].x * beta;
                        c = (device float4 *)((device float *)c + n);
                    }
                    break;
                case 2:
                    for (ushort i = 0;  i < remainderA - 4; i++) {
                        c[0].xy = s_22[i].xy * alpha + c[0].xy * beta;
                        c = (device float4 *)((device float *)c + n);
                    }
                    break;
                case 3:
                    for (ushort i = 0;  i < remainderA - 4; i++) {
                        c[0].xyz = s_22[i].xyz * alpha + c[0].xyz * beta;
                        c = (device float4 *)((device float *)c + n);
                    }
                    break;
                case 4:
                    for (ushort i = 0;  i < remainderA - 4; i++) {
                        c[0] = s_22[i] * alpha + c[0] * beta;
                        c = (device float4 *)((device float *)c + n);
                    }
                    break;
                default:
                    break;
            }
        }
    } else {
        do {
            float4 aCurr0 = trans_a ? a[0] : float4(*((device float *)a + k * 0),
                                                    *((device float *)a + k * 1),
                                                    *((device float *)a + k * 2),
                                                    *((device float *)a + k * 3));
            float4 aCurr1 = trans_a ? a[1] : float4(*((device float *)a + k * 4),
                                                    *((device float *)a + k * 5),
                                                    *((device float *)a + k * 6),
                                                    *((device float *)a + k * 7));
            
            float4 bCurr0 = trans_b ? float4(*((device float *)b + k * 0),
                                             *((device float *)b + k * 1),
                                             *((device float *)b + k * 2),
                                             *((device float *)b + k * 3)) : b[0];
            float4 bCurr1 = trans_b ? float4(*((device float *)b + k * 4),
                                             *((device float *)b + k * 5),
                                             *((device float *)b + k * 6),
                                             *((device float *)b + k * 7)) : b[1];
            
            s0   += (aCurr0.x * bCurr0);
            s2   += (aCurr0.y * bCurr0);
            s4   += (aCurr0.z * bCurr0);
            s6   += (aCurr0.w * bCurr0);
            
            s1   += (aCurr0.x * bCurr1);
            s3   += (aCurr0.y * bCurr1);
            s5   += (aCurr0.z * bCurr1);
            s7   += (aCurr0.w * bCurr1);
            
            s8   += (aCurr1.x * bCurr0);
            s10  += (aCurr1.y * bCurr0);
            s12  += (aCurr1.z * bCurr0);
            s14  += (aCurr1.w * bCurr0);
            
            s9   += (aCurr1.x * bCurr1);
            s11  += (aCurr1.y * bCurr1);
            s13  += (aCurr1.z * bCurr1);
            s15  += (aCurr1.w * bCurr1);
            
            a = trans_a ? (device float4 *)((device float *)a + m) : (device float4 *)((device float *)a + 1);
            b = trans_b ? (device float4 *)((device float *)b + 1) : (device float4 *)((device float *)b + n);
            
        } while (b < Bend);
        
        device float4 *c = (device float4 *)(C + gidIn.y);
        
        c[0] = s0 * alpha + c[0] * beta;  c[1] = s1 * alpha + c[1] * beta;  c = (device float4 *)((device float *)c + n);
        c[0] = s2 * alpha + c[0] * beta;  c[1] = s3 * alpha + c[1] * beta;  c = (device float4 *)((device float *)c + n);
        c[0] = s4 * alpha + c[0] * beta;  c[1] = s5 * alpha + c[1] * beta;  c = (device float4 *)((device float *)c + n);
        c[0] = s6 * alpha + c[0] * beta;  c[1] = s7 * alpha + c[1] * beta;  c = (device float4 *)((device float *)c + n);
        c[0] = s8 * alpha + c[0] * beta;  c[1] = s9 * alpha + c[1] * beta;  c = (device float4 *)((device float *)c + n);
        c[0] = s10 * alpha + c[0] * beta; c[1] = s11 * alpha + c[1] * beta; c = (device float4 *)((device float *)c + n);
        c[0] = s12 * alpha + c[0] * beta; c[1] = s13 * alpha + c[1] * beta; c = (device float4 *)((device float *)c + n);
        c[0] = s14 * alpha + c[0] * beta; c[1] = s15 * alpha + c[1] * beta;
    }
}
