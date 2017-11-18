//
//  MetalGemmTunedV1.h
//  MetalGemm
//
//  Created by Lun on 17/11/2017.
//  Copyright © 2017 Lun. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface MetalGemmTunedV1 : NSObject

void metal_gemm_tuned_v1(const BOOL transA,
                         const BOOL transB,
                         const int M,
                         const int N,
                         const int K,
                         const float alpha,
                         const float *A,
                         const float *B,
                         const float beta,
                         float *C);

@end
