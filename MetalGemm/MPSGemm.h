//
//  MPSGemm.h
//  MetalGemm
//
//  Created by Lun on 17/11/2017.
//  Copyright © 2017 Lun. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface MPSGemm : NSObject

void mps_gemm(const BOOL transA,
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
