//
//  ViewController.m
//  MetalGemm
//
//  Created by Lun on 17/11/2017.
//  Copyright © 2017 Lun. All rights reserved.
//

#import <Accelerate/Accelerate.h>
#import "MPSGemm.h"
#import "MetalGemm.h"
#import "ViewController.h"

@interface ViewController () {
    NSMutableArray *mpsTime;
    NSMutableArray *metalNaiveTime;
    NSMutableArray *metalTunedTime;
}

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
    MetalGemm *metalGemmNaive = [[MetalGemm alloc] initWithKernel:@"MetalGemmNaive"
                                                 threadgroupWidth:4
                                                threadgroupHeight:8
                                          threadgroupCoveredWidth:4 * 8
                                         threadgroupCoveredHeight:8 * 8];
    
    MetalGemm *metalGemmTunedV2 = [[MetalGemm alloc] initWithKernel:@"MetalGemmTunedV2"
                                                   threadgroupWidth:4
                                                  threadgroupHeight:8
                                            threadgroupCoveredWidth:16
                                           threadgroupCoveredHeight:8];
    
    mpsTime = [[NSMutableArray alloc] init];
    metalNaiveTime = [[NSMutableArray alloc] init];
    metalTunedTime = [[NSMutableArray alloc] init];
    
    for (int iter = 1; iter <= 75; iter++) {
        int m = iter * 16;
        int n = m;
        int k = m;
        float alpha = 1;
        float beta = 1;
        bool transA = 0;
        bool transB = 0;
        
        printf("m, n, k, alpha, beta - %d %d %d %f %f\n", m, n, k, alpha, beta);
        
        float *A, *B, *C ,*D;
        int alignment = getpagesize();
        posix_memalign((void**)&A, alignment, sizeof(float) * m * k);
        posix_memalign((void**)&B, alignment, sizeof(float) * k * n);
        posix_memalign((void**)&C, alignment, sizeof(float) * m * n);
        posix_memalign((void**)&D, alignment, sizeof(float) * m * n);
        
        for (int i = 0; i < m * k; i++) {
            A[i] = (float)arc4random() / UINT32_MAX - 0.5;
        }

        for (int i = 0; i < k * n; i++) {
            B[i] = (float)arc4random() / UINT32_MAX - 0.5;
        }

        for (int i = 0; i < m * n; i++) {
            D[i] = 0;
        }
        
        NSDate *startTime;
        
        // MPS
        memcpy(C, D, m * n * sizeof(float));
        startTime = [NSDate date];
        mps_gemm(transA, transB, m, n, k, alpha, A, B, beta, C);
        [mpsTime addObject:@(-[startTime timeIntervalSinceNow]*1000)];
        
        float mpsSum = 0.0;
        vDSP_sve(C, 1, &mpsSum, m * n);
        
        // Metal naive
        memcpy(C, D, m * n * sizeof(float));
        startTime = [NSDate date];
        [metalGemmNaive gemmWithTransA:transA TransB:transB M:m N:n K:k alpha:alpha A:A B:B beta:beta C:C];
        [metalNaiveTime addObject:@(-[startTime timeIntervalSinceNow]*1000)];
        
        float metalNaiveSum = 0.0;
        vDSP_sve(C, 1, &metalNaiveSum, m * n);
        
        // Metal tuned
        memcpy(C, D, m * n * sizeof(float));
        startTime = [NSDate date];
        [metalGemmTunedV2 gemmWithTransA:transA TransB:transB M:m N:n K:k alpha:alpha A:A B:B beta:beta C:C];
        [metalTunedTime addObject:@(-[startTime timeIntervalSinceNow]*1000)];
        
        float metalTunedSum = 0.0;
        vDSP_sve(C, 1, &metalTunedSum, m * n);
        
        if (fabsf(mpsSum - metalNaiveSum) > 0.1 || fabsf(mpsSum - metalTunedSum) > 0.1) {
            printf("Diff too large: %f %f\n", mpsSum - metalNaiveSum, mpsSum - metalTunedSum);
        }
        
        free(A);
        free(B);
        free(C);
        free(D);
    }
    
    printf("MPS:\n");
    for (NSNumber *time in mpsTime) {
        printf("%.0f ", time.floatValue);
    }
    printf("\n");
    
    printf("Metal naive:\n");
    for (NSNumber *time in metalNaiveTime) {
        printf("%.0f ", time.floatValue);
    }
    printf("\n");
    
    printf("Metal tuned:\n");
    for (NSNumber *time in metalTunedTime) {
        printf("%.0f ", time.floatValue);
    }
    printf("\n");
}


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}


@end
