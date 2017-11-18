//
//  ViewController.m
//  MetalGemm
//
//  Created by Lun on 17/11/2017.
//  Copyright © 2017 Lun. All rights reserved.
//

#import <Accelerate/Accelerate.h>
#import "MPSGemm.h"
#import "MetalGemmNaive.h"
#import "MetalGemmTunedV1.h"
#import "ViewController.h"

@interface ViewController () {
    NSMutableArray *mpsTime;
    NSMutableArray *metalNaiveTime;
    NSMutableArray *metalTunedV1Time;
}

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
    mpsTime = [[NSMutableArray alloc] init];
    metalNaiveTime = [[NSMutableArray alloc] init];
    metalTunedV1Time = [[NSMutableArray alloc] init];
    
    for (int iter = 2; iter <= 75; iter++) {
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
        metal_gemm_naive(transA, transB, m, n, k, alpha, A, B, beta, C);
        [metalNaiveTime addObject:@(-[startTime timeIntervalSinceNow]*1000)];
        
        float metalNaiveSum = 0.0;
        vDSP_sve(C, 1, &metalNaiveSum, m * n);
        
        // Metal tuned v1
        memcpy(C, D, m * n * sizeof(float));
        startTime = [NSDate date];
        metal_gemm_tuned_v1(transA, transB, m, n, k, alpha, A, B, beta, C);
        [metalTunedV1Time addObject:@(-[startTime timeIntervalSinceNow]*1000)];
        
        float metalTunedV1Sum = 0.0;
        vDSP_sve(C, 1, &metalTunedV1Sum, m * n);
        
        if (fabsf(mpsSum - metalTunedV1Sum) > 0.1) {
            printf("Diff too large: %f\n", mpsSum - metalTunedV1Sum);
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
    
    printf("Metal tuned v1:\n");
    for (NSNumber *time in metalTunedV1Time) {
        printf("%.0f ", time.floatValue);
    }
    printf("\n");
}


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}


@end
