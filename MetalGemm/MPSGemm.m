//
//  MPSGemm.m
//  MetalGemm
//
//  Created by Lun on 17/11/2017.
//  Copyright © 2017 Lun. All rights reserved.
//

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import "MPSGemm.h"

typedef NS_ENUM (NSInteger,MetalMatrixBufferTypes) {
    eMPSMatBufferA = 0,
    eMPSMatBufferB,
    eMPSMatBufferC,
    eMPSMatBufferMax
};

@implementation MPSGemm {
    int _m;
    int _n;
    int _k;
    float _alpha;
    float _beta;
    float *_A;
    float *_B;
    float *_C;
    bool _transA;
    bool _transB;
    
    id<MTLDevice>        _device;
    id<MTLCommandQueue>  _commandQueue;
    id<MTLCommandBuffer> _commandBuffer;
    
    NSMutableArray<MPSMatrix *> *_matrices;
    dispatch_group_t _dispatchGroup;
    dispatch_queue_t _dispatchQueue;
}

static MPSGemm *multiplicationHandler = nil;

+ (MPSGemm *)sharedInstance {
    @synchronized(self) {
        if (!multiplicationHandler)
            multiplicationHandler = [[self alloc] init];
    }
    return multiplicationHandler;
}

+ (id)allocWithZone:(NSZone *)zone {
    @synchronized(self) {
        if (!multiplicationHandler) {
            multiplicationHandler = [super allocWithZone:zone];
            return multiplicationHandler;
        }
    }
    return nil;
}

- (id)init {
    if (self = [super init]) {
        _device = MTLCreateSystemDefaultDevice();
        NSAssert(_device, @">> ERROR: Failed creating a system default device!");
        
        _commandQueue = [_device newCommandQueue];
        NSAssert(_commandQueue, @">> ERROR: Failed creating a command queue!");
        
        _matrices = [[NSMutableArray alloc] initWithCapacity:eMPSMatBufferMax];
        NSAssert(_matrices, @">> ERROR: Failed creating a mutable array for Metal buffers!");
        
        _dispatchGroup = dispatch_group_create();
        NSAssert(_dispatchGroup, @">> ERROR: Failed creating a dispatch group!");
        
        _dispatchQueue = dispatch_queue_create("com.lun.metalgemm.naive", DISPATCH_QUEUE_SERIAL);
        NSAssert(_dispatchQueue, @">> ERROR: Failed creating a dispatch queue!");
    }
    return self;
}

- (void)_multiplyWithTransA:(const bool)transA
                     TransB:(const bool)transB
                          M:(const int)M
                          N:(const int)N
                          K:(const int)K
                      alpha:(const float)alpha
                          A:(const float *)A
                          B:(const float *)B
                       beta:(const float)beta
                          C:(float *)C
                 completion:(void (^)(void))completion {
    @synchronized(self) {
        _m = M;
        _n = N;
        _k = K;
        _alpha = alpha;
        _beta  = beta;
        _A = (float *)A;
        _B = (float *)B;
        _C = C;
        _transA = transA;
        _transB = transB;
        
        _commandBuffer = [_commandQueue commandBuffer];
        NSAssert(_commandBuffer, @">> ERROR: Failed creating a command buffer!");
        
        [self _setMatrices];
        [self _encode];
        [self _compute];
        
        completion();
    }
}

- (void)_setMatrices {
    [self _initMatrixWithData:_A  row:_m column:_k atIndex:eMPSMatBufferA];
    [self _initMatrixWithData:_B  row:_k column:_n atIndex:eMPSMatBufferB];
    [self _initMatrixWithData:_C  row:_m column:_n atIndex:eMPSMatBufferC];
    
    dispatch_group_wait(_dispatchGroup, DISPATCH_TIME_FOREVER);
}

- (void)_initMatrixWithData:(float *)data
                        row:(size_t)row
                     column:(size_t)col
                    atIndex:(NSUInteger)idx {
    dispatch_group_enter(_dispatchGroup);
    dispatch_group_async(_dispatchGroup, _dispatchQueue, ^{
        size_t dataSize = row * col * sizeof(float);
        size_t bufferSize = dataSize % getpagesize()? (dataSize / getpagesize() + 1) * getpagesize(): dataSize;
        id<MTLBuffer> buffer = data ?
        [_device newBufferWithBytesNoCopy:data
                                   length:bufferSize
                                  options:MTLResourceStorageModeShared
                              deallocator:nil]:
        [_device newBufferWithLength:bufferSize
                             options:MTLResourceStorageModeShared];
        NSAssert(buffer, @">> ERROR: Failed creating a buffer!");
        
        MPSMatrixDescriptor *descriptor = [MPSMatrixDescriptor matrixDescriptorWithRows:row
                                                                                columns:col
                                                                               rowBytes:col * sizeof(float)
                                                                               dataType:MPSDataTypeFloat32];
        MPSMatrix *matrix = [[MPSMatrix alloc] initWithBuffer:buffer descriptor:descriptor];
        NSAssert(matrix, @">> ERROR: Failed creating a MPSMatrix!");
        
        _matrices[idx] = matrix;
        
        dispatch_group_leave(_dispatchGroup);
    });
}

- (void)_encode {
    MPSMatrixMultiplication *kernel = [[MPSMatrixMultiplication alloc] initWithDevice:_device
                                                                        transposeLeft:_transA
                                                                       transposeRight:_transB
                                                                           resultRows:_m
                                                                        resultColumns:_n
                                                                      interiorColumns:_k
                                                                                alpha:_alpha
                                                                                 beta:_beta];
    [kernel encodeToCommandBuffer:_commandBuffer
                       leftMatrix:(MPSMatrix *)_matrices[eMPSMatBufferA]
                      rightMatrix:(MPSMatrix *)_matrices[eMPSMatBufferB]
                     resultMatrix:(MPSMatrix *)_matrices[eMPSMatBufferC]];
}

- (void)_compute {
    [_commandBuffer commit];
    [_commandBuffer waitUntilCompleted];
}

void mps_gemm(const bool transA,
              const bool transB,
              const int M,
              const int N,
              const int K,
              const float alpha,
              const float *A,
              const float *B,
              const float beta,
              float *C) {
    dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        [[MPSGemm sharedInstance] _multiplyWithTransA:transA
                                               TransB:transB
                                                    M:M
                                                    N:N
                                                    K:K
                                                alpha:alpha
                                                    A:A
                                                    B:B
                                                 beta:beta
                                                    C:C
                                           completion:^() {
                                               dispatch_semaphore_signal(semaphore);
                                           }];
    });
    dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);
}

@end
