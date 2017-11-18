//
//  MetalGemmTuned.m
//  MetalGemm
//
//  Created by Lun on 17/11/2017.
//  Copyright © 2017 Lun. All rights reserved.
//

#import <Metal/Metal.h>
#import "MetalGemmTuned.h"

typedef NS_ENUM (NSInteger,MetalMatrixBufferTypes) {
    eMTLMatBufferA = 0,
    eMTLMatBufferB,
    eMTLMatBufferC,
    eMTLMatBufferD,
    eMTLMatBufferMax
};

struct MetalMatrixDim {
    bool trans_a, trans_b;
    uint16_t m, n, k, rmd_a, rmd_b;
    float alpha, beta;
};

typedef struct MetalMatrixDim  MetalMatrixDim;
typedef        MetalMatrixDim* MetalMatrixDimRef;

@implementation MetalGemmTuned {
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
    
    id<MTLDevice>                _device;
    id<MTLCommandQueue>          _commandQueue;
    id<MTLCommandBuffer>         _commandBuffer;
    id<MTLComputePipelineState>  _kernel;
    id<MTLComputeCommandEncoder> _encoder;
    
    NSMutableArray<id<MTLBuffer>> *_buffers;
    MTLSize _threadsPerGroup;
    MTLSize _threadgroupsPerGrid;
    dispatch_group_t _dispatchGroup;
    dispatch_queue_t _dispatchQueue;
}

static MetalGemmTuned *multiplicationHandler = nil;

+ (MetalGemmTuned *)sharedInstance {
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
        
        id<MTLLibrary> library = [_device newDefaultLibrary];
        NSAssert(library, @">> ERROR: Failed creating a library!");
        
        id<MTLFunction> func = [library newFunctionWithName:@"MetalGemmTunedV1"];
        NSAssert(func, @">> ERROR: Failed creating a named function!");
        
        _kernel = [_device newComputePipelineStateWithFunction:func error:nil];
        NSAssert(_kernel, @">> ERROR: Failed creating a compute pipeline state!");
        NSLog(@"%lu", _kernel.threadExecutionWidth);
        
        _buffers = [[NSMutableArray alloc] initWithCapacity:eMTLMatBufferMax];
        NSAssert(_buffers, @">> ERROR: Failed creating a mutable array for Metal buffers!");
        
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
        
        _encoder = [_commandBuffer computeCommandEncoder];
        NSAssert(_encoder, @">> ERROR: Failed creating a command encoder!");
        
        [self _setKernel];
        [self _setBuffers];
        [self _setDim];
        [self _encodeBuffers];
        [self _setThreadGroups];
        [self _compute];
        
        completion();
    }
}

- (void)_setKernel {
    [_encoder setComputePipelineState:_kernel];
}

- (void)_setBuffers {
    [self _initBufferWithData:_A  size:_m * _k * sizeof(float) atIndex:eMTLMatBufferA];
    [self _initBufferWithData:_B  size:_k * _n * sizeof(float) atIndex:eMTLMatBufferB];
    [self _initBufferWithData:_C  size:_m * _n * sizeof(float) atIndex:eMTLMatBufferC];
    [self _initBufferWithData:nil size:sizeof(MetalMatrixDim)  atIndex:eMTLMatBufferD];
    
    dispatch_group_wait(_dispatchGroup, DISPATCH_TIME_FOREVER);
}

- (void)_initBufferWithData:(float *)data
                       size:(size_t)size
                    atIndex:(NSUInteger)idx {
    dispatch_group_enter(_dispatchGroup);
    dispatch_group_async(_dispatchGroup, _dispatchQueue, ^{
        size_t bufferSize = size % getpagesize()? (size / getpagesize() + 1) * getpagesize(): size;
        _buffers[idx] = data ?
        [_device newBufferWithBytesNoCopy:data
                                   length:bufferSize
                                  options:MTLResourceStorageModeShared
                              deallocator:nil]:
        [_device newBufferWithLength:bufferSize
                             options:MTLResourceStorageModeShared];
        NSAssert(_buffers[idx], @">> ERROR: Failed creating a buffer!");
        
        dispatch_group_leave(_dispatchGroup);
    });
}

- (void)_setDim {
    MetalMatrixDimRef matrixDim = (MetalMatrixDimRef)_buffers[eMTLMatBufferD].contents;
    
    matrixDim->m = _m;
    matrixDim->n = _n;
    matrixDim->k = _k;
    
    matrixDim->rmd_a = _m % 8;
    matrixDim->rmd_b = _n % 8;
    
    matrixDim->trans_a = _transA;
    matrixDim->trans_b = _transB;
    
    matrixDim->alpha = _alpha;
    matrixDim->beta  = _beta;
}

- (void)_encodeBuffers {
    [_buffers enumerateObjectsUsingBlock:^(id<MTLBuffer> _Nonnull buffer,
                                           NSUInteger idx,
                                           BOOL * _Nonnull stop) {
        dispatch_group_enter(_dispatchGroup);
        dispatch_group_async(_dispatchGroup, _dispatchQueue, ^{
            [_encoder setBuffer:buffer offset:0 atIndex:idx];
            dispatch_group_leave(_dispatchGroup);
        });
    }];
    dispatch_group_wait(_dispatchGroup, DISPATCH_TIME_FOREVER);
}

- (void)_setThreadGroups {
    _threadsPerGroup = MTLSizeMake(8, 8, 1);
    _threadgroupsPerGrid = MTLSizeMake(_m % 8 ? (_m + 8) / 8 : _m / 8,
                                       _n % 8 ? (_n + 8) / 8 : _n / 8,
                                       1);
    
    [_encoder dispatchThreadgroups:_threadgroupsPerGrid
             threadsPerThreadgroup:_threadsPerGroup];
}

- (void)_compute {
    [_encoder endEncoding];
    [_commandBuffer commit];
    [_commandBuffer waitUntilCompleted];
}

void metal_gemm_tuned(const bool transA,
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
        [[MetalGemmTuned sharedInstance] _multiplyWithTransA:transA
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
