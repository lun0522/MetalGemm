//
//  MetalGemm.h
//  MetalGemm
//
//  Created by Lun on 17/11/2017.
//  Copyright © 2017 Lun. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface MetalGemm : NSObject

- (instancetype)init __attribute__((unavailable("use initWithKernel:threadgroupWidth:threadgroupHeight:threadgroupCoveredWidth:threadgroupCoveredHeight instead")));
- (instancetype)initWithKernel:(NSString *)kernelName
              threadgroupWidth:(NSUInteger)threadgroupWidth
             threadgroupHeight:(NSUInteger)threadgroupHeight
       threadgroupCoveredWidth:(NSUInteger)threadgroupCoveredWidth
      threadgroupCoveredHeight:(NSUInteger)threadgroupCoveredHeight
NS_DESIGNATED_INITIALIZER;

- (void)gemmWithTransA:(const bool)transA
                TransB:(const bool)transB
                     M:(const int)M
                     N:(const int)N
                     K:(const int)K
                 alpha:(const float)alpha
                     A:(const float *)A
                     B:(const float *)B
                  beta:(const float)beta
                     C:(float *)C ;

@end
