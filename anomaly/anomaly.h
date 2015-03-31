//
//  anomaly.h
//  anomaly
//
//  Created by Ruya Gong on 3/25/15.
//  Copyright (c) 2015 Ruya Gong. All rights reserved.
//

#ifndef __anomaly__anomaly__
#define __anomaly__anomaly__

#endif /* defined(__anomaly__anomaly__) */
#include <iostream>
#include <opencv2/opencv.hpp>

class anomalyDetection {
public:
    static void train(std::string videoName,  int startFrame, int endFrame, std::string detPath);
    static void demo(std::string videoName, int frameNumber, std::string detPath);
};