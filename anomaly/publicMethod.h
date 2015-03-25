//
//  publicMethod.h
//  anomaly
//
//  Created by Ruya Gong on 3/25/15.
//  Copyright (c) 2015 Ruya Gong. All rights reserved.
//

#ifndef __anomaly__publicMethod__
#define __anomaly__publicMethod__

#include <iostream>
#include <opencv2/opencv.hpp>

#endif /* defined(__anomaly__publicMethod__) */

int addLog(std::string, ...);
cv::Mat shuffleRows(const cv::Mat &matrix);

void saveMat(std::string fileName, cv::Mat matrix);
cv::Mat loadMat(std::string fileName);