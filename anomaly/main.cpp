//
//  main.cpp
//  anomaly
//
//  Created by Ruya Gong on 3/25/15.
//  Copyright (c) 2015 Ruya Gong. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include "publicMethod.h"
#include "detector.h"
#include "anomaly.h"

using namespace cv;
using namespace std;

int main(int argc, const char * argv[]) {
    
    
    if (0) {        //Training
        string videoName = "/Users/gongruya/Documents/MATLAB/data/huangshan-anomaly/1.mp4";
        string detPath = "/Users/gongruya/Documents/Computer Vision/anomaly/detector/";
        int startFrame = 0;
        int endFrame = 10375;
        anomalyDetection::train(videoName, startFrame, endFrame, detPath);
    } else {        //Demo
        //int startFrame = 17*60*25-250;
        //int startFrame = 140*25;          //2
        //int startFrame = 100*25;          //4
        //int startFrame = (18*60+45)*25;   //2
        //int startFrame = 8*60*25;         //3
        int startFrame = 410*25;          //1
        
        int videoID = 11;
        
        string videoName = "/Users/gongruya/Documents/MATLAB/data/huangshan-anomaly/" + std::to_string(videoID) + ".mp4";
        string detPath = "/Users/gongruya/Documents/Computer Vision/anomaly/detector/";
        
        anomalyDetection::demo(videoName, startFrame, detPath);
    }
    
    return 0;
}
