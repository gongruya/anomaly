//
//  detector.h
//  anomaly
//
//  Created by Ruya Gong on 3/25/15.
//  Copyright (c) 2015 Ruya Gong. All rights reserved.
//

#ifndef __anomaly__detector__
#define __anomaly__detector__

#include <iostream>
#include <opencv2/opencv.hpp>

#endif /* defined(__anomaly__detector__) */
class trainParam {
public:
    double thr, alpha;
    int dim, pcaDim, maxIter, maxRows;
    
    trainParam();
};
class testParam {
public:
    double thr, optThr;
    testParam();
};
class featureParam {
public:
    int ssr, tsr, depth;
    int winW, winH, winWNum, winHNum, W, H;
    double motionThr;
    cv::Size videoSize;
    featureParam();
};

class frameDiff {
public:
    int size;
    int count;
    cv::Mat current;
    std::vector<cv::Mat> queue;
    
    frameDiff();
    void add(cv::Mat);
    void addDiff(cv::Mat);
    cv::Mat last();
    cv::Mat orig();
};

class cuboid {
public:
    cv::Mat feaMat;
    std::vector<int> locX, locY;
    featureParam feaParam;
    
    void feaTest(frameDiff, featureParam);
    void feaTrain(frameDiff, featureParam);
    int feaRow();
};

class detector {
public:
    std::vector<cv::Mat> R;
    int sparseDim, feaDim;
    
    detector();
    detector(std::string fileName);
    void train(cuboid, trainParam);
    int detNum();
    void saveToFile(std::string fileName);
    void initFromFile(std::string fileName);
private:
    cv::Mat gradientS(cv::Mat S, std::vector<cv::Mat> beta, cv::Mat feaMat);
};

class result {
public:
    std::vector<int> locX, locY;
    std::vector<bool> normal;
    cv::Mat anomalyMap;
    
    cv::Mat detect(detector, cuboid, testParam);   //Return anomaly map
};