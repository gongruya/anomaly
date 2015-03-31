//
//  anomaly.cpp
//  anomaly
//
//  Created by Ruya Gong on 3/25/15.
//  Copyright (c) 2015 Ruya Gong. All rights reserved.
//

#include "anomaly.h"
#include "publicMethod.h"
#include "detector.h"


void anomalyDetection::train(std::string videoName, int startFrame, int endFrame, std::string detPath) {
    cv::VideoCapture capture(videoName);
    addLog("%s\n", "Initializing the sparse learning system...");
    trainParam tParam;
    featureParam fParam;
    frameDiff frames;
    cuboid features;
    detector detector1;
    
    capture.set(CV_CAP_PROP_POS_FRAMES, startFrame);
    cv::Size videoSize = fParam.videoSize;
    
    addLog("%s\n", "Starting feature extraction...");
    
    for (int i = startFrame; i < endFrame; ++i) {
        cv::Mat gray, I;
        if (!capture.read(I))
            break;
        cv::cvtColor(I, gray, CV_BGR2GRAY);
        cv::resize(gray, gray, videoSize, 0, 0, cv::INTER_CUBIC);
        cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
        gray.convertTo(gray, CV_64FC1);
        gray /= 255;
        
        frames.addDiff(gray);
        if (i-startFrame >= fParam.depth) {
            features.feaTrain(frames, fParam);
        }
        if (!((i-startFrame+1) % 100))
            addLog("%d, ", i);
    }
    addLog("%s", "\n------\n");
    addLog("%s\n", "Feature extraction of the training video is done.");
    addLog("rows:%d\n", features.feaRow());
    
    cv::Mat features1 = features.feaMat.clone();
    cv::PCA pca(features1, cv::Mat(), CV_PCA_DATA_AS_ROW, 150);
    features.feaMat = pca.project(features1);
    
    saveMat(detPath+"PCAeigenvalues.matrix", pca.eigenvalues);
    saveMat(detPath+"PCAeigenvectors.matrix", pca.eigenvectors);
    saveMat(detPath+"PCAmean.matrix", pca.mean);
    saveMat(detPath+"feaPCA.matrix", features.feaMat);
    
    addLog("%s\n", "PCA for the training video is done.");
    
    detector1.train(features, tParam);
    addLog("%s\n", "Sparse learning is done.");
    detector1.saveToFile(detPath+"myDetector.detector");
    addLog("%s\n", "Detector has been saved.");
}


void anomalyDetection::demo(std::string videoName, int frameNumber, std::string detPath) {
    cv::VideoCapture capture(videoName);
    capture.set(CV_CAP_PROP_POS_FRAMES, frameNumber);
    frameDiff frames;
    featureParam fParam;
    testParam tParam;
    
    detector detector1(detPath+"myDetector.detector");
    
    cv::Size videoSize = fParam.videoSize;
    double optThr = tParam.optThr;
    
    cv::PCA pca;
    pca.eigenvalues = loadMat(detPath+"PCAeigenvalues.matrix");
    pca.eigenvectors = loadMat(detPath+"PCAeigenvectors.matrix");
    pca.mean = loadMat(detPath+"PCAmean.matrix");
    //std::cout << pca.eigenvalues << std::endl;
    
    for (int i = 1; ; ++i) {
        cv::Mat I, gray;
        if (!capture.read(I))
            break;
        //cv::imshow("RGB", I);
        cv::cvtColor(I, gray, CV_BGR2GRAY);
        cv::resize(gray, gray, videoSize, 0, 0, cv::INTER_CUBIC);
        cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
        gray.convertTo(gray, CV_64FC1);
        gray /= 255;

        frames.addDiff(gray);
        
        if (i > fParam.depth) {
            cuboid features;
            features.feaTest(frames, fParam);
            if (features.feaRow() > 0) {
                cv::Mat features1 = features.feaMat.clone();
                features.feaMat = pca.project(features.feaMat);
            }
            result result1;
            cv::Mat mask = result1.detect(detector1, features, tParam);
            cv::Mat mask255, withMask;
            
            cv::GaussianBlur(mask, mask, cv::Size(9,5), 0, 0, cv::BORDER_DEFAULT);
            
            mask255 = mask * 255;
            mask255.convertTo(mask255, CV_8UC1);
            cv::threshold(mask255, mask255, optThr * 255, 255, CV_THRESH_BINARY);
            
            cvtColor(mask255, mask255, CV_GRAY2BGR, 3);

            cv::resize(mask255, mask255, cv::Size(I.cols, I.rows), 0, 0, cv::INTER_NEAREST);
            cv::addWeighted(I, 1, mask255, 0.5, 0, withMask);
            
            imshow("Result", withMask);
            mask.release();
            mask255.release();
        }
        if (cv::waitKey(1) == 27)
            break;
    }
    
}