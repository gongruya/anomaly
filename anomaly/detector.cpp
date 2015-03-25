//
//  detector.cpp
//  anomaly
//
//  Created by Ruya Gong on 3/25/15.
//  Copyright (c) 2015 Ruya Gong. All rights reserved.
//

#include "detector.h"
#include "publicMethod.h"

///Begin trainParam
trainParam::trainParam() {
    dim = 20;
    pcaDim = 150;
    thr = 0.15;
    maxIter = 100;
    alpha = 3e-2;
    maxRows = 40;
}
///End trainParam


///Begin testParam
testParam::testParam() {
    optThr = 0.12;
    thr = 0.2;
}
///End testParam


///Begin featureParam
featureParam::featureParam() {
    ssr = 5, tsr = 1, depth = 5;
    winH = winW = 10;
    winHNum = 9, winWNum = 16;
    H = 90, W =160;
    motionThr = 5;
    videoSize = cv::Size(W, H);
}
///End featureParam


///Begin frameDiff
frameDiff::frameDiff() {
    count = 0;
    size = 5;
}
void frameDiff::add(cv::Mat M) {
    if (queue.size() == size)
        queue.erase(queue.begin());
    queue.push_back(M.clone());
}
void frameDiff::addDiff(cv::Mat M) {
    cv::Mat diff;
    if (queue.size() > 0)
        cv::absdiff(M, current, diff);
    current = M;
    add(diff);
    ++count;
    diff.release();
}
cv::Mat frameDiff::last() {
    return queue.back();
}
cv::Mat frameDiff::orig() {
    return current;
}
///End frameDiff


///Begin cuboid
void cuboid::feaTrain(frameDiff frames, featureParam param) {
    int depth = param.depth;
    int winH = param.winH;
    int winW = param.winW;
    int H = param.H;
    int W = param.W;
    int ssr = param.ssr;
    int tsr = param.tsr;
    double motionThr = param.motionThr;
    
    if (frames.count % tsr) return;
    
    std::vector<cv::Mat> img;
    for (int k = 0; k < depth; ++k) {
        img.push_back(frames.queue[k]);
    }
    
    extern int detectorArea[9];
    
    for (int i = 0; i < W - winW; i += ssr)
        for (int j = 0; j < H - winH - 20; j += ssr)
            if ( (detectorArea[j/winH] >> (i/winW))&1 ) {
                cv::Rect rect(i, j, winW, winH);
                cv::Mat cube;
                cube = img[0](rect).clone().reshape(0, 1);
                for (int k = 1; k < depth; ++k)
                    cv::hconcat(cube, img[k](rect).clone().reshape(0, 1), cube);
                if (cv::sum(abs(cube))[0] >= motionThr) {
                    normalize(cube, cube);
                    feaMat.push_back(cube.clone());
                }
            }
    feaParam = param;
}
void cuboid::feaTest(frameDiff frames, featureParam param) {
    int depth = param.depth;
    int winH = param.winH;
    int winW = param.winW;
    int winHNum = param.winHNum;
    int winWNum = param.winWNum;
    double motionThr = param.motionThr;
    
    std::vector<cv::Mat> img;
    for (int k = 0; k < depth; ++k) {
        img.push_back(frames.queue[k]);
    }
    
    extern int detectorArea[9];
    
    for (int i = 0; i < winWNum; ++i)
        for (int j = 0; j < winHNum; ++j)
            if ( (detectorArea[j]>>i)&1 ) {
                cv::Rect rect(i * winW, j * winH, winW, winH);
                cv::Mat cube;
                cube = img[0](rect).clone().reshape(0, 1);
                for (int k = 1; k < depth; ++k)
                    cv::hconcat(cube, img[k](rect).clone().reshape(0, 1), cube);
                if (cv::sum(abs(cube))[0] >= motionThr) {
                    normalize(cube, cube);
                    feaMat.push_back(cube.clone());
                    locX.push_back(i);
                    locY.push_back(j);
                }
            }
    feaParam = param;
}
int cuboid::feaRow() {
    return feaMat.rows;
}
///End cuboid


///Begin detector
detector::detector() {

}
detector::detector(std::string fileName) {
    initFromFile(fileName);
}

void detector::train(cuboid feaMat1, trainParam param) {
    cv::Mat feaMat = feaMat1.feaMat;
    int N = feaMat.cols;    //Feature Dimension
    
    int dim = param.dim;
    int maxIter = param.maxIter;
    double alpha = param.alpha;
    double thr = param.thr;
    int maxRows = param.maxRows;
    
    
    feaDim = N;
    sparseDim = dim;    //Prepare for saving parameters
    
    addLog("Features remain: %d\n", feaMat.rows);
    
    int rowsCurrent = std::min(maxRows, feaMat.rows);
    
    while (feaMat.rows > 0) {
        cv::Mat feaMatCurrent = feaMat(cv::Range(0, rowsCurrent), cv::Range::all());
        cv::Mat S = cv::Mat(N, dim, CV_64FC1);
        if (rowsCurrent >= sparseDim) {
            cv::Mat centers, labels, tmp;
            feaMatCurrent.convertTo(tmp, CV_32FC1);
            cv::kmeans(tmp, dim, labels, cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 10, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);
            ((cv::Mat)centers.t()).convertTo(S, CV_64FC1);
        } else {
            cv::theRNG().state = time(0);
            randu(S, cv::Scalar::all(0), cv::Scalar::all(0.1));
        }
        
        std::vector<cv::Mat> beta(feaMatCurrent.rows);
        for (int j = 0; j < beta.size(); ++j) {
            beta[j] = cv::Mat(dim, 1, CV_64FC1);
            randu(beta[j], cv::Scalar::all(0), cv::Scalar::all(0.1));
        }
        
        for (int iter = 1; iter <= maxIter; ++iter) {
            cv::Mat grad = gradientS(S, beta, feaMatCurrent);
            S -= alpha * grad;
            cv::Mat tmp = (S.t() * S).inv(cv::DECOMP_SVD) * S.t();
            for (int j = 0; j < beta.size(); ++j) {
                beta[j] = tmp * feaMatCurrent.row(j).t();
            }
        }
        cv::normalize(S, S);
        cv::Mat RCur = (S * (S.t() * S).inv(cv::DECOMP_SVD) * S.t() - cv::Mat::eye(S.rows, S.rows, CV_64FC1)).t();
        cv::Mat res = feaMat * RCur;
        cv::Mat feaRemain;
        bool works = false;
        for (int j = 0; j < res.rows; ++j) {
            double error = ((cv::Mat)(res.row(j) * res.row(j).t())).at<double>(0,0);
            if (error > thr) {
                feaRemain.push_back(feaMat.row(j));
            } else {
                works = true;
            }
        }
        if (works) {
            feaMat = feaRemain.clone();
            feaRemain.release();
            R.push_back(RCur);
            rowsCurrent = std::min(maxRows, feaMat.rows);
        } else {
            rowsCurrent >>= 1;      //Reducing scale
            feaMat = shuffleRows(feaRemain);
            feaRemain.release();
            addLog("Reducing data scale to: %d\n", rowsCurrent);
        }
        addLog("Features remain: %d\n", feaMat.rows);
        addLog("Total detectors %d\n", R.size());
    }
    addLog("Total detectors %d\n", R.size());
}

int detector::detNum() {
    return (int) R.size();
}
cv::Mat detector::gradientS(cv::Mat S, std::vector<cv::Mat> beta, cv::Mat feaMat) {
    int size = (int) beta.size();
    cv::Mat ans = cv::Mat::zeros(S.rows, S.cols, CV_64FC1);
    for (int j = 0; j < size; ++j) {
        ans += 2 * (S * beta[j] - feaMat.row(j).t()) * beta[j].t();
    }
    return ans;
    
}
void detector::saveToFile(std::string fileName) {
    ///feaDim(4B), sparseDim(4B), size(4B), Data(8B * feaDim * feaDim * number)
    FILE *fp = fopen(fileName.c_str(), "wb");
    fwrite(&feaDim, sizeof(int), 1, fp);
    fwrite(&sparseDim, sizeof(int), 1, fp);
    int size = (int) R.size();
    fwrite(&size, sizeof(int), 1, fp);
    for (int k = 0; k < size; ++k)
        for (int i = 0; i < feaDim; ++i)
            for (int j = 0; j < feaDim; ++j)
                fwrite(&R[k].at<double>(i, j), sizeof(double), 1, fp);
    fclose(fp);
}
void detector::initFromFile(std::string fileName) {
    FILE *fp = fopen(fileName.c_str(), "rb");
    fread(&feaDim, sizeof(int), 1, fp);
    fread(&sparseDim, sizeof(int), 1, fp);
    int size;
    fread(&size, sizeof(int), 1, fp);
    double tmp;
    for (int k = 0; k < size; ++k) {
        R.push_back(cv::Mat::zeros(feaDim, feaDim, CV_64FC1));
        for (int i = 0; i < feaDim; ++i)
            for (int j = 0; j < feaDim; ++j) {
                fread(&tmp, sizeof(double), 1, fp);
                R[k].at<double>(i, j) = tmp;
            }
    }
    fclose(fp);
}
///End detector


///Begin result
cv::Mat result::detect(detector detector1, cuboid feaMat, testParam param) {
    int feaRow = feaMat.feaRow();
    int detNum = detector1.detNum();
    double thr = param.thr;
    anomalyMap = cv::Mat::zeros(feaMat.feaParam.winHNum, feaMat.feaParam.winWNum, CV_64FC1);
    
    for (int i = 0; i < feaRow; ++i) {
        normal.push_back(false);
        for (int j = 0; j < detNum; ++j) {
            cv::Mat error = feaMat.feaMat.row(i) * detector1.R[j];
            error *= error.t();
            double err = error.at<double>(0,0);
            anomalyMap.at<double>(feaMat.locY[i], feaMat.locX[i]) = err;
            if (err < thr) {
                normal[i] = true;
                break;
            }
        }
        if (!normal[i]) {
            locX.push_back(feaMat.locX[i]);
            locY.push_back(feaMat.locY[i]);
        }
    }
    return anomalyMap;
}
///End result