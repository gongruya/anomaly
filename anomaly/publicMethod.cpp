//
//  publicMethod.cpp
//  anomaly
//
//  Created by Ruya Gong on 3/25/15.
//  Copyright (c) 2015 Ruya Gong. All rights reserved.
//

#include "publicMethod.h"
#include <stdarg.h>

int detectorArea[9] = {
    0,
    1+2+4+8+16+32+64+128,
    65535-(1<<15),
    65535,
    65535,
    65535-1,
    65535-1-2-4-8-16-32,
    0,
    0
};          //Bit Map, reverse order

int addLog(std::string format, ...) {
    va_list arglist;
    va_start(arglist, format);
    char s[1000];
    vsprintf(s, format.c_str(), arglist);
    printf("%s", s);
    return 0;
}
cv::Mat shuffleRows(const cv::Mat &matrix) {
    std::vector <int> seeds;
    for (int i = 0; i < matrix.rows; ++i)
        seeds.push_back(i);
    cv::theRNG().state = time(0);
    cv::randShuffle(seeds);
    cv::Mat output;
    for (int i = 0; i < matrix.rows; ++i)
        output.push_back(matrix.row(seeds[i]));
    return output;
}
void saveMat(std::string fileName, cv::Mat matrix) {
    ///rows(4B), cols(4B), Data(8B * rows * cols)
    FILE *fp = fopen(fileName.c_str(), "wb");
    fwrite(&matrix.rows, sizeof(int), 1, fp);
    fwrite(&matrix.cols, sizeof(int), 1, fp);
    
    for (int i = 0; i < matrix.rows; ++i)
        for (int j = 0; j < matrix.cols; ++j)
            fwrite(&matrix.at<double>(i, j), sizeof(double), 1, fp);
    fclose(fp);
}
cv::Mat loadMat(std::string fileName) {
    ///rows(4B), cols(4B), Data(8B * rows * cols)
    int r, c;
    FILE *fp = fopen(fileName.c_str(), "rb");
    fread(&r, sizeof(int), 1, fp);
    fread(&c, sizeof(int), 1, fp);
    cv::Mat matrix = cv::Mat::zeros(r, c, CV_64FC1);
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j) {
            double tmp;
            fread(&tmp, sizeof(double), 1, fp);
            matrix.at<double>(i, j) = tmp;
        }
    fclose(fp);
    return matrix;
}