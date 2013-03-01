/*
 * TagRegion.h
 *
 *  Created on: 17 Feb 2013
 *      Author: trndr
 */

#include <cv.h>
#ifndef TAGREGION_H_
#define TAGREGION_H_

using namespace std;
using namespace cv;
class TagRegion{
  private:
    int meanCounter;
    Mat opticFlowStatus;
    Mat opticFlowErr;
    Mat rectangleTransformation;

    Rect rectangleFlowDest();
    Rect growRegionOfInterest(Rect original, double factor);
    void calculateSize(Mat gray);
    vector<Point2f> pointsOld;

    int meanValues[5];
    Size matrixSize;

    Point2f centreMass(Mat gray);

  public:
    String name;
    vector<Point2f> points;
    int size;
    Rect ROI;
    Point2f centre;

  //TagRegion(vector<Point2f> pointsOld, vector<Point2f> points, Rect ROI);
  //TagRegion(vector<Point2f> points, Rect ROI);
  TagRegion(vector<Point2f> points, Rect ROI, string name, Size matrixSize);
  void update(Mat oldGray, Mat Gray);


};
bool operator<(const TagRegion &, const TagRegion &);

#endif /* TAGREGION_H_ */
