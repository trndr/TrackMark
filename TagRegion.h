#ifndef TAGREGION_H
#define TAGREGION_H
#include <cv.h>
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

  public:
    String name;
    vector<Point2f> points;
    int size;
    Rect ROI;

  //TagRegion(vector<Point2f> pointsOld, vector<Point2f> points, Rect ROI);
  //TagRegion(vector<Point2f> points, Rect ROI);
  TagRegion(vector<Point2f> points, Rect ROI, string name, Size matrixSize);
  void update(Mat oldGray, Mat Gray);
  

};
#endif
