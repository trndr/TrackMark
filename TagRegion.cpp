#include "TagRegion.h"
#include <highgui.h>

TagRegion::TagRegion(vector<Point2f> points, Rect ROI, string name){
  this->name=name;
  this->pointsOld=points;
  this->points=points;
  this->ROI=ROI;
  this->meanCounter=0;
  this->ROI=growRegionOfInterest(ROI, 1.3);
}

TagRegion::TagRegion(vector<Point2f> pointsOld, vector<Point2f> points, Rect ROI){
  this->pointsOld=pointsOld;
  this->points=points;
  this->ROI=ROI;
}

void TagRegion::update(Mat oldGray, Mat gray){
  this->pointsOld=this->points;
//  calcOpticalFlowPyrLK(oldGray, gray, this->pointsOld, this->points, this->opticFlowStatus, this->opticFlowErr);
  /* reduce matrises
   * move points old to be in matrix
   *
   * calcOpicalFlow
   * calcRect
   * move points back
   * move rect
   * calc size*/

  Rect increasedROI=growRegionOfInterest(this->ROI, 5);
  this->ROI.x-=increasedROI.x;
  this->ROI.y-=increasedROI.y;
  Mat thisOldGray=oldGray(increasedROI);
  Mat thisGray=gray(increasedROI);

  this->pointsOld.clear();
  for (vector<Point2f>::const_iterator i = this->points.begin(); i != this->points.end(); i++){
    this->pointsOld.push_back(*i-Point2f(increasedROI.x, increasedROI.y));
  }
  calcOpticalFlowPyrLK(thisOldGray, thisGray, this->pointsOld, this->points, this->opticFlowStatus, this->opticFlowErr);

  Rect nextROI=rectangleFlowDest();
  vector<Point2f> pointsTmp=this->points;
  
  this->points.clear();
  for (vector<Point2f>::const_iterator i = pointsTmp.begin(); i !=pointsTmp.end(); i++){
    this->points.push_back(*i+Point2f(increasedROI.x, increasedROI.y));
  }
  this->ROI=nextROI;
  this->ROI.x+=increasedROI.x;
  this->ROI.y+=increasedROI.y;
  calculateSize(gray);
}

Rect TagRegion::rectangleFlowDest(){
  this->rectangleTransformation = cv::Mat::zeros(2,3,6);
  int count =0;
  for (int j = 0; j < this->points.size(); j=j+3) {
    count++;
    std::vector<cv::Point2f> tempFeaturesCurrent(this->points.begin() + j, this->points.begin() + j+3);
    std::vector<cv::Point2f> tempFeaturesOld(this->pointsOld.begin() + j, this->pointsOld.begin() + j+3);
    this->rectangleTransformation+=getAffineTransform(tempFeaturesOld, tempFeaturesCurrent);
  }
  this->rectangleTransformation /= (count);
  //cout << this->rectangleTransformation << endl;
  vector <Point2f> rectangleThing;
  
  //specify the ROI as a vector represinting a square
  rectangleThing.push_back(Point2f(this->ROI.x, this->ROI.y));
  rectangleThing.push_back(Point2f(this->ROI.x+ this->ROI.width, this->ROI.y));
  rectangleThing.push_back(Point2f(this->ROI.x, this->ROI.y+this->ROI.height));
  rectangleThing.push_back(Point2f(this->ROI.x+ this->ROI.width, this->ROI.y+this->ROI.height));

  vector <Point2f> rectangleThing2;
  //transform the square and remake the ROI from the transformed square
  transform(rectangleThing, rectangleThing, this->rectangleTransformation);
/*  cout << "next" << endl;
  for (int i =0; i<rectangleThing2.size();i++){
    cout << rectangleThing[i]-rectangleThing2[i] << endl;
  }*/
  int rectX=(rectangleThing[0].x+rectangleThing[2].x+1)/2;
  int rectY=(rectangleThing[0].y+rectangleThing[1].y+1)/2;
  int rectWidth=(rectangleThing[1].x+rectangleThing[3].x+1)/2-rectX;
  int rectHeight=(rectangleThing[2].y+rectangleThing[3].y+1)/2-rectY;
  Rect newROI=Rect(rectX, rectY, rectWidth, rectHeight);
  return newROI;
}

Rect TagRegion::growRegionOfInterest(Rect original, double factor){
  //cout << "growing ROI" <<endl;
  int x1 = std::max(0.0,((original.x + (1+(double)original.width)/2)-((1+factor*original.width)/2))); 
  int y1 = std::max(0.0,((original.y + (1+(double)original.height)/2)-((1+factor*original.height)/2)));
  int x2 = std::min(799.0 ,((original.x + (1+(double)original.width)/2)+((1+factor*original.width)/2)));  
  int y2 = std::min(447.0 ,((original.y + (1+(double)original.height)/2)+((1+factor*original.height)/2)));
  return Rect(Point2f(x1, y1), Point2f(x2, y2));
}

void TagRegion::calculateSize(Mat gray){
  Mat threshed;
//  Canny(gray(this->ROI), threshed, 30, 60);
  Mat image;
  /*Mat kern = (Mat_<char>(3,3) <<  0, -1,  0,
      -1,  5, -1,
      0, -1,  0);
  filter2D(gray(this->ROI), image, gray(this->ROI).depth(), kern );*/
  GaussianBlur(gray(this->ROI), image, cv::Size(0, 0), 3);
  addWeighted(gray(this->ROI), 5, image, -2.5, 0, image);
//  threshold(/*gray(this->ROI)*/image, threshed, 75, 255,THRESH_BINARY_INV); 
  adaptiveThreshold(/*gray(this->ROI)*/image, threshed, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 9, 15);
  //imshow(this->name, threshed);
/*  this->meanValues[this->meanCounter]=sum(threshed)[0];
  this->meanCounter++;
  if (this->meanCounter==5){
    this->meanCounter=0;
    this->size=0;
    for (int i = 0; i < 5; ++i){
      this->size+=this->meanValues[i];
    }
    this->size/=5;
  }*/
  this->size=sum(threshed)[0];
}


