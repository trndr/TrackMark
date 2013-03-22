#include "TagRegion.h"
#include <highgui.h>
#include <math.h>


TagRegion::TagRegion(vector<Point2f> points, Rect ROI, string name, Size matrixSize){
  this->name=name;
  this->pointsOld=points;
  this->points=points;
  this->ROI=ROI;
  this->meanCounter=0;
  this->matrixSize=matrixSize;
  this->ROI=growRegionOfInterest(ROI, 1.3);
}

void TagRegion::update(Mat oldGray, Mat gray){
  this->pointsOld=this->points;
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
  centreMass(gray(this->ROI));
}

Rect TagRegion::rectangleFlowDest(){
  this->rectangleTransformation = cv::Mat::zeros(2,3,6);
  int count =0;
  for (unsigned int j = 0; j < this->points.size(); j=j+3) {
    count++;
    std::vector<cv::Point2f> tempFeaturesCurrent(this->points.begin() + j, this->points.begin() + j+3);
    std::vector<cv::Point2f> tempFeaturesOld(this->pointsOld.begin() + j, this->pointsOld.begin() + j+3);
    this->rectangleTransformation+=getAffineTransform(tempFeaturesOld, tempFeaturesCurrent);
  }
  this->rectangleTransformation /= (count);
  vector <Point2f> rectangleThing;

  //specify the ROI as a vector represinting a square
  rectangleThing.push_back(Point2f(this->ROI.x, this->ROI.y));
  rectangleThing.push_back(Point2f(this->ROI.x+ this->ROI.width, this->ROI.y));
  rectangleThing.push_back(Point2f(this->ROI.x, this->ROI.y+this->ROI.height));
  rectangleThing.push_back(Point2f(this->ROI.x+ this->ROI.width, this->ROI.y+this->ROI.height));
  rectangleThing.push_back(this->centre);
  vector <Point2f> rectangleThing2;
  //transform the square and remake the ROI from the transformed square
  transform(rectangleThing, rectangleThing, this->rectangleTransformation);
  signed int rectX=(rectangleThing[0].x+rectangleThing[2].x+1)/2;
  signed int rectY=(rectangleThing[0].y+rectangleThing[1].y+1)/2;
  signed int rectWidth=(rectangleThing[1].x+rectangleThing[3].x+1)/2-rectX;
  signed int rectHeight=(rectangleThing[2].y+rectangleThing[3].y+1)/2-rectY;
  Rect newROI=Rect(rectX, rectY, rectWidth, rectHeight);
  if (rectangleThing[4].x>0&&rectangleThing[4].y>0){
    this->centre=rectangleThing[4];
  }
  return newROI;
}

Rect TagRegion::growRegionOfInterest(Rect original, double factor){
  int x1 = std::max(0,(int)((original.x + (1+(double)original.width)/2)-((1+factor*original.width)/2)));
  int y1 = std::max(0,(int)((original.y + (1+(double)original.height)/2)-((1+factor*original.height)/2)));
  int x2 = std::min((this->matrixSize.width-1) ,(int)((original.x + (1+(double)original.width)/2)+((1+factor*original.width)/2)));
  int y2 = std::min((this->matrixSize.height-1) ,(int)((original.y + (1+(double)original.height)/2)+((1+factor*original.height)/2)));
  return Rect(Point2f(x1, y1), Point2f(x2, y2));
}

void TagRegion::centreMass(Mat gray){
  Mat canny;

  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;

  int cannyLow=15;
  //Resize the area so it's 2 the size (this shouldn't help, but it does)
  resize(gray, gray, gray.size()*2);
  Canny(gray, canny, cannyLow, cannyLow*3, 3, true);
  findContours(canny, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
  Mat draw;
  cvtColor(gray, draw, CV_GRAY2RGB);


  Mat mask=Mat::zeros(gray.size(), CV_8UC1);

  vector<Point2f> mc;
  unsigned int area=0;
  unsigned countourCounter;
  

  for( unsigned int i = 0; i < contours.size(); i++ ){
    unsigned int areaOfThis = contourArea(contours[i]);
    if(areaOfThis>(gray.size().width*gray.size().height/3)){
      if (areaOfThis>area){
        countourCounter=i;
        area=areaOfThis;
      }
      Moments moment=moments(contours[i], false);
      mc.push_back(Point2f( (moment.m10/moment.m00) , (moment.m01/moment.m00) ));
      drawContours(mask, contours, i, CV_RGB(255, 255, 255), -1, 8, hierarchy, 0, Point() );
    }
  }
  if (mc.size()>0){
    Point2f mean=accumulate(mc.begin(), mc.end(), Point2f(0.0f,0.0f))*(1.0f/mc.size());
    this->centre=mean*0.5;
    this->centre.x+=this->ROI.x;
    this->centre.y+=this->ROI.y;
    Rect boundingRectangle = boundingRect(contours[countourCounter]);
    this->size=(float)(countNonZero(mask));

    //Move everything back so the resize is mitigated
    this->ROI.x=this->ROI.x+boundingRectangle.x/2;
    this->ROI.y=this->ROI.y+boundingRectangle.y/2;
    this->ROI.width=boundingRectangle.width/2;
    this->ROI.height=boundingRectangle.height/2;
    this->ROI=growRegionOfInterest(this->ROI, 1.4);
  }
  else{
    this->ROI=growRegionOfInterest(this->ROI, 2);
  }
}

void TagRegion::calculateSize(Mat gray){
  Mat threshed;
  Mat image;
  GaussianBlur(gray(this->ROI), image, cv::Size(0, 0), 3);
  addWeighted(gray(this->ROI), 5, image, -2.5, 0, image);
  adaptiveThreshold(image, threshed, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 9, 15);
  this->size=(float)sum(threshed)[0];
}

bool operator<(const TagRegion & a,const TagRegion &other){
  return a.centre.x<other.centre.x;
}
