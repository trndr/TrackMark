#include "TagRegion.h"
#include <highgui.h>

TagRegion::TagRegion(vector<Point2f> points, Rect ROI, string name, Size matrixSize){
  this->name=name;
  this->pointsOld=points;
  this->points=points;
  this->ROI=ROI;
  this->meanCounter=0;
  this->matrixSize=matrixSize;
  this->ROI=growRegionOfInterest(ROI, 1.3);
}
/*
TagRegion::TagRegion(vector<Point2f> pointsOld, vector<Point2f> points, Rect ROI){
  this->pointsOld=pointsOld;
  this->points=points;
  this->ROI=ROI;
}
*/
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
  //calculateSize(gray);
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
  int x2 = std::min((this->matrixSize.width-1) ,(int)((original.x + (1+(double)original.width)/2)+((1+factor*original.width)/2)));
  int y2 = std::min((this->matrixSize.height-1) ,(int)((original.y + (1+(double)original.height)/2)+((1+factor*original.height)/2)));
  return Rect(Point2f(x1, y1), Point2f(x2, y2));
}

Point2f TagRegion::centreMass(Mat gray){
  Mat canny;

  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;

  int cannyLow=15;
  resize(gray, gray, gray.size()*2);
  Canny(gray, canny, cannyLow, cannyLow*3, 3, true);
  findContours(canny, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
  Mat draw;
  cvtColor(gray, draw, CV_GRAY2RGB);


  Mat mask=Mat::zeros(gray.size(), CV_8UC1);

  cout << this->name << endl;
//  cout << gray.size().width*gray.size().height << endl;
  vector<Point2f> mc;
  unsigned int area=0;
  unsigned countourCounter;

  for( unsigned int i = 0; i < contours.size(); i++ ){
    unsigned int areaOfThis = contourArea(contours[i]);
    cout << areaOfThis << endl;
    if(areaOfThis>400){
      if (areaOfThis>area){
        countourCounter=i;
      }
      Moments moment=moments(contours[i], false);
      mc.push_back(Point2f( (moment.m10/moment.m00) , (moment.m01/moment.m00) ));
      //drawContours(mask, contours, i, CV_RGB(255, 255, 255), -1, 8, hierarchy, 0, Point() );
      drawContours(draw, contours, i, CV_RGB(0, 255, 0), 1, 8, hierarchy, 0, Point() );
    }
  }
  Mat tmp;
  draw.copyTo(tmp, mask);
  imshow(this->name, draw);
//  if(mc.size()<1){
    waitKey(50);
//  }
  
  
/*  for (unsigned int i =0;i<interestingContors.size();i++){
    Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
    cout << contourArea(interestingContors[i])<<endl;
    cout << color << endl;
    drawContours(draw, interestingContors, i, color, 1, 8, hierarchy, 0, Point() );*/
//    circle(draw, tmp, 1,  CV_RGB(255,0,0), -1);
//    cout << mc[i] << endl;
  //}
//  Mat element = getStructuringElement( MORPH_RECT,Size( 3, 3 ), Point( 1, 1 ) );


  ///  Get the mass centers:
 // vector<Point2f> mcR( contours.size() );
  /*for( unsigned int i = 0; i < mu.size(); i++ ){
    Point2f tmp = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 );
    mc.push_back(tmp);
  }*/
  Point2f mean=accumulate(mc.begin(), mc.end(), Point2f(0.0f,0.0f))*(1.0f/mc.size());
//  Mat tmp;
  cout << mean <<endl;
  //circle(draw, mean, 3,  CV_RGB(255,0,0), -1);
  this->centre=mean*0.5;
  this->centre.x+=this->ROI.x;
  this->centre.y+=this->ROI.y;
  Rect boundingRectangle = boundingRect(contours[countourCounter]);
  //rectangle(draw, growRegionOfInterest(boundingRectangle, 1.35), CV_RGB(0,255,0), 1);
  this->ROI.x=this->ROI.x+boundingRectangle.x/2;
  this->ROI.y=this->ROI.y+boundingRectangle.y/2;
  this->ROI.width=boundingRectangle.width/2;
  this->ROI.height=boundingRectangle.height/2;
  this->ROI=growRegionOfInterest(this->ROI, 1.4);

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

