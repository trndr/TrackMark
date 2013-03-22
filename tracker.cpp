#include <cv.h>
#include <highgui.h>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <thread>
#include <mutex>
#include "TagRegion.h"
#include <sys/time.h>

#define FPS
#define showTrack
#define SingleThreadOff


using namespace cv;
using namespace std;


void detectAndTrack();
void updateLoopO(vector<TagRegion>* tags);
void updateLoop();
void drawTags(vector<TagRegion>* tags);
vector<Point2f> keyPoint2Point2f(vector<KeyPoint> keyPoints);
Mat frame, displayFrame, gray, oldGray, oldFrame;


vector <TagRegion> unsortedTags;
mutex tagRegionMutex;
mutex frameMutex;
mutex haarMutex;
mutex grayFrameMutex;

int detThreads =1; //max number of detection threads
CascadeClassifier cascade;
VideoCapture capture;
Size captureSize;
GoodFeaturesToTrackDetector detector(3, 0.1,  19.0);

vector<Mat> imageBuffer;
mutex imageBufferMutex;
vector<TagRegion> topTags;
vector<TagRegion> headTags;
vector<TagRegion> bedTags;
vector<TagRegion> bottomTags;
/* Based on cv::getPerspictiveTransform
 * Calculates coefficients of perspective transformation
 * which maps (xi,yi,zi) to (ui,vi,wi), (i=1,2,3,4):
 *
 *      c00*xi + c01*yi + c02*zi + c03
 * ui = ------------------------------
 *      c30*xi + c31*yi + c32*zi + c33
 *
 * c30*ui*xi + c31*ui*yi + c31*ui*zi + ui = c00*xi + c01*yi + c02*zi + c03
 *
 *      c10*xi + c11*yi + c12*zi + c13
 * vi = ------------------------------
 *      c30*xi + c31*yi + c32*zi + c33
 *
 *
 *      c20*xi + c21*yi + c22*zi + c23
 * wi = ------------------------------
 *      c30*xi + c31*yi + c32*zi + c33
 *
 * ui = c00*xi + c01*yi + c02*zi + c03 - c30*ui*xi - c31*ui*yi - c32*ui*zi
 * vi = c10*xi + c11*yi + c12*zi + c13 - c30*vi*xi - c31*vi*yi - c32*vi*zi
 * wi = c20*xi + c21*yi + c22*zi + c23 - c30*wi*xi - c31*wi*yi - c32*wi*zi
 *
 * Coefficients are calculated by solving linear system:
 * / x0 y0  z0  1  0  0  0  0  0  0  0  0 -x0*u0 -y0*u0 -z0*u0 \ /c00\ /u0\
 * | x1 y1  z1  1  0  0  0  0  0  0  0  0 -x1*u1 -y1*u1 -z1*u1 | |c01| |u1|
 * | x2 y2  z2  1  0  0  0  0  0  0  0  0 -x2*u2 -y2*u2 -z2*u2 | |c02| |u2|
 * | x3 y3  z3  1  0  0  0  0  0  0  0  0 -x3*u3 -y3*u3 -z3*u3 | |c03| |u3|
 * | x4 y4  z4  1  0  0  0  0  0  0  0  0 -x4*u4 -y4*u4 -z4*u4 | |c10| |u4|
 * |  0  0  0  0  x0 y0 z0  1  0  0  0  0 -x0*v0 -y0*v0 -z0*v0 | |c11| |v0|
 * |  0  0  0  0  x1 y1 z1  1  0  0  0  0 -x1*v1 -y1*v1 -z1*v1 | |c12| |v1|
 * |  0  0  0  0  x2 y2 z2  1  0  0  0  0 -x2*v2 -y2*v2 -z2*v2 |.|c13|=|v2|
 * |  0  0  0  0  x3 y3 z3  1  0  0  0  0 -x3*v3 -y3*v3 -z3*v3 | |c20| |v3|
 * |  0  0  0  0  x4 y4 z4  1  0  0  0  0 -x4*v2 -y4*v2 -z4*v2 | |c21| |v4|
 * |  0  0  0  0   0  0  0  0 x0 y0 z0  1 -x0*w0 -y0*w0 -z0*w0 | |c22| |w0|
 * |  0  0  0  0   0  0  0  0 x1 y1 z1  1 -x1*w1 -y1*w1 -z1*w1 | |c23| |w1|
 * |  0  0  0  0   0  0  0  0 x2 y2 z2  1 -x2*w2 -y2*w2 -z2*w2 | |c30| |w2|
 * |  0  0  0  0   0  0  0  0 x3 y3 z3  1 -x3*w3 -y3*w3 -z3*w3 | |c31| |w3|
 * \  0  0  0  0   0  0  0  0 x4 y4 z4  1 -x4*w2 -y4*w2 -z4*w2 / \c32/ \w4/
 * 
 *
 * where:
 *   cij - matrix coefficients, c33 = 1
 */
Mat getPerspectiveTransform3d( const Point3f src[], const Point3f dst[] )
{
    Mat M(4, 4, CV_64F), X(15, 1, CV_64F, M.data);
    double a[15][15], b[15];
    Mat A(15, 15, CV_64F, a), B(15, 1, CV_64F, b);

    for( int i = 0; i < 5; ++i )
    {
        a[i][0] = a[i+5][4] = a[i+10][8] = src[i].x;
        a[i][1] = a[i+5][5] = a[i+10][9] = src[i].y;
        a[i][2] = a[i+5][6] = a[i+10][10] = src[i].z;
        a[i][3] = a[i+5][7] = a[i+10][11] = 1;

        a[i][4] = a[i][5] = a[i][6] = a[i][7] =
        a[i][8] = a[i][9] = a[i][10] = a[i][11] =
        a[i+5][0] = a[i+5][1] = a[i+5][2] = a[i+5][3] =
        a[i+5][8] = a[i+5][9] = a[i+5][10] = a[i+5][11] =
        a[i+10][0] = a[i+10][1] = a[i+10][2] = a[i+10][3] =
        a[i+10][4] = a[i+10][5] = a[i+10][6] = a[i+10][7] = 0;

        a[i][12] = -src[i].x*dst[i].x;
        a[i][13] = -src[i].y*dst[i].x;
        a[i][14] = -src[i].z*dst[i].x;
        a[i+5][12] = -src[i].x*dst[i].y;
        a[i+5][13] = -src[i].y*dst[i].y;
        a[i+5][14] = -src[i].z*dst[i].y;
        a[i+10][12] = -src[i].x*dst[i].z;
        a[i+10][13] = -src[i].y*dst[i].z;
        a[i+10][14] = -src[i].z*dst[i].z;
        
        b[i] = dst[i].x;
        b[i+5] = dst[i].y;
        b[i+10] = dst[i].z;
    }

    solve( A, B, X, DECOMP_LU + DECOMP_NORMAL );
    ((double*)M.data)[15] = 1.;

    return M;
}
int main( int argc, const char** argv ){
  bool isDigit=1;
  for (int i=0; argv[1][i]; i++){
    if(!isdigit(argv[1][i])){
     isDigit=0;
    }
  }
  if (isDigit){
    capture.open(atoi(argv[1]));
  }
  else {
    capture.open(argv[1]);
  }
  captureSize=Size(capture.get(CV_CAP_PROP_FRAME_WIDTH), capture.get(CV_CAP_PROP_FRAME_HEIGHT));
  cascade.load(argv[2]);
  //TODO add possibility to read camera calibration matrix form file, and use
//  Mat cameraMatrix=(Mat_<float>(3,3) << 5.8068001364114616e+02, 0., 3.9861365437608777e+02, 0., 5.8129606194986047e+02, 2.2232152890941003e+02, 0., 0., 1. );
//  Mat distCoeffs=(Mat_<float>(5,1) << 1.1857045354410395e-01, -2.2645377882099613e-01, 1.3848599412617020e-03, -8.6535512545680364e-04, 6.5155217563888132e-02);
  namedWindow("main", CV_WINDOW_NORMAL);
#ifdef FPS
  timeval tickM, tockM;
#endif
  capture >> frame;
  cvtColor(frame, gray, CV_RGB2GRAY);
//    undistort(gray, gray, cameraMatrix, distCoeffs);
  int sorted=0;
  //TODO change so it's read from file
  /* X,Y,Z positions of markers measured from the physical printer
   * All coordinates must have at least 2 different values
   */
  Mat realPossitions=(Mat_<float>(5,3) <<  87,  30, 0,
                                          161,  31, 100,
                                           56, 207, 0,
                                          118, 208, 100,
                                          182, 208, 0);
  while(1){
#ifdef FPS
    gettimeofday(&tickM, NULL);
#endif
    gray.copyTo(oldGray);
    capture >> frame;
    cvtColor(frame, gray, CV_RGB2GRAY);
//    undistort(gray, gray, cameraMatrix, distCoeffs);
#ifdef SingleThread
    imageBuffer.clear();
    imageBuffer.push_back(gray);
    detectAndTrack();
    updateLoop();
#else
    if(!sorted){
      //Search for tags if not all are found and the max number of detect threads aren't in use.
      //TODO generalize to use the number of tags from file
      if (unsortedTags.size()<8){
        if (detThreads>0){
          detThreads--;
          imageBuffer.clear();
          imageBuffer.push_back(gray);
          thread detectThread(detectAndTrack);
          detectThread.detach();
        }
        //Buffer the frame if the max number of detect threads is in use
        else{
          imageBufferMutex.lock();
          imageBuffer.push_back(gray);
          imageBufferMutex.unlock();
        }
        thread updateThread(updateLoop);
        updateThread.join();
      }
      else{
        thread updateThread(updateLoop);
        updateThread.join();
        /* organizes points so it's easy to match them to the equivalent in real space */
        //TODO generalize so it's read from file according to the real position data
        Mat centrePointsToGroup= Mat::zeros(unsortedTags.size(), 1, CV_32F);
        for (unsigned int i=0; i<unsortedTags.size(); i++){
          centrePointsToGroup.at<float>(i)=unsortedTags[i].centre.y;
        }
        Mat groupings;
        Mat centers;
        kmeans(centrePointsToGroup, 4, groupings, TermCriteria(CV_TERMCRIT_EPS, 1000, 0.01), 4, KMEANS_PP_CENTERS, centers );
        Mat centersSorted;
        cv::sort(centers, centersSorted, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
        for (int i = 0; i<4;i++){
          for (unsigned int j =0;j<unsortedTags.size();j++){
            if (groupings.at<int>(0,j)==i){
              if (centers.at<float>(0,i)==centersSorted.at<float>(0,0)){
                topTags.push_back(unsortedTags[j]);
              }
              if (centers.at<float>(0,i)==centersSorted.at<float>(0,1)){
                headTags.push_back(unsortedTags[j]);
              }
              if (centers.at<float>(0,i)==centersSorted.at<float>(0,2)){
                bedTags.push_back(unsortedTags[j]);
              }
              if (centers.at<float>(0,i)==centersSorted.at<float>(0,3)){
                bottomTags.push_back(unsortedTags[j]);
              }
            }
          }
        }
        sort(topTags.begin(), topTags.end());
        sort(bedTags.begin(), bedTags.end());
        sort(bottomTags.begin(), bottomTags.end());
        for (unsigned int i=0;i<topTags.size();i++){
          char nameString[21];
          sprintf(nameString, "top %i", i);
          topTags[i].name=nameString;
        }
        for (unsigned int i=0;i<bedTags.size();i++){
          char nameString[21];
          sprintf(nameString, "bed %i", i);
          bedTags[i].name=nameString;
        }
        for (unsigned int i=0;i<bottomTags.size();i++){
          char nameString[21];
          sprintf(nameString, "bottom %i", i);
          bottomTags[i].name=nameString;
        }
        sort(headTags.begin(), headTags.end());
        for (unsigned int i=0;i<headTags.size();i++){
          char nameString[21];
          sprintf(nameString, "head %i", i);
          headTags[i].name=nameString;
        }
        sorted = 1;
      }
    }
    else{
      thread updateTop(updateLoopO, &topTags);
      thread updateHead(updateLoopO, &headTags);
      thread updateBed(updateLoopO, &bedTags);
      thread updateBottom(updateLoopO, &bottomTags);
      updateTop.join();
      updateHead.join();
      updateBed.join();
      updateBottom.join();
    }
#endif
    if (!sorted){
#ifdef showTrack
      drawTags(&unsortedTags);
#endif
    }
    else{
#ifdef showTrack
      drawTags(&topTags);
      drawTags(&headTags);
      drawTags(&bedTags);
      drawTags(&bottomTags);
#endif
      /* Takes the found tags convert them to a matrix estimate the 
       * transformation between real and image space
       * and calculate the real space position given the estimated transformation
       */
      //TODO generalize so it works for any printer
      vector<Point3f> pointsImage;
      pointsImage.push_back(Point3f(topTags[0].centre.x, topTags[0].centre.y, topTags[0].size));
      pointsImage.push_back(Point3f(topTags[1].centre.x, topTags[1].centre.y, topTags[1].size));
      pointsImage.push_back(Point3f(headTags[0].centre.x, headTags[0].centre.y, headTags[0].size));
      pointsImage.push_back(Point3f(bedTags[0].centre.x, bedTags[0].centre.y, bedTags[0].size));
      pointsImage.push_back(Point3f(bedTags[1].centre.x, bedTags[1].centre.y, bedTags[1].size));
      pointsImage.push_back(Point3f(bottomTags[0].centre.x, bottomTags[0].centre.y, bottomTags[0].size));
      pointsImage.push_back(Point3f(bottomTags[1].centre.x, bottomTags[1].centre.y, bottomTags[1].size));
      pointsImage.push_back(Point3f(bottomTags[2].centre.x, bottomTags[2].centre.y, bottomTags[2].size));
      Mat headFrame, bedFrame, partial, solved, framePossitions;
      Mat matImage(pointsImage, 1);
      Mat matEstimatedReal;
      framePossitions=(Mat_<float>(5,3) << topTags[0].centre.x, topTags[0].centre.y, topTags[0].size,
                                 topTags[1].centre.x, topTags[1].centre.y, topTags[1].size,
                                 bottomTags[0].centre.x, bottomTags[0].centre.y, bottomTags[0].size,
                                 bottomTags[1].centre.x, bottomTags[1].centre.y, bottomTags[1].size,
                                 bottomTags[2].centre.x, bottomTags[2].centre.y, bottomTags[2].size);
      solved = getPerspectiveTransform3d((const Point3f*)realPossitions.data, (const Point3f*)framePossitions.data);

      perspectiveTransform(matImage, matEstimatedReal, solved.inv());
      cout << matEstimatedReal.at<float>(3,0) << " ";
      cout << matEstimatedReal.at<float>(2,1) << " ";
      cout << matEstimatedReal.at<float>(3,2) << " ";
      cout << capture.get(CV_CAP_PROP_POS_FRAMES) << endl;
    }

#ifdef FPS
    gettimeofday(&tockM, NULL);
    std::stringstream sstm;
    sstm << (1000/(((double)tockM.tv_sec-(double)tickM.tv_sec)*1000 + ((double)tockM.tv_usec-(double)tickM.tv_usec)/1000));
    putText(frame, sstm.str(),Point2f(10, 10), CV_FONT_HERSHEY_COMPLEX, .6, Scalar(255, 0, 255), 1, 1 );
#endif
    imshow("main", frame);
    waitKey(10);
  }
}


/* draws the tags taken as argument on the view
 */
void drawTags(vector<TagRegion>* tags){
  for (unsigned int i=0;i <(*tags).size(); i++){
    for (unsigned int j=0; j< (*tags)[i].points.size();j++){
      circle(frame, (*tags)[i].points[j], 3,  CV_RGB(255,0,0), -1);
    }
    std::stringstream sstm;
    sstm << (*tags)[i].name << "(" << (*tags)[i].centre.x << ", " << (*tags)[i].centre.y << ", "<< (*tags)[i].size << ")";
    string result = sstm.str();
    putText(frame, result,Point2f((*tags)[i].ROI.x, (*tags)[i].ROI.y), CV_FONT_HERSHEY_COMPLEX, .6, Scalar(255, 0, 255), 1, 1 );
    rectangle(frame, (*tags)[i].ROI, CV_RGB(255,0,0));
    circle(frame, (*tags)[i].centre, 3, CV_RGB(255, 0, 0), -1);
  }
}
/* updates the tags specified as argument
 */
void updateLoopO(vector<TagRegion>* tags){
  for (unsigned int i=0;i <(*tags).size(); i++){
    (*tags)[i].update(oldGray, gray);
  }
}

/* updates all the tags*/
void updateLoop(){
#ifdef timeLoops
    timeval tick, tock;
    gettimeofday(&tick, NULL);
#endif
    if (unsortedTags.size()>0){
    tagRegionMutex.lock();
      for (unsigned int i=0;i <unsortedTags.size(); i++){
        unsortedTags[i].update(oldGray, gray);
      }
    tagRegionMutex.unlock();
    }
#ifdef timeLoops
    gettimeofday(&tock, NULL);
    cout << "Update = " << ((double)tock.tv_sec-(double)tick.tv_sec)*1000 + ((double)tock.tv_usec-(double)tick.tv_usec)/1000 << endl;
#endif

}

/* Takes a vector of keyPoints and returns a vector of points
 */
vector<Point2f> keyPoint2Point2f(vector<KeyPoint> keyPoints){
  vector<Point2f> points;
  for (vector<KeyPoint>::const_iterator k = keyPoints.begin(); k != keyPoints.end(); k++){
      points.push_back(k->pt);
  }
  return points;
}


/* detects tags fitting to the cascade on the frame
 */
void detectAndTrack(){
#ifdef timeLoops
  timeval tickD, tockD;
  gettimeofday(&tickD, NULL);
#endif
  haarMutex.lock();

  vector<Rect> trackers;
  Mat mask;
  vector< vector<Point2f> > keyPoints;

  trackers.clear();
  tagRegionMutex.lock();
  vector<TagRegion> tmpTagRegion2=unsortedTags;
  tagRegionMutex.unlock();
  vector<TagRegion> tmpTagRegion;
  //Search for places fitting to the cascade
  cascade.detectMultiScale(imageBuffer[0], trackers);
  int numberOfTags=tmpTagRegion2.size();

  for(vector<Rect>::const_iterator r = trackers.begin(); r != trackers.end(); r++){
    int notFoundBefore = 1;
    for (vector<TagRegion>::const_iterator t = tmpTagRegion2.begin(); t != tmpTagRegion2.end(); t++){
      if (abs((*t).ROI.x - (*r).x)<20 && abs((*t).ROI.y - (*r).y) <20){
        notFoundBefore=0;
      }
    }
    if (notFoundBefore){
      //Extract trackable points from the area specified by the cascade
      vector<KeyPoint> keyPoint;
      mask = Mat::zeros(imageBuffer[0].size(), CV_8UC1);
      Mat subsection = imageBuffer[0](*r);
      rectangle(mask, *r, CV_RGB(255,255,255), -1);
      detector.detect(imageBuffer[0], keyPoint, mask);
      std::stringstream sstm;
      sstm << "mask" << numberOfTags++;
      string result = sstm.str();
      tmpTagRegion.push_back(TagRegion(keyPoint2Point2f(keyPoint), *r, result, captureSize));

    }
  }
  //Track the found objects to the end of the buffer, so the found positions reflect the current frame
  for (unsigned int region = 0; region < tmpTagRegion.size(); region++){
    imageBufferMutex.lock();
    for (unsigned int image = 0; image < imageBuffer.size()-1;image++){
      tmpTagRegion[region].update(imageBuffer[image], imageBuffer[image+1]);
    }
    imageBufferMutex.unlock();
    tagRegionMutex.lock();
    unsortedTags.push_back(tmpTagRegion[region]);
    tagRegionMutex.unlock();
  }
#ifdef timeLoops
  gettimeofday(&tockD, NULL);
  cout << "Detect = " << ((double)tockD.tv_sec-(double)tickD.tv_sec)*1000 + ((double)tockD.tv_usec-(double)tickD.tv_usec)/1000 << endl;
#endif
  haarMutex.unlock();
  detThreads++;
}
