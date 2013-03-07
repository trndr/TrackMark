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


void detectAndTrack();//VideoCapture& capture, CascadeClassifier& cascade);
void updateLoopO(vector<TagRegion>* tags);
void updateLoop();
void drawTags(vector<TagRegion>* tags);
void displayLoop();
vector<Point2f> keyPoint2Point2f(vector<KeyPoint> keyPoints);
//vector<Point2f> detetectAndFlow(VideoCapture& capture, CascadeClassifier& cascade);
Mat frame, displayFrame, gray, oldGray, oldFrame;


vector <TagRegion> thing;
mutex tagRegionMutex;
mutex frameMutex;
mutex haarMutex;
mutex grayFrameMutex;
int detThreads =1;
CascadeClassifier cascade;
VideoCapture capture;
Size captureSize;
GoodFeaturesToTrackDetector detector(3, 0.1,  15.0);

vector<Mat> imageBuffer;
mutex imageBufferMutex;
vector<TagRegion> topTags;
vector<TagRegion> headTags;
vector<TagRegion> bedTags;
vector<TagRegion> bottomTags;
int main( int argc, const char** argv ){
  //capture.open(0);
  capture.open(argv[1]);
  captureSize=Size(capture.get(CV_CAP_PROP_FRAME_WIDTH), capture.get(CV_CAP_PROP_FRAME_HEIGHT));
  cascade.load(argv[2]);
//  Mat cameraMatrix=(Mat_<float>(3,3) << 5.8068001364114616e+02, 0., 3.9861365437608777e+02, 0., 5.8129606194986047e+02, 2.2232152890941003e+02, 0., 0., 1. );
//  Mat distCoeffs=(Mat_<float>(5,1) << 1.1857045354410395e-01, -2.2645377882099613e-01, 1.3848599412617020e-03, -8.6535512545680364e-04, 6.5155217563888132e-02);
//  capture.set(CV_CAP_PROP_POS_FRAMES, 1580);
  namedWindow("main", CV_WINDOW_NORMAL);
#ifdef FPS
  timeval tickM, tockM;
#endif
  capture >> frame;
//  undistort(frame, frame, cameraMatrix, distCoeffs);
  cvtColor(frame, gray, CV_RGB2GRAY);
  int play=1;
  int sorted=0;
  while(1){
#ifdef FPS
    gettimeofday(&tickM, NULL);
#endif
    gray.copyTo(oldGray);
    capture >> frame;
//    undistort(frame, frame, cameraMatrix, distCoeffs);
    cvtColor(frame, gray, CV_RGB2GRAY);
#ifdef SingleThread
    imageBuffer.clear();
    imageBuffer.push_back(gray);
    detectAndTrack();
    updateLoop();
#else
    if(!sorted){
      if (thing.size()<8){
        if (detThreads>0){
          detThreads--;
          imageBuffer.clear();
          imageBuffer.push_back(gray);
          thread detectThread(detectAndTrack);
          detectThread.detach();
        }
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
        Mat centrePointsToGroup= Mat::zeros(thing.size(), 1, CV_32F);
        for (int i=0; i<thing.size(); i++){
          centrePointsToGroup.at<float>(i)=thing[i].centre.y;
        }
        Mat groupings;
        Mat centers;
        kmeans(centrePointsToGroup, 4, groupings, TermCriteria(CV_TERMCRIT_EPS, 1000, 0.01), 4, KMEANS_PP_CENTERS, centers );
        Mat centersSorted;
        cv::sort(centers, centersSorted, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
        for (int i = 0; i<4;i++){
          for (int j =0;j<thing.size();j++){
            if (groupings.at<int>(0,j)==i){
              if (centers.at<float>(0,i)==centersSorted.at<float>(0,0)){
                topTags.push_back(thing[j]);
              }
              if (centers.at<float>(0,i)==centersSorted.at<float>(0,1)){
                headTags.push_back(thing[j]);
              }
              if (centers.at<float>(0,i)==centersSorted.at<float>(0,2)){
                bedTags.push_back(thing[j]);
              }
              if (centers.at<float>(0,i)==centersSorted.at<float>(0,3)){
                bottomTags.push_back(thing[j]);
              }
            }
          }
        }
        sort(topTags.begin(), topTags.end());
        sort(bedTags.begin(), bedTags.end());
        sort(bottomTags.begin(), bottomTags.end());
        for (int i=0;i<topTags.size();i++){
          char nameString[21];
          sprintf(nameString, "top %i", i);
          topTags[i].name=nameString;
        }
        for (int i=0;i<bedTags.size();i++){
          char nameString[21];
          sprintf(nameString, "bed %i", i);
          bedTags[i].name=nameString;
        }
        for (int i=0;i<bottomTags.size();i++){
          char nameString[21];
          sprintf(nameString, "bottom %i", i);
          bottomTags[i].name=nameString;
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
      drawTags(&thing);
    /*  for (unsigned int i=0;i <thing.size(); i++){
        for (unsigned int j=0; j< thing[i].points.size();j++){
          circle(frame, thing[i].points[j], 3,  CV_RGB(255,0,0), -1);
        }
        std::stringstream sstm;
        sstm << thing[i].name << "(" << thing[i].centre.x << ", " << thing[i].centre.y << ")";//thing[i].ROI.x << ", " << thing[i].ROI.y << ", " << thing[i].size << ") " << thing[i].name;
        string result = sstm.str();
        putText(frame, result,Point2f(thing[i].ROI.x, thing[i].ROI.y), CV_FONT_HERSHEY_COMPLEX, .6, Scalar(255, 0, 255), 1, 1 );
        rectangle(frame, thing[i].ROI, CV_RGB(255,0,0));
      }
      for (unsigned int i=0;i<thing.size();i++){
        circle(frame, thing[i].centre, 3, CV_RGB(255, 0, 0), -1);
      }*/
#endif
    }
    else{
#ifdef showTrack
      drawTags(&topTags);
      drawTags(&headTags);
      drawTags(&bedTags);
      drawTags(&bottomTags);
#endif
      vector<Point3f> pointsImage;
      pointsImage.push_back(Point3f(topTags[0].centre.x, topTags[0].centre.y, topTags[0].size));
      pointsImage.push_back(Point3f(topTags[1].centre.x, topTags[1].centre.y, topTags[1].size));
      pointsImage.push_back(Point3f(bottomTags[0].centre.x, bottomTags[0].centre.y, bottomTags[0].size));
      pointsImage.push_back(Point3f(bottomTags[1].centre.x, bottomTags[1].centre.y, bottomTags[1].size));
      pointsImage.push_back(Point3f(headTags[0].centre.x, headTags[0].centre.y, headTags[0].size));
      pointsImage.push_back(Point3f(bedTags[0].centre.x, bedTags[0].centre.y, bedTags[0].size));
      pointsImage.push_back(Point3f(bedTags[1].centre.x, bedTags[1].centre.y, bedTags[1].size));
      Mat headFrame, bedFrame, realPossitions, partial, solved;
      /* 8.7,3,0
       * 16.1,3.1,0
       * 5.6,20.7,0
       * 11.8,20.8,0
       * 18.2,20.8,0
       * ?,16.4,?
       * ?,16.5,?
       * 11.2,?,9.5
       */
      headFrame=(Mat_<float>(4,4) << topTags[0].centre.x, topTags[0].centre.y, topTags[0].size, 1,
                                    topTags[1].centre.x, topTags[1].centre.y, topTags[1].size, 1,
                                    bottomTags[0].centre.x, bottomTags[0].centre.y, bottomTags[0].size, 1,
                                    headTags[0].centre.x, headTags[0].centre.y, headTags[0].size, 1);
      realPossitions=(Mat_<float>(4,1) << 87 //X possition of the top left marker
                                        , 161 //X possition of the top right marker
                                        , 118  //X possition of the bottom left marker
                                        , 112 //X possition of the head marker
                                        );
      solve(headFrame, realPossitions, partial);
      solved.push_back(partial.reshape(0,1));

      bedFrame=(Mat_<float>(4,4) << topTags[0].centre.x, topTags[0].centre.y, topTags[0].size, 1,
                                    topTags[1].centre.x, topTags[1].centre.y, topTags[1].size, 1,
                                    bottomTags[0].centre.x, bottomTags[0].centre.y, bottomTags[0].size, 1,
                                    bedTags[0].centre.x, bedTags[0].centre.y, bedTags[0].size, 1);
      realPossitions=(Mat_<float>(4,1) << 30 //Y possition of the top left marker
                                        , 31 //Y possition of the top right marker
                                        , 208 //Y possition of the bottom left marker
                                        , 164 //Y possition of the left bed marker
                                        );
      solve(bedFrame, realPossitions, partial);
      solved.push_back(partial.reshape(0,1));
      realPossitions=(Mat_<float>(4,1) << 0 //Z possition of the top left marker
                                        , 0 //Z possition of the top right marker
                                        , 0 //Z possition of the bottom left marker
                                        , 95 //Z possition of the head marker
                                        );
      solve(headFrame, realPossitions, partial);
      solved.push_back(partial.reshape(0,1));
      Mat forthRow = (Mat_<float>(1,4)<<0,0,0,1);
      solved.push_back(forthRow);

      Mat matImage(pointsImage, 1);
      Mat matImage2;
      transform(matImage, matImage2, solved);
      /*if (matImage2.at<float>(5,2) < 160){
        cout << "ERR" << endl;
        cout << matImage << endl;
      }*/

//      cout << "#Frame " << capture.get(CV_CAP_PROP_POS_FRAMES) << endl;
      cout << matImage2.at<float>(5,0) << " ";
      cout << matImage2.at<float>(5,2) << " ";
      cout << matImage2.at<float>(4,1) << " ";
      cout << capture.get(CV_CAP_PROP_POS_FRAMES) << endl;
      /*
      cout << matImage2.at<float>(5,0) << ", ";
      cout << matImage2.at<float>(4,1) << ", ";
      cout << matImage2.at<float>(5,2) << endl;
      */
//      waitKey(5000);
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

void displayLoop(){
}

void drawTags(vector<TagRegion>* tags){
  for (unsigned int i=0;i <(*tags).size(); i++){
    for (unsigned int j=0; j< (*tags)[i].points.size();j++){
      circle(frame, (*tags)[i].points[j], 3,  CV_RGB(255,0,0), -1);
    }
    std::stringstream sstm;
    sstm << (*tags)[i].name << "(" << (*tags)[i].centre.x << ", " << (*tags)[i].centre.y << ")";//thing[i].ROI.x << ", " << thing[i].ROI.y << ", " << thing[i].size << ") " << thing[i].name;
    string result = sstm.str();
    putText(frame, result,Point2f((*tags)[i].ROI.x, (*tags)[i].ROI.y), CV_FONT_HERSHEY_COMPLEX, .6, Scalar(255, 0, 255), 1, 1 );
    rectangle(frame, (*tags)[i].ROI, CV_RGB(255,0,0));
    circle(frame, (*tags)[i].centre, 3, CV_RGB(255, 0, 0), -1);
  }
}
void updateLoopO(vector<TagRegion>* tags){
  for (unsigned int i=0;i <(*tags).size(); i++){
    (*tags)[i].update(oldGray, gray);
  }
}

void updateLoop(){
#ifdef timeLoops
    timeval tick, tock;
    gettimeofday(&tick, NULL);
#endif
    if (thing.size()>0){
    tagRegionMutex.lock();
      for (unsigned int i=0;i <thing.size(); i++){
        thing[i].update(oldGray, gray);
      }
    tagRegionMutex.unlock();
    }
#ifdef timeLoops
    gettimeofday(&tock, NULL);
    cout << "Update = " << ((double)tock.tv_sec-(double)tick.tv_sec)*1000 + ((double)tock.tv_usec-(double)tick.tv_usec)/1000 << endl;
#endif
    //waitKey(10);
  //}

}

vector<Point2f> keyPoint2Point2f(vector<KeyPoint> keyPoints){
  vector<Point2f> points;
  for (vector<KeyPoint>::const_iterator k = keyPoints.begin(); k != keyPoints.end(); k++){
      points.push_back(k->pt);
  }
  return points;
}


void detectAndTrack()/*VideoCapture& capture, CascadeClassifier& cascade)*/{
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
  vector<TagRegion> tmpTagRegion2=thing;
  tagRegionMutex.unlock();
  vector<TagRegion> tmpTagRegion;
  cascade.detectMultiScale(imageBuffer[0], trackers);
  int numberOfTags=tmpTagRegion2.size();

  for(vector<Rect>::const_iterator r = trackers.begin(); r != trackers.end(); r++){
    int add = 1;
    for (vector<TagRegion>::const_iterator t = tmpTagRegion2.begin(); t != tmpTagRegion2.end(); t++){
      if (abs((*t).ROI.x - (*r).x)<20 && abs((*t).ROI.y - (*r).y) <20){
        add=0;
      }
    }
    if (add){
      vector<KeyPoint> keyPoint;
      mask = Mat::zeros(imageBuffer[0].size(), CV_8UC1);
      Mat subsection = imageBuffer[0](*r);
      rectangle(mask, *r, CV_RGB(255,255,255), -1);
      detector.detect(imageBuffer[0], keyPoint, mask);
      std::stringstream sstm;
      sstm << "mask" << numberOfTags++;
      string result = sstm.str();
      tmpTagRegion.push_back(TagRegion(keyPoint2Point2f(keyPoint), *r, result, captureSize));
//      namedWindow(result, CV_WINDOW_NORMAL);

    }
  }
  for (unsigned int region = 0; region < tmpTagRegion.size(); region++){
    imageBufferMutex.lock();
    for (unsigned int image = 0; image < imageBuffer.size()-1;image++){
      tmpTagRegion[region].update(imageBuffer[image], imageBuffer[image+1]);
    }
    imageBufferMutex.unlock();
    tagRegionMutex.lock();
    thing.push_back(tmpTagRegion[region]);
    tagRegionMutex.unlock();
  }
#ifdef timeLoops
  gettimeofday(&tockD, NULL);
  cout << "Detect = " << ((double)tockD.tv_sec-(double)tickD.tv_sec)*1000 + ((double)tockD.tv_usec-(double)tickD.tv_usec)/1000 << endl;
#endif
  haarMutex.unlock();
  detThreads++;
}
