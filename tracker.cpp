#include <cv.h>
#include <highgui.h>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <thread>
#include <mutex>
#include "TagRegion.h"
#include <sys/time.h>

#define FPS
#define timeLoops
#define showTrack

using namespace cv;
using namespace std;


void detetectAndTrack();//VideoCapture& capture, CascadeClassifier& cascade);
void updateLoop();
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
int main( int argc, const char** argv ){
  //capture.open(0);
  capture.open(argv[1]);
  captureSize=Size(capture.get(CV_CAP_PROP_FRAME_WIDTH), capture.get(CV_CAP_PROP_FRAME_HEIGHT));
  cascade.load(argv[2]);
//  capture.set(CV_CAP_PROP_POS_FRAMES, 1580);
//  cout << capture.get(CV_CAP_PROP_POS_FRAMES) << endl;
  namedWindow("main", CV_WINDOW_NORMAL);
#ifdef FPS
  timeval tickM, tockM;
#endif
  capture >> frame;
  cvtColor(frame, gray, CV_RGB2GRAY);
  int play=1;
  while(1){
#ifdef FPS    
    gettimeofday(&tickM, NULL);
#endif
    gray.copyTo(oldGray);
    capture >> frame;
    cvtColor(frame, gray, CV_RGB2GRAY);
    if (thing.size()<8 && detThreads>0){
      detThreads--;
      imageBuffer.clear();
      imageBuffer.push_back(gray);
      thread detectThread(detetectAndTrack);
      detectThread.detach();
    }
    else{
      imageBufferMutex.lock();
      imageBuffer.push_back(gray);
      imageBufferMutex.unlock();
    }
    thread updateThread(updateLoop);
    updateThread.join();
#ifdef showTrack
    for ( int i=0;i <thing.size(); i++){
      for (int j=0; j< thing[i].points.size();j++){
        circle(frame, thing[i].points[j], 3,  CV_RGB(255,0,0), -1);
      }
      std::stringstream sstm;
      sstm << "(" << thing[i].ROI.x << ", " << thing[i].ROI.y << ", " << thing[i].size << ") " << thing[i].name;
      string result = sstm.str();
      putText(frame, result,Point2f(thing[i].ROI.x, thing[i].ROI.y), CV_FONT_HERSHEY_COMPLEX, .6, Scalar(255, 0, 255), 1, 1 ); 
      rectangle(frame, thing[i].ROI, CV_RGB(255,0,0));
    }
#endif
    for (int i=0;i<thing.size();i++){
      circle(frame, thing[i].centre, 3, CV_RGB(255, 0, 0), -1);
    }
#ifdef FPS
    gettimeofday(&tockM, NULL);
    std::stringstream sstm;
    sstm << (1000/(((double)tockM.tv_sec-(double)tickM.tv_sec)*1000 + ((double)tockM.tv_usec-(double)tickM.tv_usec)/1000));
    putText(frame, sstm.str(),Point2f(10, 10), CV_FONT_HERSHEY_COMPLEX, .6, Scalar(255, 0, 255), 1, 1 ); 
#endif
    imshow("main", frame);
    waitKey(1);
  }
}

void displayLoop(){
}

void updateLoop(){
#ifdef timeLoops
    timeval tick, tock;
    gettimeofday(&tick, NULL);
#endif
    if (thing.size()>0){
      tagRegionMutex.lock();
      for ( int i=0;i <thing.size(); i++){
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


void detetectAndTrack()/*VideoCapture& capture, CascadeClassifier& cascade)*/{
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
  for (int region = 0; region < tmpTagRegion.size(); region++){
    imageBufferMutex.lock();
    for (int image = 0; image < imageBuffer.size()-1;image++){
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
