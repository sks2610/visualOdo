#include"stdafx.h"
#include<stdio.h>
#include<opencv2\core\core.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\features2d\features2d.hpp>
#include<opencv2/video/tracking.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/calib3d/calib3d.hpp>
#include"libviso/matcher.h"
#include<time.h>
#include <stdlib.h>
#include <math.h>
#include<string.h>
#include<vector>
#include<fstream>
#define MaxCount 100
//Max number of corners allowed to be detected 
#define MinCount 50
//Corner count should not beless than this 

#define Skip 100
//Maximum clique size

#define MAX_VERTICES 500
//Maximum size of the consistent matrix

#define w 1200

double newx, newy;

bool processStereo (std::vector<Matcher::p_match> p_matched,FLOAT deltaT);
void displayTime(clock_t, clock_t);
void MyLine( cv::Mat img, cv::Point start, cv::Point end );
void MyFilledCircle( cv::Mat img, cv::Point center );
char mapwindow[] = "Visual Odometry Map2";
cv::Mat map = cv::Mat::zeros( w, w, CV_8UC3 );
int Size;


void displayTime(clock_t start1, clock_t start2)
{
  if (difftime(time(NULL), start1) > 2000)
    printf( "%f\n",difftime(time(NULL), start1));
  else
    printf("%f\n\n",((double) (clock() - start2)) / CLOCKS_PER_SEC);
}

int main(int argc, char** argv[])
{
  std::fstream infile, myinsdata;
  infile.open("insdata.txt");
  if (!infile)
	  std::cerr << "Unable to open file insdata.txt";
  int LeftCounter=1,RightCounter=1;
  int Frames=0;
  int Count=0;
  int nCount;
  int Size;
  using namespace cv;
  Mat ImageLeftM,ImageRightM;
  Mat *pyramid,*prev_pyramid, *swap_temp;
  IplImage* ImageLeft=cvLoadImage("I1_000000c.png");
  ImageLeftM=imread("I1_000000c.png");
  CvSize ImageSize = cvGetSize(ImageLeft);
  vector<Point2f> FeaturesLeft, FeaturesLeftPrev, FeaturesRight, FeaturesRightPrev;
  vector<Point2f> FeaturesLeftConsistent, FeaturesLeftConsistentPrev;
  CvPoint2D32f* swap_points;
  Mat gray, prevGray, image;
  cv::Size subPixWinSize(10,10), winSize(31,31);
  TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03);
  /*CvMat* mx1 = cvCreateMat( ImageSize.height,
			    ImageSize.width, CV_32F );
  CvMat* my1 = cvCreateMat( ImageSize.height,
		            ImageSize.width, CV_32F );
  CvMat* mx2 = cvCreateMat( ImageSize.height,
		            ImageSize.width, CV_32F );
  CvMat* my2 = cvCreateMat( ImageSize.height,
		            ImageSize.width, CV_32F );*/
  Mat FundementalMatrix(3,3,CV_32F);
  vector<DMatch> good_matches, good_matchesPrev, matches, matchesPrev;
  vector<KeyPoint> kpLeft, kpRight, kpLeftPrev, kpRightPrev;
  Mat LeftDescriptors, RightDescriptors,LeftDescriptorsPrev,RightDescriptorsPrev;
  Mat ImageRightPrevM, ImageLeftPrevM;
  SurfFeatureDetector detector(400);
  FlannBasedMatcher matcher;
  KeyPoint kp;
  float fx, fy, fz, froll, fyaw, fpitch, fxP, fyP, fzP, frollP, fyawP, fpitchP;
  float delZ=0, delYaw=0, totalYaw=0, xmap=0, ymap=0, xmapP=0, ymapP=0;
  int counter = 1;
  while(1)
    {
	  //an infinite loop
	  clock_t strt1, strt2;
      strt1 = time(NULL);
      strt2 = clock();
      char LeftImageName[14];
      char RightImageName[14];
      int name = sprintf(LeftImageName,"I1_%6dc.png",LeftCounter++);
	  LeftCounter++;
      name = sprintf(RightImageName,"I2_%6dc.png",RightCounter++);
	  RightCounter++;
      for(int k=0;k<name;k++)
	if(LeftImageName[k]==' ')
	  LeftImageName[k]='0';
      for(int k=0;k<name;k++)
	if(RightImageName[k]==' ')
	  RightImageName[k]='0';
      //Making sure that we load images one after the other form the store
      //Mat ImageLeftUnrectifiedM = imread(LeftImageName,0);
      //Mat ImageRightUnrectifiedM = imread(RightImageName,0);
      ImageLeftM = imread(LeftImageName,0);
      ImageRightM = imread(RightImageName,0);
      imshow("Image",ImageLeftM);
	  if(LeftCounter == 7)
		  waitKey(0);
      waitKey(20);
      printf("Loading Time:\t");displayTime(strt1, strt2);
      //Undistorting images
      //Mat Mx1(mx1,1);
      //Mat My1(my1,1);
      //Mat Mx2(mx2,1);
      //Mat My2(my2,1);
      //remap(ImageLeftUnrectifiedM,ImageLeftM,Mx1,My1,CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0, 0));
      //remap(ImageRightUnrectifiedM,ImageRightM,Mx2,My2,CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0, 0));
      Frames++;	
      /********************Feature Detection stage******************************/
      printf("Frames Processed:\t%d\n", LeftCounter);
      if((Frames==1))
	{
	  for(int i = 0; i<5; i++)
		  infile>>fzP;
	  infile>>fyP;
	  infile>>fxP;
	  infile>>frollP;
	  infile>>fpitchP;
	  infile>>fyawP;
	  detector.detect(ImageLeftM, kpLeft);
	  detector.detect(ImageRightM, kpRight);
	  kp.convert(kpLeft, FeaturesLeft, vector<int>());
	  kp.convert(kpRight, FeaturesRight, vector<int>());
	  SurfDescriptorExtractor extractor;
	  extractor.compute(ImageLeftM, kpLeft, LeftDescriptors);
	  extractor.compute(ImageRightM, kpRight, RightDescriptors);
	  matcher.match(LeftDescriptors, RightDescriptors, matches);
	  double max_dist = 0; double min_dist = 100;
	  //Quick calculation of max and min distances between keypoints
	  for( int i = 0; i < LeftDescriptors.rows; i++ )
	    {
	      double dist = matches[i].distance;
	      if( dist < min_dist && dist!=0) min_dist = dist;
	      if( dist > max_dist ) max_dist = dist;
	    }
	  
	  printf("Current-- Max dist : %f \n", max_dist );
	  printf("Current-- Min dist : %f \n", min_dist );
	  
	  //Draw only "good" matches (i.e. whose distance is less than 2*min_dist )
	  //PS.- radiusMatch can also be used here.
	  for( int i = 0; i < LeftDescriptors.rows; i++ )
	    if( matches[i].distance < 4*min_dist )
	      good_matches.push_back( matches[i]);
	  
	  //Draw only "good" matches
	  Mat img_matches;
	  namedWindow("Good_Matches", 1);
	  drawMatches( ImageLeftM, kpLeft, ImageRightM, kpRight,good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		       vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	  //-- Show detected matches
	  imshow( "Good_Matches", img_matches );
	  kp.convert(kpLeft, FeaturesLeft, vector<int>());
	  kp.convert(kpRight, FeaturesRight, vector<int>());
	  vector<Point2f> templeft, tempright;
	  templeft.resize(FeaturesLeft.size());
	  tempright.resize(FeaturesRight.size());
	  std::copy(FeaturesLeft.begin(),FeaturesLeft.end(),templeft.begin());
	  std::copy(FeaturesRight.begin(),FeaturesRight.end(),tempright.begin());
	  for(int resize=0;resize<good_matches.size();resize++){
	    int query = good_matches[resize].queryIdx;
	    int train = good_matches[resize].trainIdx;
	    FeaturesLeft[resize] = templeft[query];
	    FeaturesRight[resize] = tempright[train];
	  }
	  FeaturesLeft.resize(good_matches.size());
	  FeaturesRight.resize(good_matches.size());
	  waitKey(20);
	}
      
      if(Frames>1)
	{
		for(int i = 0; i<5; i++)
		  infile>>fz;
	  infile>>fx;
	  infile>>fy;
	  infile>>froll;
	  infile>>fpitch;
	  infile>>fyaw;
	  //printf("Roll:%10f\nYaw:%10f\nPitch:%10f\nX:%10f\nY:%10f\nZ:%10f\n",froll-frollP,fyaw-fyawP,fpitch-fpitchP,fx-fxP, fy-fyP,fz-fzP);
	  delZ=fz-fzP;
	  delYaw = fyaw - fyawP;
	  totalYaw+=delYaw;
	  fyawP = fyaw;
	  frollP = froll;
	  fpitchP = fpitch;
	  fxP = fx;
	  fyP = fy;
	  fzP = fz;
	  xmap = -1*delZ*cos(totalYaw)*2;
	  ymap = delZ*sin(totalYaw)*2;
	  /*if((ymapP+ymap)>1200||(xmapP+xmap>1200||ymapP+ymap<=0||xmapP+xmap<=0)){
		  ymapP=50*(counter++);xmapP=0;}*/
	  //MyLine( map, Point( xmapP, ymapP ), Point( xmapP+xmap, ymapP+ymap) );
	  //MyFilledCircle( map, Point(xmapP+xmap, ymapP+ymap) );
	  //imshow( mapwindow, map);
  	  //cvMoveWindow( mapwindow, 0, 200 );
	  ymapP+=ymap; xmapP+=xmap; //slopep = slope;
	  //waitKey(10);
    
	  //if frames is greater than 1 that means we have more than one image in stack. So...
	  
	  /*************************Tracking them down the frames****************************/
	  int flags=0;
	  vector<uchar> statusleft, statusright;
	  vector<float> err;
	  //waitKey();
	  int CornerCount=Count;
	  calcOpticalFlowPyrLK(ImageLeftPrevM, ImageLeftM, FeaturesLeftPrev, FeaturesLeft, statusleft, err, winSize, 3, termcrit, 0, 0, 0.001);
	  calcOpticalFlowPyrLK(ImageRightPrevM, ImageRightM, FeaturesRightPrev, FeaturesRight, statusright, err, winSize, 3, termcrit, 0, 0, 0.001);
	  vector<Point2f> templeft, tempright,templeftprev,temprightprev;
	  tempright.resize(FeaturesRight.size());
	  templeft.resize(FeaturesLeft.size());
	  temprightprev.resize(FeaturesRight.size());
	  templeftprev.resize(FeaturesLeft.size());
	  std::copy(FeaturesRight.begin(),FeaturesRight.end(),tempright.begin());
	  std::copy(FeaturesLeft.begin(),FeaturesLeft.end(),templeft.begin());
	  std::copy(FeaturesRightPrev.begin(),FeaturesRightPrev.end(),temprightprev.begin());
	  std::copy(FeaturesLeftPrev.begin(),FeaturesLeftPrev.end(),templeftprev.begin());
	  
	  int k=0;
	  int minSize = FeaturesLeft.size()>FeaturesRight.size()?FeaturesRight.size():FeaturesLeft.size();
	  for(int resize=0;resize<minSize;resize++)
	    if((statusleft[resize]==1)&&(statusright[resize]==1))
	      {
		FeaturesRight[k] = tempright[resize];
		FeaturesLeft[k] = templeft[resize];
		FeaturesLeftPrev[k] = templeftprev[resize];
		FeaturesRightPrev[k] = temprightprev[resize];
		k++;
	      }
	  
	  FeaturesRight.resize(k);
	  FeaturesLeft.resize(k);
	  FeaturesRightPrev.resize(k);
	  FeaturesLeftPrev.resize(k);
	  cornerSubPix(ImageRightM, FeaturesRight,subPixWinSize, cv::Size(-1,-1),termcrit);
	  cornerSubPix(ImageLeftM, FeaturesLeft,subPixWinSize, cv::Size(-1,-1),termcrit);
	  kp.convert(FeaturesLeft, kpLeft);
	  kp.convert(FeaturesRight, kpRight);
	
	  if((Frames!=2)&&((FeaturesLeft.size()<MinCount)||(FeaturesRight.size()<MinCount)))
	    {
	      matches.clear();
	      good_matches.clear();
	      detector.detect(ImageLeftM, kpLeft);
	      detector.detect(ImageRightM, kpRight);
	      kp.convert(kpLeft, FeaturesLeft, vector<int>());
	      kp.convert(kpRight, FeaturesRight, vector<int>());
	      SurfDescriptorExtractor extractor;
	      extractor.compute(ImageLeftM, kpLeft, LeftDescriptors);
	      extractor.compute(ImageRightM, kpRight, RightDescriptors);
	      matcher.match(LeftDescriptors, RightDescriptors, matches);
	      double max_dist = 0; double min_dist = 100;
	      //Quick calculation of max and min distances between keypoints
	      for( int i = 0; i < LeftDescriptors.rows; i++ )
		{
		  double dist = matches[i].distance;
		  if( dist < min_dist && dist!=0) min_dist = dist;
		  if( dist > max_dist ) max_dist = dist;
		}
	      
	      printf("Current-- Max dist : %f \n", max_dist );
	      printf("Current-- Min dist : %f \n", min_dist );
	      
	      // Draw only "good" matches (i.e. whose distance is less than 2*min_dist )
	      // PS.- radiusMatch can also be used here.
	      for( int i = 0; i < LeftDescriptors.rows; i++ )
		if( matches[i].distance < 4*min_dist )
		  good_matches.push_back( matches[i]);
	      
	      //Draw only "good" matches
	      Mat img_matches;
	      namedWindow("Good_Matches", 1);
	      drawMatches( ImageLeftM, kpLeft, ImageRightM, kpRight,good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
			   vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	      //-- Show detected matches
	      imshow( "Good_Matches", img_matches );
	      kp.convert(kpLeft, FeaturesLeft, vector<int>());
	      kp.convert(kpRight, FeaturesRight, vector<int>());
	      vector<Point2f> templeft, tempright;
	      templeft.resize(FeaturesLeft.size());
	      tempright.resize(FeaturesRight.size());
	      std::copy(FeaturesLeft.begin(),FeaturesLeft.end(),templeft.begin());
	      std::copy(FeaturesRight.begin(),FeaturesRight.end(),tempright.begin());
	      for(int resize=0;resize<good_matches.size();resize++)
		{
		  int query = good_matches[resize].queryIdx;
		  int train = good_matches[resize].trainIdx;
		  FeaturesLeft[resize] = templeft[query];
		  FeaturesRight[resize] = tempright[train];
		}
	      FeaturesLeft.resize(good_matches.size());
	      FeaturesRight.resize(good_matches.size());
	      waitKey(20);
	      swap(ImageRightM,ImageRightPrevM);
	      swap(ImageLeftM,ImageLeftPrevM);
	      swap(LeftDescriptors, LeftDescriptorsPrev);
	      swap(RightDescriptors, RightDescriptorsPrev);
	      std::swap(matches, matchesPrev);
	      std::swap(good_matches, good_matchesPrev);
	      std::swap(kpLeft, kpLeftPrev);
	      std::swap(kpRight, kpRightPrev);
	      std::swap( FeaturesLeft,FeaturesLeftPrev);
	      std::swap( FeaturesRight,FeaturesRightPrev);
	      continue;
	    }
	  vector<Matcher::p_match> myMatches(FeaturesLeft.size());
	  for(int i = 0; i < FeaturesLeft.size(); i++)
	    {
	      Matcher::p_match temp(FeaturesLeft[i].x, FeaturesLeft[i].y, i, FeaturesRight[i].x, FeaturesRight[i].y, i,
				    FeaturesLeftPrev[i].x, FeaturesLeftPrev[i].y, i, FeaturesRightPrev[i].x, FeaturesRightPrev[i].y, i);
	      myMatches[i] = temp;
	    }
	  printf("*************The next frame***************");
	  printf("RESULT: Success :%d", processStereo(myMatches, 0.4)); 
	  myinsdata.open("mydata.txt", std::ios::app);
	  if(!myinsdata)
		 std::cerr<<"Unable to Open file mydata.txt\n";
      myinsdata<<LeftCounter-2<<"\t"<<newx<<"\t"<<newy<<"\n";
	  myinsdata.close();	
	}      
      printf("Total Time:\t");displayTime(strt1, strt2);						
      swap(ImageRightM,ImageRightPrevM);
      swap(ImageLeftM,ImageLeftPrevM);
      swap(LeftDescriptors, LeftDescriptorsPrev);
      swap(RightDescriptors, RightDescriptorsPrev);
      std::swap(matches, matchesPrev);
      std::swap(good_matches, good_matchesPrev);
      std::swap(kpLeft, kpLeftPrev);
      std::swap(kpRight, kpRightPrev);
      std::swap( FeaturesLeft,FeaturesLeftPrev);
      std::swap( FeaturesRight,FeaturesRightPrev);
    }	
}
