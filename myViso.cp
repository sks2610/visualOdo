#include"stdafx.h"
#include<iostream>
#include<stdio.h>
#include<vector>
#include<math.h>
#include<stdlib.h>
#include<stdlib.h>
#include"libviso\matcher.h"
#include"libviso\matrix.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>

#define w 1200

using namespace cv;

#define min(a, b) a<b?a:b
#define max(a, b) a>b?a:b

int counter = 1;
enum result { UPDATED, FAILED, CONVERGED };
// focal length, principal point, baseline/camera height, pitch
FLOAT calib_f = 260,calib_cu=320,calib_cv=93,calib_b=3.68/6.45,calib_p=0;
FLOAT *X,*Y,*Z;               // 3d points
FLOAT *J;                     // jacobian
FLOAT *p_observe,*p_predict;  // observed and predicted 2d points
FLOAT param[6];               // parameter set
std::vector<int32_t> inliers; // inlier set
Matrix KF_x; // state vector
Matrix KF_z; // observation vector
Matrix KF_A; // state transition matrix
Matrix KF_H; // observation matrix
Matrix KF_P; // covariance of state vector
Matrix KF_Q; // process noise
Matrix KF_R; // measurement covariance
FLOAT net[6] = {0, 0, 0, 0, 0, 0}, zx = 0.0;
extern double newx, newy;
void parcpy(FLOAT *param_dst,FLOAT *param_src);
std::vector<int32_t> stereoGetInlier(const std::vector<Matcher::p_match> p_matched,const FLOAT *param,FLOAT tau);
void stereoComputePredictionsAndJacobian(const FLOAT *param,const std::vector<int32_t> &active);
void stereoComputeObservations(const std::vector<Matcher::p_match> &p_matched,const std::vector<int32_t> &active);
result stereoUpdateParameters(const std::vector<Matcher::p_match> &p_matched,const std::vector<int32_t> &active,
	FLOAT *param,const FLOAT &step_size,const FLOAT &eps);
std::vector<int32_t> stereoGetRandomSample(const std::vector<Matcher::p_match> &p_matched,const float &min_dist);
bool processStereo (std::vector<Matcher::p_match> p_matched,FLOAT deltaT);
void MyLine( cv::Mat img, cv::Point start, cv::Point end );
void MyFilledCircle( cv::Mat img, cv::Point center );
char mapwindow2[] = "Visual Odometry Map";
cv::Mat map2 = cv::Mat::zeros( w, w, CV_8UC3 );
float x, y, xp = 0, yp = 0, slopep = 0;
inline void MyFilledCircle( Mat img, Point center )
{
  int thickness = -1;
  int lineType = 8;
  center.x+=10;
  center.y+=500;
  circle( img, 
	  center,
	  w/280.0,
	  Scalar( 100, 100, 100 ),
	  thickness, 
	  lineType );
}

inline void MyLine( Mat img, Point start, Point end )
{
  int thickness = 2;
  int lineType = CV_AA;
  start.x+=10;
  start.y+=500;
  end.x+=10;
  end.y+=500;
  line( img, 
	start,
	end,
	Scalar( 0, 255, 255 ),
	thickness,
	lineType );
}

void parcpy(FLOAT *param_dst,FLOAT *param_src) {
  memcpy(param_dst,param_src,6*sizeof(FLOAT));
}

bool processStereo (std::vector<Matcher::p_match> p_matched,FLOAT deltaT) {
  
  // return value
  bool success = true;
  
  // compute minimum distance for RANSAC samples
  float width=0,height=0;
  for (std::vector<Matcher::p_match>::iterator it=p_matched.begin(); it!=p_matched.end(); it++) {
    if (it->u1c>width)  width  = it->u1c;
    if (it->v1c>height) height = it->v1c;
  }
  float min_dist = min(width,height)/3.0;
  
  // allocate dynamic memory
  int32_t N = p_matched.size();
  X         = new FLOAT[N];
  Y         = new FLOAT[N];
  Z         = new FLOAT[N];
  J         = new FLOAT[4*N*6];
  p_predict = new FLOAT[4*N];
  p_observe = new FLOAT[4*N];
  
  // get number of matches
  if (N<6) {
    success = false;
    goto failed;
  }

  // project matches of previous image into 3d
  for (int32_t i=0; i<N; i++) {
    FLOAT d = max(p_matched[i].u1p - p_matched[i].u2p,(float)1.0);
    X[i]    = (p_matched[i].u1p-calib_cu)*calib_b/d;
    Y[i]    = (p_matched[i].v1p-calib_cv)*calib_b/d;
    Z[i]    = calib_f*calib_b/d;
  }

  // loop variables
  FLOAT param_curr[6];
  
  // clear parameter and inlier vectors
  inliers.clear();
  for (int32_t i=0; i<6; i++)
      param[i] = 0;

  //////////////////////////////////
  // initial RANSAC estimate
  for (int32_t k=0;k<100;k++) {

    // draw random sample set
    std::vector<int32_t> active = stereoGetRandomSample(p_matched,min_dist);
    if (active.size()<3) {
      success = false;
      goto failed;
    }

    // clear parameter vector
    for (int32_t i=0; i<6; i++)
      param_curr[i] = 0;

    // perform bundle adjustment
    result result = UPDATED;
    int32_t iter=0;
    while (result==UPDATED) {
      result = stereoUpdateParameters(p_matched,active,param_curr,1,1e-6);
      if (iter++ > 20 || result==CONVERGED)
        break;
    }

    // overwrite best parameters if we have more inliers
    if (result!=FAILED) {
      std::vector<int32_t> inliers_curr = stereoGetInlier(p_matched,param_curr,4);
      if (inliers_curr.size()>inliers.size()) {
        inliers = inliers_curr;
        parcpy(param,param_curr);
      }
    }
  }

  //////////////////////////////////
  // final optimization (refinement)
  if (inliers.size()>=6) {
    int32_t iter=0;
    result result = UPDATED;
    while (result==UPDATED) {     
      result = stereoUpdateParameters(p_matched,inliers,param,1,1e-8);
      if (iter++ > 100 || result==CONVERGED)
        break;
    }
	for(int i = 0; i < 6; i++)
		net[i]+=param[i];
	printf("\nThe odometric readings produced by Visual Odometry are:\n  ");
	printf("roll %10f\nyaw %10f\n pitch %10f\nx %10f\ny %10f\n z %10f\n", param[0],param[1],param[2],param[3],param[4],param[5]);
	//param[3]/=10;
	//param[5]/=10;
	zx+=param[1];
	x = param[5]*cos(zx)*2;
	y = param[5]*sin(zx)*2;
	/*float slope = y/x;
	float beta = 3.14285714/2 - atan(slope) - atan(slopep);
	float r = sqrt(x*x + y*y);
	x= r*sin(beta);
	y = r*cos(beta);*/
	newx = x;
	newy = y;
	MyLine( map2, Point( xp+100.0, yp+100.0 ), Point( xp+x+100.0, yp+y+100.0 ) );
	//if((yp+y)>500||(xp+x>500)){
//		yp=50+50*(counter++);xp=0;
//	}

	MyFilledCircle( map2, Point(xp+100.0+x, yp+y+100.0) );
	imshow( mapwindow2, map2);
	//cvMoveWindow( mapwindow, 0, 200 );
	yp +=y; xp +=x; //slopep = slope;
	waitKey(10);
    // not converged
    if (result!=CONVERGED)
      success = false;
    
  // not enough inliers
  } else {
    success = false;
  }
  
  // jump here if something went wrong
  failed:

  // release dynamic memory
  delete X;
  delete Y;
  delete Z;
  delete J;
  delete p_predict;
  delete p_observe;
  
  // parameter estimate succeeded?
  return success;
}

std::vector<int32_t> stereoGetInlier(const std::vector<Matcher::p_match> p_matched,const FLOAT *param,FLOAT tau) {

  // mark all observations active
  std::vector<int32_t> active;
  for (int32_t i=0; i<(int32_t)p_matched.size(); i++)
    active.push_back(i);

  // extract observations and compute predictions
  stereoComputeObservations(p_matched,active);
  stereoComputePredictionsAndJacobian(param,active);

  // compute inliers
  std::vector<int32_t> inliers;
  for (int32_t i=0; i<(int32_t)p_matched.size(); i++)
    if (pow(p_predict[4*i+0]-p_observe[4*i+0],2)+pow(p_predict[4*i+1]-p_observe[4*i+1],2) +
        pow(p_predict[4*i+2]-p_observe[4*i+2],2)+pow(p_predict[4*i+3]-p_observe[4*i+3],2)<pow(tau,2))
      inliers.push_back(i);
  return inliers;
}

std::vector<int32_t> stereoGetRandomSample(const std::vector<Matcher::p_match> &p_matched,const float &min_dist) {

  // init sample and totalset
  std::vector<int32_t> sample;
  std::vector<int32_t> totalset;
  
  bool success = false;
  int32_t k=0;
  
  // try maximally 100 times to create a sample
  while (!success && ++k<100) {

    // create std::vector containing all indices
    totalset.clear();
    for (int32_t i=0; i<(int32_t)p_matched.size(); i++)
      totalset.push_back(i);

    // add 3 indices to current sample
    sample.clear();
    for (int32_t i=0; i<3; i++) {
      int32_t j = rand()%totalset.size();
      sample.push_back(totalset[j]);
      totalset.erase(totalset.begin()+j);
    }
    
    // check distances
    float du = p_matched[sample[0]].u1c-p_matched[sample[1]].u1c;
    float dv = p_matched[sample[0]].v1c-p_matched[sample[1]].v1c;
    if (sqrt(du*du+dv*dv)>min_dist) {
      float norm = sqrt(du*du+dv*dv);
      float nu   = +dv/norm;
      float nv   = -du/norm;
      float ru   = p_matched[sample[2]].u1c-p_matched[sample[0]].u1c;
      float rv   = p_matched[sample[2]].v1c-p_matched[sample[0]].v1c;
      if (fabs(nu*ru+nv*rv)>min_dist) {
        success = true;
        break;
      }
    }
  }
  
  // return empty sample on failure
  if (!success)
    sample.clear();

  // return sample
  return sample;
}

result stereoUpdateParameters(const std::vector<Matcher::p_match> &p_matched,const std::vector<int32_t> &active,
	FLOAT *param,const FLOAT &step_size,const FLOAT &eps)
{
  
  // we need at least 3 observations
  if (active.size()<3)
    return FAILED;
  
  // extract observations and compute predictions
  stereoComputeObservations(p_matched,active);
  stereoComputePredictionsAndJacobian(param,active);

  // init
  Matrix A(6,6);
  Matrix B(6,1);

  // fill matrices A and B
  for (int32_t m=0; m<6; m++) {
    for (int32_t n=0; n<6; n++) {
      FLOAT a = 0;
      for (int32_t i=0; i<4*(int32_t)active.size(); i++)
        a += J[i*6+m]*J[i*6+n];
      A.val[m][n] = a;
    }
    FLOAT b = 0;
    for (int32_t i=0; i<4*(int32_t)active.size(); i++)
      b += J[i*6+m]*(p_observe[i]-p_predict[i]);
    B.val[m][0] = b;
  }

  // perform elimination
  if (B.solve(A)) {
    bool converged = true;
    for (int32_t m=0; m<6; m++) {
      param[m] += step_size*B.val[m][0];
      if (fabs(B.val[m][0])>eps)
        converged = false;
    }
    if (converged)
      return CONVERGED;
    else
      return UPDATED;
  } else {
    return FAILED;
  }
}

void stereoComputeObservations(const std::vector<Matcher::p_match> &p_matched,const std::vector<int32_t> &active) {

  // set all observations
  for (int32_t i=0; i<(int32_t)active.size(); i++) {
    p_observe[4*i+0] = p_matched[active[i]].u1c; // u1
    p_observe[4*i+1] = p_matched[active[i]].v1c; // v1
    p_observe[4*i+2] = p_matched[active[i]].u2c; // u2
    p_observe[4*i+3] = p_matched[active[i]].v2c; // v2
  }
}

void stereoComputePredictionsAndJacobian(const FLOAT *param,const std::vector<int32_t> &active) {

  // extract motion parameters
  FLOAT rx = param[0]; FLOAT ry = param[1]; FLOAT rz = param[2];
  FLOAT tx = param[3]; FLOAT ty = param[4]; FLOAT tz = param[5];

  // precompute sine/cosine
  FLOAT sx = sin(rx); FLOAT cx = cos(rx); FLOAT sy = sin(ry);
  FLOAT cy = cos(ry); FLOAT sz = sin(rz); FLOAT cz = cos(rz);

  // compute rotation matrix and derivatives
  FLOAT r00    = +cy*cz;          FLOAT r01    = -cy*sz;          FLOAT r02    = +sy;
  FLOAT r10    = +sx*sy*cz+cx*sz; FLOAT r11    = -sx*sy*sz+cx*cz; FLOAT r12    = -sx*cy;
  FLOAT r20    = -cx*sy*cz+sx*sz; FLOAT r21    = +cx*sy*sz+sx*cz; FLOAT r22    = +cx*cy;
  FLOAT rdrx10 = +cx*sy*cz-sx*sz; FLOAT rdrx11 = -cx*sy*sz-sx*sz; FLOAT rdrx12 = -cx*cy;
  FLOAT rdrx20 = +sx*sy*cz+cx*sz; FLOAT rdrx21 = -sx*sy*sz+cx*cz; FLOAT rdrx22 = -sx*cy;
  FLOAT rdry00 = -sy*cz;          FLOAT rdry01 = +sy*sz;          FLOAT rdry02 = +cy;
  FLOAT rdry10 = +sx*cy*cz;       FLOAT rdry11 = -sx*cy*sz;       FLOAT rdry12 = +sx*sy;
  FLOAT rdry20 = -cx*cy*cz;       FLOAT rdry21 = +cx*cy*sz;       FLOAT rdry22 = -cx*sy;
  FLOAT rdrz00 = -cy*sz;          FLOAT rdrz01 = -cy*cz;
  FLOAT rdrz10 = -sx*sy*sz+cx*cz; FLOAT rdrz11 = -sx*sy*cz-cx*sz;
  FLOAT rdrz20 = +cx*sy*sz+sx*cz; FLOAT rdrz21 = +cx*sy*cz-sx*sz;

  // loop variables
  FLOAT X1p,Y1p,Z1p;
  FLOAT X1c,Y1c,Z1c,X2c;
  FLOAT X1cd,Y1cd,Z1cd;

  // for all observations do
  for (int32_t i=0; i<(int32_t)active.size(); i++) {

    // get 3d point in previous coordinate system
    X1p = X[active[i]];
    Y1p = Y[active[i]];
    Z1p = Z[active[i]];

    // compute 3d point in current left coordinate system
    X1c = r00*X1p+r01*Y1p+r02*Z1p+tx;
    Y1c = r10*X1p+r11*Y1p+r12*Z1p+ty;
    Z1c = r20*X1p+r21*Y1p+r22*Z1p+tz;

    // compute 3d point in current right coordinate system
    X2c = X1c-calib_b;

    // for all paramters do
    for (int32_t j=0; j<6; j++) {

      // compute derivatives of 3d point in current
      // left coordinate system wrt. parameter j
      switch (j) {
        case 0: X1cd = 0;
                Y1cd = rdrx10*X1p+rdrx11*Y1p+rdrx12*Z1p;
                Z1cd = rdrx20*X1p+rdrx21*Y1p+rdrx22*Z1p;
                break;
        case 1: X1cd = rdry00*X1p+rdry01*Y1p+rdry02*Z1p;
                Y1cd = rdry10*X1p+rdry11*Y1p+rdry12*Z1p;
                Z1cd = rdry20*X1p+rdry21*Y1p+rdry22*Z1p;
                break;
        case 2: X1cd = rdrz00*X1p+rdrz01*Y1p;
                Y1cd = rdrz10*X1p+rdrz11*Y1p;
                Z1cd = rdrz20*X1p+rdrz21*Y1p;
                break;
        case 3: X1cd = 1; Y1cd = 0; Z1cd = 0; break;
        case 4: X1cd = 0; Y1cd = 1; Z1cd = 0; break;
        case 5: X1cd = 0; Y1cd = 0; Z1cd = 1; break;
      }

      // set jacobian entries (project via K)
      J[(4*i+0)*6+j] = calib_f*(X1cd*Z1c-X1c*Z1cd)/(Z1c*Z1c); // left u'
      J[(4*i+1)*6+j] = calib_f*(Y1cd*Z1c-Y1c*Z1cd)/(Z1c*Z1c); // left v'
      J[(4*i+2)*6+j] = calib_f*(X1cd*Z1c-X2c*Z1cd)/(Z1c*Z1c); // right u'
      J[(4*i+3)*6+j] = calib_f*(Y1cd*Z1c-Y1c*Z1cd)/(Z1c*Z1c); // right v'
    }

    // set prediction (project via K)
    p_predict[4*i+0] = calib_f*X1c/Z1c+calib_cu; // left u
    p_predict[4*i+1] = calib_f*Y1c/Z1c+calib_cv; // left v
    p_predict[4*i+2] = calib_f*X2c/Z1c+calib_cu; // right u
    p_predict[4*i+3] = calib_f*Y1c/Z1c+calib_cv; // right v
  }
}
