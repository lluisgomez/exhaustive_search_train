#define _MAIN

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include <math.h>       /* atan2 */

#define PI 3.14159265

using namespace std;
using namespace cv;

bool sortRects (Rect i,Rect j) { return (i.x<j.x); }
bool sortContours (vector<Point> i, vector<Point> j) { return (boundingRect(i).x<boundingRect(j).x); }

// Fit line from two points
// out a0 is the intercept
// out a1 is the slope
void fitLine(Point p1, Point p2, float &a0, float &a1)
{
  CV_Assert ( p1.x != p2.x );

  a1 = (float)(p2.y - p1.y) / (p2.x - p1.x);
  a0 = a1 * -1 * p1.x + p1.y;
}

// Fit line from three points using Ordinary Least Squares
// out a0 is the intercept
// out a1 is the slope
void fitLineOLS(Point p1, Point p2, Point p3, float &a0, float &a1)
{
  float sumx  = p1.x + p2.x + p3.x;
  float sumy  = p1.y + p2.y + p3.y;
  float sumxy = p1.x*p1.y + p2.x*p2.y + p3.x*p3.y;
  float sumx2 = p1.x*p1.x + p2.x*p2.x + p3.x*p3.x;

  // line coefficients
  a0=(float)(sumy*sumx2-sumx*sumxy) / (3*sumx2-sumx*sumx);
  a1=(float)(3*sumxy-sumx*sumy) / (3*sumx2-sumx*sumx);
}

// Fit line from three points using (heutistic) Least-Median of Squares
// out a0 is the intercept
// out a1 is the slope
float fitLineLMS(Point p1, Point p2, Point p3, float &a0, float &a1)
{

  //Least-Median of Squares does not make sense with only three points
  //becuse any line passing by two of them has median_error = 0
  //So we'll take the one with smaller slope
  float l_a0, l_a1, best_slope, err;

  fitLine(p1,p2,a0,a1);
  best_slope = abs(a1);
  err = (p3.y - (a0+a1*p3.x));

  fitLine(p1,p3,l_a0,l_a1);
  if (abs(l_a1) < best_slope)
  {
    best_slope = abs(l_a1);
    a0 = l_a0;
    a1 = l_a1;
    err = (p2.y - (a0+a1*p2.x));
  }

  fitLine(p2,p3,l_a0,l_a1);
  if (abs(l_a1) < best_slope)
  {
    best_slope = abs(l_a1);
    a0 = l_a0;
    a1 = l_a1;
    err = (p1.y - (a0+a1*p1.x));
  }
  return err;
}

struct line_estimates
{
  float top1_a0;
  float top1_a1;
  float top2_a0;
  float top2_a1;
  float bottom1_a0;
  float bottom1_a1;
  float bottom2_a0;
  float bottom2_a1;
  int x_min;
  int x_max;
  int h_max;
};

void fitLineEstimates(vector< vector<Point> > &regions, Vec3i triplet, line_estimates &estimates)
{
  vector<Rect> char_boxes;
  char_boxes.push_back(boundingRect(regions[triplet[0]]));
  char_boxes.push_back(boundingRect(regions[triplet[1]]));
  char_boxes.push_back(boundingRect(regions[triplet[2]]));

  estimates.x_min = min(min(char_boxes[0].tl().x,char_boxes[1].tl().x), char_boxes[2].tl().x);
  estimates.x_max = max(max(char_boxes[0].br().x,char_boxes[1].br().x), char_boxes[2].br().x);
  estimates.h_max = max(max(char_boxes[0].height,char_boxes[1].height), char_boxes[2].height);

  // Fit one bottom line
  float err = fitLineLMS(char_boxes[0].br(), char_boxes[1].br(), char_boxes[2].br(), estimates.bottom1_a0, estimates.bottom1_a1);

  // Slope for all lines is the same
  estimates.bottom2_a1 = estimates.bottom1_a1;
  estimates.top1_a1    = estimates.bottom1_a1;
  estimates.top2_a1    = estimates.bottom1_a1;

  if (abs(err) > (float)estimates.h_max/6)
  {
    // We need two different bottom lines
    estimates.bottom2_a0 = estimates.bottom1_a0 + err;
  }
  else 
  {
    // Second bottom line is the same
    estimates.bottom2_a0 = estimates.bottom1_a0;
  }

  // Fit one top line within the two (Y)-closer coordinates
  int d_12 = abs(char_boxes[0].tl().y - char_boxes[1].tl().y);
  int d_13 = abs(char_boxes[0].tl().y - char_boxes[2].tl().y);
  int d_23 = abs(char_boxes[1].tl().y - char_boxes[2].tl().y);
  if ((d_12<d_13) && (d_12<d_23))
  {
    Point p = Point((char_boxes[0].tl().x + char_boxes[1].tl().x)/2, 
                    (char_boxes[0].tl().y + char_boxes[1].tl().y)/2);
    estimates.top1_a0 = estimates.bottom1_a0 + (p.y - (estimates.bottom1_a0+p.x*estimates.bottom1_a1));
    p = char_boxes[2].tl();
    err = (p.y - (estimates.top1_a0+p.x*estimates.top1_a1));
  }
  else if (d_13<d_23)
  {
    Point p = Point((char_boxes[0].tl().x + char_boxes[2].tl().x)/2, 
                    (char_boxes[0].tl().y + char_boxes[2].tl().y)/2);
    estimates.top1_a0 = estimates.bottom1_a0 + (p.y - (estimates.bottom1_a0+p.x*estimates.bottom1_a1));
    p = char_boxes[1].tl();
    err = (p.y - (estimates.top1_a0+p.x*estimates.top1_a1));
  }
  else
  {
    Point p = Point((char_boxes[1].tl().x + char_boxes[2].tl().x)/2, 
                    (char_boxes[1].tl().y + char_boxes[2].tl().y)/2);
    estimates.top1_a0 = estimates.bottom1_a0 + (p.y - (estimates.bottom1_a0+p.x*estimates.bottom1_a1));
    p = char_boxes[0].tl();
    err = (p.y - (estimates.top1_a0+p.x*estimates.top1_a1));
  }

  if (abs(err) > (float)estimates.h_max/6)
  {
    // We need two different top lines
    estimates.top2_a0 = estimates.top1_a0 + err;
  }
  else 
  {
    // Second top line is the same
    estimates.top2_a0 = estimates.top1_a0;
  }
}

int main( int argc, char** argv )
{

    if (argc < 2)
    {

      cout << "Usage: ./get_triplet_intervals <gt_file1> <gt_file2> ... <gt_fileN>" << endl;
      return -1;

    } 

    vector<float> top_height_ratios;
    vector<float> bottom_height_ratios;
    vector<float> slopes;
    
    for (int f=1; f<argc; f++) {


      Mat gt;
      gt = imread(argv[f]);
      cvtColor(gt, gt, CV_RGB2GRAY);
      threshold(gt, gt, 1, 255, CV_THRESH_BINARY_INV);
     
      vector<vector<Point> > contours;
      vector<Vec4i> hierarchy;
       
      findContours( gt, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

      vector<Rect> char_boxes;
      vector< vector<Point> > char_contours;

      for( int i = 0; i< contours.size(); i++ )
      {
        Rect bbox = boundingRect(contours[i]);

        if((hierarchy[i][3] == 0) && ((bbox.width>6) || (bbox.height>6)))
        {
          char_boxes.push_back(bbox);
          char_contours.push_back(contours[i]);
          //Mat drawing = Mat::zeros( gt.size(), CV_8UC1);
          //drawContours( drawing, contours, i, Scalar(255), CV_FILLED, 8, hierarchy, 0, Point() );
          //for( int j = 0; j< contours.size(); j++ )
          //{
          //  if (hierarchy[j][3] == i)
          //    drawContours( drawing, contours, j, Scalar(0), CV_FILLED, 8, hierarchy, 0, Point() );
          //}
          //imshow( "gt contours", drawing );
          //waitKey(0);
        }
      }


      if (char_boxes.size() < 3)
        continue;
      
      sort(char_boxes.begin(),char_boxes.end(), sortRects);
      sort(char_contours.begin(),char_contours.end(), sortContours);


      for( int i = 0; i< char_boxes.size()-2; i++ )
      {
        Mat drawing = Mat::zeros( gt.size(), CV_8UC3);
        drawContours(drawing, char_contours, i, Scalar(255,255,255), CV_FILLED);
        drawContours(drawing, char_contours, i+1, Scalar(255,255,255), CV_FILLED);
        drawContours(drawing, char_contours, i+2, Scalar(255,255,255), CV_FILLED);
        rectangle(drawing, char_boxes[i].tl(), char_boxes[i].br(), Scalar(255,0,0));
        rectangle(drawing, char_boxes[i+1].tl(), char_boxes[i+1].br(), Scalar(255,0,0));
        rectangle(drawing, char_boxes[i+2].tl(), char_boxes[i+2].br(), Scalar(255,0,0));

        line_estimates estimates;
        fitLineEstimates(char_contours, Vec3i(i,i+1,i+2), estimates);

        //cout << " top1 " << estimates.top1_a0 << endl;
        //cout << " top2 " << estimates.top2_a0 << endl;
        //cout << " bottom1 " << estimates.bottom1_a0 << endl;
        //cout << " bottom2 " << estimates.bottom2_a0 << endl;

        int central_height = min(estimates.bottom1_a0,estimates.bottom2_a0)-max(estimates.top1_a0,estimates.top2_a0);
        int top_height     = abs(estimates.top1_a0 - estimates.top2_a0);
        int bottom_height  = abs(estimates.bottom1_a0 - estimates.bottom2_a0);

        top_height_ratios.push_back((float)top_height/central_height);
        bottom_height_ratios.push_back((float)bottom_height/central_height);
        slopes.push_back(abs(estimates.bottom1_a1));

        //cout << "top height ratio " << top_height_ratios[top_height_ratios.size()-1] << endl;
        //cout << "bottom height ratio " << bottom_height_ratios[bottom_height_ratios.size()-1] << endl;

        // Set this to true if you want to inspect the line estimates
        if ( false ) 
        {
          line(drawing, Point(0,(int)estimates.bottom1_a0), 
                        Point(gt.cols,(int)(estimates.bottom1_a0+estimates.bottom1_a1*gt.cols)), Scalar(0,255,0));
          line(drawing, Point(0,(int)estimates.bottom2_a0), 
                        Point(gt.cols,(int)(estimates.bottom2_a0+estimates.bottom2_a1*gt.cols)), Scalar(0,255,0));
          line(drawing, Point(0,(int)estimates.top1_a0), 
                        Point(gt.cols,(int)(estimates.top1_a0+estimates.top1_a1*gt.cols)), Scalar(0,255,0));
          line(drawing, Point(0,(int)estimates.top2_a0), 
                        Point(gt.cols,(int)(estimates.top2_a0+estimates.top2_a1*gt.cols)), Scalar(0,255,0));
  
          resize(drawing,drawing,Size(800,600));
          imshow( "line estimates", drawing );
          waitKey(0);
        }

        cout << "." << flush;
      }

    } // end for all GT files

    cout << endl;
    double minValue, maxValue;
    minMaxIdx(Mat(top_height_ratios), &minValue, &maxValue);
    cout << " top height ratios        min=" << minValue << " max=" << maxValue << endl;
    minMaxIdx(Mat(bottom_height_ratios), &minValue, &maxValue);
    cout << " bottom height ratios     min=" << minValue << " max=" << maxValue << endl;
    minMaxIdx(Mat(slopes), &minValue, &maxValue);
    cout << " slopes                   min=" << minValue << " max=" << maxValue << endl;
}
