#ifndef MY_UTIL_H_
#define MY_UTIL_H_

#include <opencv2/opencv.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/core/mat.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <Eigen/LU>
#include <Eigen/StdVector>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>

#include <iostream>
#include <ctype.h>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>
#include <cmath>
#include <random>

#define MY_PI 3.14159265358979323846

using namespace cv;
using namespace std;
using namespace Eigen;

class Util {
public:
  Util () {}
  ~Util () {}

  static int find_minEigenValue3(Vector3cd v) {
    double ev_min = 100000000000;
    int min_index = 3;
    for (int i=2; i>=0; i--) {
      if (ev_min > v[i].real()*v[i].real()) {
        ev_min = v[i].real()*v[i].real();
        min_index = i;
      }
    }
    return min_index;
  }

  static Matrix3d calc_CrossMatrix(Vector3d v) {
    Matrix3d v_cross;
    v_cross << 0.0, -1*v[2], v[1],
               v[2], 0.0, -1*v[0],
               -1*v[1], v[0], 0.0;
    return v_cross;
  }

  static void set_zeroMatrix9d(Matrix<double, 9, 9> &M) {
    for (int i=0; i<9; i++) {
      for (int j=0;j<9; j++) {
        M(i, j) = 0.0;
      }
    }
  }

  static void set_eyeMatrix9d(Matrix<double, 9, 9> &M) {
    for (int i=0; i<9; i++) {
      for (int j=0;j<9; j++) {
        M(i, j) = 0.0;
        if (i==j) M(i, j) = 1.0;
      }
    }
  }

  static void point2f_multi(Point2f src1, Point2f src2, Point2f &res) {
    res.x = src1.x * src2.x;
    res.y = src1.y * src2.y;
  }

};

#endif
