#include "ransac.h"

// RANSACによるアウトライン除去
void Ransac::ransac(vector<Point3f> points1, vector<Point3f> points2, int point_num, double d, Matrix3d &E_best) {
  std::random_device rnd;     // 非決定的な乱数生成器を生成
  std::mt19937 mt(rnd());     //  メルセンヌ・ツイスタの32ビット版、引数は初期シード値
  std::uniform_int_distribution<> rand(0, points1.size()-1);        // 一様乱数

  int cnt_max = 0;

  int loop_max = log(1-0.999) / log(1-pow(0.6, point_num)); // インライアの割合0.7, 全体でアウトライアを含まない点を選択できる確率0.99
  cout << "loop times = " << loop_max << endl;
  for (int i=0; i<loop_max; i++) {
    // ランダムでindexを8つ選択
    vector<int> indexes;
    for (int j=0; j<point_num; j++) {
        int index;  
        for (;;) {
          index = rand(mt);
          bool flag = true;
          for (int k=0; k<indexes.size(); k++) {
              if (index == indexes[k]) {
                  flag = false;
                  break;
              }
          }
          if (flag == true) break;
      }
      
      indexes.push_back(index);  // ランダムでindexを生成
    }

    // 8点法
    // 行列Aを設定 Ax = 0
    Eigen::MatrixXd A(point_num, 9);
    for (int j=0; j<point_num; j++) {
      A(j, 0) = points1[indexes[j]].x*points2[indexes[j]].x;
      A(j, 1) = points1[indexes[j]].x*points2[indexes[j]].y;
      A(j, 2) = points1[indexes[j]].x*points2[indexes[j]].z;
      A(j, 3) = points1[indexes[j]].y*points2[indexes[j]].x;
      A(j, 4) = points1[indexes[j]].y*points2[indexes[j]].y;
      A(j, 5) = points1[indexes[j]].y*points2[indexes[j]].z;
      A(j, 6) = points1[indexes[j]].z*points2[indexes[j]].x;
      A(j, 7) = points1[indexes[j]].z*points2[indexes[j]].y;
      A(j, 8) = points1[indexes[j]].z*points2[indexes[j]].z;
    }
    JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
    VectorXd min_vector = svd.matrixV().col(7);
    
    // 基本行列
    Matrix3d E; 
    E << min_vector(0), min_vector(1), min_vector(2), min_vector(3), min_vector(4), min_vector(5), min_vector(6), min_vector(7), min_vector(8);

    // 条件を満たすデータを数え上げる
    int cnt = 0;
    for (int j=0; j<points1.size(); j++) {
      Matrix<double, 3, 1> point1, point2;
      point1 << points1[j].x, points1[j].y, points1[j].z;
      point2 << points2[j].x, points2[j].y, points2[j].z;
      double dis = abs(asin(point1.dot(E*point2) / (point1.norm()*(E*point2).norm())));
      if (dis < d) cnt++; // +-10度がインライン
    }

    // 最高スコアを更新したら
    if (cnt > cnt_max) {
        E_best = E;
        cnt_max = cnt;
    }

  }
  cout << "inlier radio = " << static_cast<double>(cnt_max)/points1.size() << endl;
}

