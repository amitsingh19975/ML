#include "inculde/headers.h"
#include <iostream>
#include <iomanip>

using namespace ML;
using namespace std;

int main(){
    // // cout<<setprecision(numeric_limits<double>::max_digits10);
    string trainF = "/Users/Amit/Desktop/Python/ML/test/cars";
    // string testF = "/Users/Amit/Desktop/Python/ML/test/data";
    CSV frameTrain(trainF);
    // // CSV frameTest(testF);
    // std::vector<double> trainD = {
    //     {1,2,3,4,5,6,7,8}
    // };
    // frameTrain.info();
    PPrint::print(frameTrain);
    // std::vector<std::vector<double>> testD = {
    //     {5,6,7,8},
    //     {5,6,7,8}
    // };
    // Frame frameTrain(trainD);
    // frameTrain.print();
    // f.print();
    // frameTrain.normalize();
    // frameTrain.dropCol(0);
    // frameTrain.labelToNumber(0);
    auto y = frameTrain[{"brand"}];
    auto X = frameTrain.colSlice(0);
    // PPrint::print(X);
    // auto X = frameTrain[{"radius_mean", "texture_mean", "smoothness_mean",
    //    "compactness_mean", "symmetry_mean", "fractal_dimension_mean",
    //    "radius_se", "texture_se", "smoothness_se", "compactness_se",
    //    "symmetry_se", "fractal_dimension_se"}];
    // PCA p(X);
    // FrameShared f = p.getReducedFrame();
    // PPrint::print(frameTrain);
    // auto [X_train,X_test,y_train,y_test] = frameTrain.split(X,y,30);
    // KMean k(3,42,300);
    // k.train(X_train);
    // auto q = k.predict(X_test,y_test);
    // Metrics m(y_test,p);
    // cout<<k.adjRSquared()<<'\n';
    // m.confusionMatrix();
    // m.listRates();
    // std::vector<std::vector<double>> M = {
    //     {3.0,1.0},
    //     {1.0,3.0},
    // };
    return 0;
}