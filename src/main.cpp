#include "inculde/headers.h"
#include <iostream>
#include <iomanip>

using namespace ML;
using namespace std;

int main(){
    // cout<<setprecision(numeric_limits<double>::max_digits10);
    string trainF = "/Users/Amit/Desktop/Python/ML/test/kyphosis";
    string testF = "/Users/Amit/Desktop/Python/ML/test/data";
    CSV frameTrain(testF);
    // CSV frameTest(testF);
    // frameTrain.labelToNumber(0);
    // frameTrain.at(0)->print();
    // CSV frameTest(test);
    // Frame frameTrainX(std::move(frameTrain.dropCol(0)),"Train X");
    // Frame frameTrainY(std::move(frameTrain.dropCol(0)),"Train Y");
    // Frame frameTestX(std::move(frameTest.dropCol(0)),"Test X");
    // Frame frameTestY(std::move(frameTest.dropCol(0)),"Test Y");
    // std::vector<std::vector<double>> trainD = {
    //     {1,2,3,4,5,6,7,8},
    //     {1,2,3,4,5,6,7,8}
    // };
    // std::vector<std::vector<double>> testD = {
    //     {5,6,7,8},
    //     {5,6,7,8}
    // };
    // Frame frameTrain(trainD);
    // frameTrain.print();
    // f.print();
    // frameTrain.normalize();
    frameTrain.labelToNumber(1);
    // frameTrain.normalize();
    auto [train, test] = frameTrain.split(30);
    // frameTrainX.normalize();
    // frameTrainY.normalize();
    // frameTestX.normalize();
    // frameTestY.normalize();
    // frame.print();
    // csv.print();
    LogisticRegression k;
    // frameTrain.print(3);

    auto frameTrainY = (*train)[{"diagnosis"}];
    auto frameTrainX = (*train)[{"radius_mean", "texture_mean", "smoothness_mean",
       "compactness_mean", "symmetry_mean", "fractal_dimension_mean",
       "radius_se", "texture_se", "smoothness_se", "compactness_se",
       "symmetry_se", "fractal_dimension_se"}];
    auto frameTestX = (*test)[{"radius_mean", "texture_mean", "smoothness_mean",
       "compactness_mean", "symmetry_mean", "fractal_dimension_mean",
       "radius_se", "texture_se", "smoothness_se", "compactness_se",
       "symmetry_se", "fractal_dimension_se"}];
    auto frameTestY = (*test)[{"diagnosis"}];
    // test->print(1);
    // frameTrainX->print(4);
    k.train(frameTrainX,frameTrainY);
    k.test(frameTestX,frameTestY);
    cout<<k.adjRSquared()<<'\n';
    k.confusionMatrix();
    return 0;
}