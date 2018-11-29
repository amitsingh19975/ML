#include "inculde/headers.h"
#include <iostream>
#include <iomanip>

using namespace ML;
using namespace std;

int main(){
    // cout<<setprecision(numeric_limits<double>::max_digits10);
    string trainF = "/Users/Amit/Desktop/Python/ML/test/train";
    string testF = "/Users/Amit/Desktop/Python/ML/test/test";
    CSV frameTrain(trainF);
    CSV frameTest(testF);
    // frameTrain.labelToNumber(0);
    // frameTrain[0]->print();
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
    frameTrain.normalize();
    frameTest.normalize();
    // frameTrain.labelToNumber(0);
    // auto [train, test] = frameTrain.split(80,0);
    // frameTrainX.normalize();
    // frameTrainY.normalize();
    // frameTestX.normalize();
    // frameTestY.normalize();
    // frame.print();
    // csv.print();
    LogisticRegression k;
    // frameTrain.print(3);
    auto frameTrainX = std::move(frameTrain.colSlice(1));
    cout<<"->";
    auto frameTrainY = std::move(frameTrain.colSlice(0));
    // auto frameTestX = std::move(frameTest.colSlice(1));
    // auto frameTestY = std::move(frameTest.colSlice(0));
    // frameTrainY->print();
    // auto frameTrainX = std::move(train->colSlice(1));
    // auto frameTrainY = std::move(train->colSlice(0));
    // auto frameTestX = std::move(test->colSlice(1));
    // auto frameTestY = std::move(test->colSlice(0));
    // k.train(frameTrainX.get(),frameTrainY.get());
    // k.test(frameTestX.get(),frameTestY.get());
    // cout<<(k._predic)<<'\n';
    return 0;
}