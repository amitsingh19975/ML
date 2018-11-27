#include "inculde/headers.h"
#include <iostream>
#include <iomanip>

using namespace ML;
using namespace std;

int main(){
    // cout<<setprecision(numeric_limits<double>::max_digits10);
    string train = "/Users/Amit/Desktop/Python/ML/test/kyphosis";
    string test = "/Users/Amit/Desktop/Python/ML/test/test";
    CSV frameTrain(train);
    frameTrain.convertToNumber(0);
    frameTrain[0]->print();
    // CSV frameTest(test);
    // Frame frameTrainX(std::move(frameTrain.dropCol(0)),"Train X");
    // Frame frameTrainY(std::move(frameTrain.dropCol(0)),"Train Y");
    // Frame frameTestX(std::move(frameTest.dropCol(0)),"Test X");
    // Frame frameTestY(std::move(frameTest.dropCol(0)),"Test Y");
    // std::vector<std::vector<double>> train = {
    //     {1,2,3,4},
    //     {1,2,3,4}
    // };
    // std::vector<std::vector<double>> test = {
    //     {5,6,7,8},
    //     {5,6,7,8}
    // };
    // Frame frameTrainX(train[0],"Train X");
    // Frame frameTrainY(train[1],"Train Y");
    // Frame frameTestX(test[0], "Test X");
    // Frame frameTestY(test[1], "Test Y");
    // frameTrainX.normalize();
    // frameTrainY.normalize();
    // frameTestX.normalize();
    // frameTestY.normalize();
    // frame.print();
    // csv.print();
    // LogisticRegression k;
    // k.train(frameTrainX,frameTrainY);
    // k.test(frameTestX,frameTestY);
    // cout<<k.rSquared()<<'\n';
    return 0;
}