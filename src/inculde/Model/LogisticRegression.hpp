#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include "core.hpp"
#include "PolyRegression.hpp"

namespace ML{
    using namespace boost::numeric::ublas;
    struct LogisticRegression : PolyRegression{
        size_t _rows;
        size_t _cols;
        LogisticRegression(){
            _func = [](double exp ,double p){return exp;};
        }

        void test(Frame & xTestData, Frame & yTestData) override{
            PolyRegression::test(xTestData,yTestData);
            for(int i = 0; i < this->_predic.size1(); i++) this->_predic(i,0) = calProb(this->_predic(i,0));
        }

        void train(Frame & xTrainData, Frame & yTrainData) override{
            auto x = Frame::cast(xTrainData[0]);
            auto y = Frame::cast(yTrainData[0]);
            this->_rows = x->size();
            this->_cols = xTrainData._data.size();

            this->_x.resize(this->_rows, this->_cols);
            this->_y.resize(this->_rows, 1);

            convertToUblasMatrix(xTrainData,this->_x,false);
            convertToUblasMatrix(yTrainData,this->_y);

            this->fit();
            _isTrained = true;
        }

    protected:
        double calProb(double exp){
            return (1.0 / (std::exp(-exp) + 1));
        }
        void convertToUblasMatrix(Frame & v,matrix<double> &m,bool isY = true) override{
            if(isY){
                auto vec = Frame::cast(v[0]);
                for(size_t i = 0; i < this->_y.size1(); i++){
                    m(i,0) = vec->_data[i];
                }
            }else{
                for(size_t i = 0; i < this->_cols; i++){
                    auto vec = Frame::cast(v[i]);
                    for(size_t j = 0; j < this->_rows; j++){
                        m(j,i) = (vec->_data[j]); 
                    }
                }
            }
        }

    };
} // ML


#endif // LOGISTIC_REGRESSION_H
