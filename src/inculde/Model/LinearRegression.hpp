#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include "core.hpp"
#include "PolyRegression.hpp"

namespace ML{
    using namespace boost::numeric::ublas;
    struct LinearRegression : PolyRegression{
        size_t          _rows;
        size_t          _cols;
        float           _threshold{0.5};
        
        using Core::train;
        using Core::test;
        LinearRegression(float threshold = 0.5):_threshold(threshold){
            _func = [](double exp ,double p){return exp;};
        }

        void test(Frame* xTestData, Frame* yTestData) override{
            if(!this->_isTrained) return;

            matrix<double> xTest(xTestData->_rows,xTestData->_cols + 1),yTest(yTestData->_rows,1);
            convertToUblasMatrix(xTestData,xTest,false);
            convertToUblasMatrix(yTestData,yTest);

            this->_predic.resize(xTest.size1(),1);
            this->_errorInPredic.resize(xTest.size1(),1);

            for(int i = 0; i < xTest.size1(); i++) this->_predic(i,0) = 0;

            
            for(int i = 0; i < xTest.size1(); i++){
                for(int j = 0; j < this->_coeff.size1(); j++){
                    this->_predic(i,0) += this->_coeff(j,0) * xTest(i,j); 
                }
            }  
        }

        void train(Frame* xTrainData, Frame* yTrainData) override{
            this->_rows = xTrainData->_rows;
            this->_cols = xTrainData->_cols;
            this->_dimension = xTrainData->_cols + 1;
            this->_x.resize(this->_rows, this->_cols + 1);
            this->_y.resize(this->_rows, 1);

            convertToUblasMatrix(xTrainData,this->_x,false);
            convertToUblasMatrix(yTrainData,this->_y);

            this->_mean = yTrainData->mean(0);
            this->_coeff.resize(this->_x.size2(),1);
            for(int i = 0; i < this->_coeff.size1();i++){
                this->_coeff(i,0) = 0;
            }
            this->fit();
            _isTrained = true;
        }

    protected:
        void convertToUblasMatrix(Frame* v,matrix<double> &m,bool isY = true) override{
            if(isY){
                for(size_t i = 0; i < v->_rows; i++){
                    m(i,0) = v->at(i,0);
                }
            }else{
                for(size_t i = 0; i < m.size2(); i++){
                    for(size_t j = 0; j < m.size1(); j++){
                        if(i == 0) {
                            m(j,i) = 1; 
                        }else{
                            m(j,i) = v->at(j,i - 1);
                        }
                    }
                }
            }
        }

    };
} // ML


#endif // LINEAR_REGRESSION_H
