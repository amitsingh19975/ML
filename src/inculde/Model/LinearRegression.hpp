#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include "core.hpp"
#include "PolyRegression.hpp"

namespace ML{
    struct LinearRegression : PolyRegression{
        size_t          _rows;
        size_t          _cols;
        float           _threshold{0.5};
        
        using Core::train;
        using Core::predict;
        LinearRegression(float threshold = 0.5):_threshold(threshold){
            _func = [](double exp ,double p){return exp;};
        }

        Frame* predict(Frame* xTestData) override{
            if(!this->_isTrained) return nullptr;

            Eigen::MatrixXd xTest(xTestData->_rows,xTestData->_cols + 1);
            convertToEigenMatrix(xTestData,xTest,false);

            this->_predicM.resize(xTest.rows(),1);
            _predicM.setZero();
            
            for(int i = 0; i < xTest.rows(); i++){
                for(int j = 0; j < this->_coeff.rows(); j++){
                    this->_predicM(i,0) += this->_coeff(j,0) * xTest(i,j); 
                }
            }
            setPredict();
            return this->_predic.get();
        }

        void train(Frame* xTrainData, Frame* yTrainData) override{
            this->_rows = xTrainData->_rows;
            this->_cols = xTrainData->_cols;
            this->_dimension = xTrainData->_cols + 1;
            this->_x.resize(this->_rows, this->_cols + 1);
            this->_y.resize(this->_rows, 1);

            convertToEigenMatrix(xTrainData,this->_x,false);
            convertToEigenMatrix(yTrainData,this->_y);

            this->_mean = yTrainData->mean(0);
            this->_label = yTrainData->getLabel(0);
            this->_coeff.resize(this->_x.cols(),1);

            for(int i = 0; i < this->_coeff.rows();i++){
                this->_coeff(i,0) = 0;
            }
            this->fit();
            _isTrained = true;
        }

    protected:
        void convertToEigenMatrix(Frame* v,Eigen::MatrixXd &m,bool isY = true) override{
            if(isY){
                for(size_t i = 0; i < v->_rows; i++){
                    m(i,0) = v->at(i,0);
                }
                this->_headerY = v->_headers[0];
            }else{
                for(size_t i = 0; i < m.cols(); i++){
                    for(size_t j = 0; j < m.rows(); j++){
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
