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
                this->_predic(i,0) = calProb(this->_predic(i,0)); 
                this->_errorInPredic(i,0) = this->_predic(i,0) - yTest(i,0);
                std::cout<<this->_predic(i,0)<<':'<<yTest(i,0)<<'\n';
            }  
            
        }

        void train(Frame* xTrainData, Frame* yTrainData) override{
            this->_rows = xTrainData->_rows;
            this->_cols = xTrainData->_cols;
            this->_dimension = xTrainData->_cols;
            this->_coeff.resize(xTrainData->_cols,1);

            this->_x.resize(this->_rows, this->_cols + 1);
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
        void convertToUblasMatrix(Frame* v,matrix<double> &m,bool isY = true) override{
            if(isY){
                for(size_t i = 0; i < this->_y.size1(); i++){
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


#endif // LOGISTIC_REGRESSION_H
