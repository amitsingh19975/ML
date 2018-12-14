#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include "core.hpp"
#include "LinearRegression.hpp"
namespace ML{
    using namespace boost::numeric::ublas;
    struct LogisticRegression : LinearRegression{
        size_t _rows;
        size_t _cols;
        using Core::train;
        using Core::predict;
        matrix<double> _mu;
        double _mean{0};
        float _threshold{0.5};
        LogisticRegression(float threshold = 0.5):_threshold(threshold){
            _func = [](double exp ,double p){return exp;};
        }

        Frame* predict(Frame* xTestData) override{
            if(!this->_isTrained) return nullptr;
            this->_rows = xTestData->_rows;
            this->_cols = xTestData->_cols;

            matrix<double> xTest(this->_rows,this->_cols + 1),yTest(this->_rows,1);
            
            convertToUblasMatrix(xTestData,xTest,false);

            this->_predicM = prod(xTest,this->_coeff);
            for(int i = 0; i < this->_predicM.size1(); i++){
                this->_predicM(i,0) = calProb(this->_predicM(i,0)) >= _threshold ? 1 : 0; 
            } 
            setPredict();
            return this->_predic.get();
        }

        void squareDueRegression() override{
            this->_squareDueRegression = 0;
            matrix<double> temp = prod(this->_x,this->_coeff);
            for(int i = 0; i < temp.size1(); i++){
                this->_squareDueRegression += (temp(i,0) >= 0.5 ? 1 : 0 - this->_mean) * (temp(i,0) >= 0.5 ? 1 : 0 - this->_mean);
                this->_totalSumOfSquare += (this->_y(i,0) - this->_mean) * (this->_y(i,0) - this->_mean);
            }
            this->squareDueResidual(temp);
        }
        void squareDueResidual(matrix<double> &m) override{
            m = this->_y - m;
            this->_squareDueResidual = prod(trans(m),m)(0,0);
        }


    protected:
        double calProb(double exp){
            return (1.0 / (std::exp(-exp) + 1.0));
        }

        void calCoeff() override{
            matrix<double> R(this->_y.size1(),this->_y.size1());
            matrix<double> temp, oldW, xT;
            for(int i = 0; i < 20; i++){
                calDigonal(R);
                xT = trans(this->_x);
                temp = prod(xT,R);
                temp = prod(temp, this->_x);
                matrix<double> inv(temp.size1(),temp.size2());
                inverseMatrix(temp,inv);
                // (X^TSX)^-1
                temp = prod(inv,xT);
                // (X^TSX)^-1X^T
                oldW = this->_y - this->_mu;
                temp = prod(temp,oldW);
                this->_coeff = this->_coeff + temp;
            }

        }

        void calDigonal(matrix<double>& m){
            calCoeffHelper();
            for(int i = 0; i < this->_y.size1(); i++){
                for(int j = 0; j < this->_y.size1(); j++){
                    if(i == j){
                        m(i,j) = this->_mu(i,0)*( 1 - this->_mu(i,0));
                    }else{
                        m(i,j) = 0;
                    }
                }
            }
        }

        void calCoeffHelper(){
            this->_mu = prod(this->_x,this->_coeff);
            for(int i = 0; i < this->_mu.size1(); i++){
                this->_mu(i,0) = calProb(this->_mu(i,0));
            }

        }

        bool cmp(matrix<double>& m, matrix<double>& n){
            if(m.size1() != n.size1() && m.size2() != n.size2()) return false;

            for(int i = 0; i < m.size1(); i++){
                for(int j = 0; j < m.size2(); j++){
                    if(m(i,j) != n(i,j)) return false;
                }
            }
            return true;
        }

    };
} // ML


#endif // LOGISTIC_REGRESSION_H
