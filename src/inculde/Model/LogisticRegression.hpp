#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include "core.hpp"
#include "LinearRegression.hpp"
namespace ML{
    struct LogisticRegression : LinearRegression{
        size_t _rows;
        size_t _cols;
        using Core::train;
        using Core::predict;
        Eigen::MatrixXd _mu;
        double _mean{0};
        float _threshold{0.5};
        LogisticRegression(float threshold = 0.5):_threshold(threshold){
            _func = [](double exp ,double p){return exp;};
        }

        Frame* predict(Frame* xTestData) override{
            if(!this->_isTrained) return nullptr;
            this->_rows = xTestData->_rows;
            this->_cols = xTestData->_cols;

            Eigen::MatrixXd xTest(this->_rows,this->_cols + 1),yTest(this->_rows,1);
            
            convertToEigenMatrix(xTestData,xTest,false);

            this->_predicM = (xTest * this->_coeff);
            for(int i = 0; i < this->_predicM.rows(); i++){
                this->_predicM(i,0) = calProb(this->_predicM(i,0)) >= _threshold ? 1 : 0; 
            } 
            setPredict();
            return this->_predic.get();
        }

        void squareDueRegression() override{
            this->_squareDueRegression = 0;
            Eigen::MatrixXd temp = (this->_x * this->_coeff);
            for(int i = 0; i < temp.rows(); i++){
                this->_squareDueRegression += (temp(i,0) >= 0.5 ? 1 : 0 - this->_mean) * (temp(i,0) >= 0.5 ? 1 : 0 - this->_mean);
                this->_totalSumOfSquare += (this->_y(i,0) - this->_mean) * (this->_y(i,0) - this->_mean);
            }
            this->squareDueResidual(temp);
        }
        void squareDueResidual(Eigen::MatrixXd &m) override{
            m = this->_y - m;
            this->_squareDueResidual = (m.transpose() * m)(0,0);
        }


    protected:
        double calProb(double exp){
            return (1.0 / (std::exp(-exp) + 1.0));
        }

        void calCoeff() override{
            Eigen::MatrixXd R(this->_y.rows(),this->_y.rows());
            Eigen::MatrixXd temp, oldW, xT;
            for(int i = 0; i < 20; i++){
                calDigonal(R);
                xT = this->_x.transpose();
                temp = (xT * R);
                temp = (temp * this->_x);
                Eigen::MatrixXd inv = temp.inverse();
                // (X^TSX)^-1
                temp = (inv * xT);
                // (X^TSX)^-1X^T
                oldW = this->_y - this->_mu;
                temp = (temp * oldW);
                this->_coeff = this->_coeff + temp;
            }

        }

        void calDigonal(Eigen::MatrixXd& m){
            calCoeffHelper();
            for(int i = 0; i < this->_y.rows(); i++){
                for(int j = 0; j < this->_y.rows(); j++){
                    if(i == j){
                        m(i,j) = this->_mu(i,0)*( 1 - this->_mu(i,0));
                    }else{
                        m(i,j) = 0;
                    }
                }
            }
        }

        void calCoeffHelper(){
            this->_mu = (this->_x * this->_coeff);
            for(int i = 0; i < this->_mu.rows(); i++){
                this->_mu(i,0) = calProb(this->_mu(i,0));
            }

        }

        bool cmp(Eigen::MatrixXd& m, Eigen::MatrixXd& n){
            if(m.rows() != n.rows() && m.cols() != n.cols()) return false;

            for(int i = 0; i < m.rows(); i++){
                for(int j = 0; j < m.cols(); j++){
                    if(m(i,j) != n(i,j)) return false;
                }
            }
            return true;
        }

    };
} // ML


#endif // LOGISTIC_REGRESSION_H
