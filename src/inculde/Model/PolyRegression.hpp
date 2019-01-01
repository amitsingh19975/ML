#ifndef POLY_REGRESSION_H
#define POLY_REGRESSION_H

#include "core.hpp"

namespace ML{
    struct PolyRegression : Core{
    
    public:
        Eigen::MatrixXd  _w;
        using Core::train;
        using Core::predict;
        //lambda function how to init independent variable
        std::function<double(double,double)> _func = [](double exp, double pow){return std::pow(exp,pow);};

        PolyRegression(int dimension = 1,bool ortho = false):_dimension(dimension + 1),_ortho(ortho){_coeff.resize(dimension + 1, 1);}
        //returns r^2
        virtual double RSquared();
        //returns adjusted r^2
        virtual double adjRSquared();
        //get the regression coeff
        std::vector<double> getCoeff() const noexcept;
        void train(Frame* xTrainData, Frame* yTrainData) override;
        Frame* predict(Frame* xTrainData) override;

    protected:
        int             _dimension;
        bool            _ortho{false};
        bool            _isTrained{false};
        //utility function for training
        void fit();
        virtual void convertToEigenMatrix(Frame* v,Eigen::MatrixXd &m,bool isY = true);
        void transposeMatrix(Eigen::MatrixXd& m);
        //calculates weights
        virtual void calWeights();
        //calculates regression coeff
        virtual void calCoeff();
        virtual void squareDueRegression();
        virtual void squareDueResidual(Eigen::MatrixXd &m);
        //get the orthogonal matrix
        void getOrthoMatrix(Eigen::MatrixXd& ortho);
    };

    void PolyRegression::fit(){
        calCoeff();
    }

    void PolyRegression::convertToEigenMatrix(Frame* v,Eigen::MatrixXd &m,bool isY){
        if(isY){
            for(size_t i = 0; i < m.rows(); i++){
                m(i,0) = v->at(i,0);
            }
            this->_headerY = v->_headers[0];
        }else{
            for(size_t i = 0; i < m.rows(); i++){
                for(size_t j = 0; j < m.cols(); j++){
                    m(i,j) = std::pow(v->at(i,0),j); 
                }
            }
        }
    }

    void PolyRegression::calWeights(){
        Eigen::MatrixXd temp;
        this->_w = _x.transpose();
        if(!_ortho){
            temp = _w * _x;
        }else{
            getOrthoMatrix(temp);
        }
        Eigen::MatrixXd inv(temp.rows(),temp.rows());
        inv = temp.inverse();
        this->_w = inv * _w;
    }

    void PolyRegression::calCoeff(){
        this->calWeights();
        this->_coeff = _w * _y;
    }

    std::vector<double> PolyRegression::getCoeff() const noexcept{
        std::vector<double> v(this->_coeff.rows());
        for(int i = 0; i < this->_coeff.rows(); i++) v.at(i) = this->_coeff(i,0);
        return v;
    }

    double PolyRegression::RSquared(){
        this->squareDueRegression();
        return ((this->_squareDueRegression / this->_totalSumOfSquare));
    }
    
    double PolyRegression::adjRSquared(){
        double r = this->RSquared();
        int n = this->_y.rows();
        int k = this->_x.cols() - 1;
        return (1 - ((1 - r * r) * ( n - 1)) / (n - k - 1));
    }

    void PolyRegression::squareDueRegression(){
        this->_squareDueRegression = 0;
        Eigen::MatrixXd temp = this->_x * this->_coeff;
        for(int i = 0; i < temp.rows(); i++){
            this->_squareDueRegression += (temp(i,0) - this->_mean) * (temp(i,0) - this->_mean);
            this->_totalSumOfSquare += (this->_y(i,0) - this->_mean) * (this->_y(i,0) - this->_mean);
        }
        this->squareDueResidual(temp);
    }
    void PolyRegression::squareDueResidual(Eigen::MatrixXd &m){
        m = this->_y - m;
        this->_squareDueResidual = (m.transpose() * m)(0,0);
    }

    void PolyRegression::getOrthoMatrix(Eigen::MatrixXd& ortho){
        ortho.resize(this->_x.cols(),this->_x.cols());
        
        for(int i = 0; i < ortho.rows(); i++){
            for(int j = 0; j < ortho.cols(); j++){
                ortho(i,j) = 0;
            }
        }

        for(int i = 0; i < this->_x.cols() ;i++){
            for(int j = 0; j < this->_x.rows(); j++){
                ortho(i,i) += std::pow(std::pow(this->_x(j,1),i),2);
            }
        }
    }
    Frame* PolyRegression::predict(Frame* xTestData){
        if(!this->_isTrained) return nullptr;

        auto x = Frame::cast(xTestData->at(0));

        Eigen::MatrixXd xTest(x->size(),1);
        convertToEigenMatrix(xTestData,xTest);

        this->_predicM.resize(xTest.rows(),1);

        for(int i = 0; i < xTest.rows(); i++) this->_predicM(i,0) = 0;

        for(int i = 0; i < xTest.rows(); i++){
            for(int j = 0; j < this->_coeff.rows(); j++){
                this->_predicM(i,0) += this->_coeff(j,0) * _func(xTest(i,0),j); 
            }
        }  
        setPredict();
        return this->_predic.get();
    }
    void PolyRegression::train(Frame* xTrainData, Frame* yTrainData){
        auto x = Frame::cast(xTrainData->at(0));
        auto y = Frame::cast(yTrainData->at(0));
        
        this->_x.resize(x->size(), this->_dimension);
        this->_y.resize(y->size(), 1);

        convertToEigenMatrix(xTrainData,this->_x,false);
        convertToEigenMatrix(yTrainData,this->_y);
        this->_label = yTrainData->getLabel(0);
        this->_mean = yTrainData->mean(0);

        this->fit();
        _isTrained = true;
    }
} // ML


#endif // POLY_REGRESSION_H
