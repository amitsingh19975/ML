#ifndef POLY_REGRESSION_H
#define POLY_REGRESSION_H

#include "core.hpp"
#include <utility>
#include <iostream>
#include <vector>
#include <cmath>

namespace ML{
    using namespace boost::numeric::ublas;
    struct PolyRegression : Core{
    
    public:
        matrix<double>  _w;
        matrix<double>  _coeff;
        double          _squareDueResidual;
        double          _squareDueRegression;
        double          _totalSumOfSquare;
        matrix<double>  _predic;
        matrix<double>  _errorInPredic;
        std::function<double(double,double)> _func = [](double exp, double pow){return std::pow(exp,pow);};

        PolyRegression(int dimension = 1,bool ortho = false):_dimension(dimension + 1),_ortho(ortho),_coeff(dimension + 1, 1){}

        double rSquared();
        std::vector<double> getCoeff() const noexcept;
        void train(Frame* xTrainData, Frame* yTrainData) override;
        void test(Frame* xTrainData, Frame* yTrainData) override;
    protected:
        matrix<double>  _x;
        matrix<double>  _y;
        int             _dimension;
        bool            _ortho{false};
        bool            _isTrained{false};

        void fit();
        virtual void convertToUblasMatrix(Frame* v,matrix<double> &m,bool isY = true);
        void transposeMatrix(matrix<double>& m);
        bool inverseMatrix(matrix<double>& m, matrix<double>& inv);
        void calWeights();
        void calCoeff();
        void squareDueRegression();
        void squareDueResidual(double bxy,double mean);
        void totalSumOfSquare(double mean,double yy);
        void getOrthoMatrix(matrix<double>& ortho);
        
        template<typename T>
        void print(matrix<T> const& m){
            for(int i = 0; i < m.size1(); i++){
                for(int j = 0; j < m.size2(); j++){
                    std::cout<<m(i,j)<<' ';
                }
                std::cout<<'\n';
            }
        }
    };

    void PolyRegression::fit(){
        calCoeff();
    }

    void PolyRegression::convertToUblasMatrix(Frame* v,matrix<double> &m,bool isY){
        if(isY){
            for(size_t i = 0; i < m.size1(); i++){
                m(i,0) = v->at(i,0);
            }
        }else{
            for(size_t i = 0; i < m.size1(); i++){
                for(size_t j = 0; j < m.size2(); j++){
                    m(i,j) = std::pow(v->at(i,0),j); 
                }
            }
        }
    }

    bool PolyRegression::inverseMatrix(matrix<double>& m, matrix<double>& inv){
        matrix<double> temp(m);
        permutation_matrix<size_t> pm(temp.size1());

        if(lu_factorize(temp,pm) != 0) return false;

        inv.assign(identity_matrix<double>(temp.size1()));
        lu_substitute(temp, pm, inv);
        return true;
    }

    void PolyRegression::calWeights(){
        matrix<double> temp;
        this->_w = trans(this->_x);
        if(!_ortho){
            temp = prod(this->_w,this->_x);
        }else{
            getOrthoMatrix(temp);
        }
        matrix<double> inv(temp.size1(),temp.size2());
        inverseMatrix(temp,inv);
        this->_w = prod(inv,this->_w);
    }

    void PolyRegression::calCoeff(){
        this->calWeights();
        this->_coeff = prod(this->_w,this->_y);
    }

    std::vector<double> PolyRegression::getCoeff() const noexcept{
        std::vector<double> v(this->_coeff.size1());
        for(int i = 0; i < this->_coeff.size1(); i++) v[i] = this->_coeff(i,0);
        return v;
    }

    double PolyRegression::rSquared(){
        this->squareDueRegression();
        return ((this->_squareDueRegression / this->_totalSumOfSquare));
    }

    void PolyRegression::squareDueRegression(){
        double bxy{0};

        double temp{0};
        for(int i = 0; i < this->_y.size1(); i++){
            temp += (this->_y(i,0));
        }
        temp = (temp * temp) / this->_y.size1();

        matrix<double> bTrans = trans(this->_coeff);
        matrix<double> xTrans = trans(this->_x);
        matrix<double> product = prod(bTrans,xTrans);
        bxy = prod(product,this->_y)(0,0);
        this->_squareDueRegression = bxy - temp;
        this->squareDueResidual(bxy, temp);
    }
    void PolyRegression::squareDueResidual(double bxy, double mean){
        double yy = prod(trans(this->_y), this->_y)(0,0);
        this->_squareDueResidual = yy - bxy;
        this->totalSumOfSquare(mean,yy);
    }
    void PolyRegression::totalSumOfSquare(double mean,double yy){
        this->_totalSumOfSquare = yy - mean;
    }

    void PolyRegression::getOrthoMatrix(matrix<double>& ortho){
        ortho.resize(this->_x.size2(),this->_x.size2());
        
        for(int i = 0; i < ortho.size1(); i++){
            for(int j = 0; j < ortho.size2(); j++){
                ortho(i,j) = 0;
            }
        }

        for(int i = 0; i < this->_x.size2() ;i++){
            for(int j = 0; j < this->_x.size1(); j++){
                ortho(i,i) += std::pow(std::pow(this->_x(j,1),i),2);
            }
        }
    }
    void PolyRegression::test(Frame* xTestData, Frame* yTestData){
        if(!this->_isTrained) return;

        auto x = Frame::cast(xTestData->at(0));
        auto y = Frame::cast(yTestData->at(0));

        matrix<double> xTest(x->size(),1),yTest(y->size(),1);
        convertToUblasMatrix(xTestData,xTest);
        convertToUblasMatrix(yTestData,yTest);

        this->_predic.resize(xTest.size1(),1);
        this->_errorInPredic.resize(xTest.size1(),1);

        for(int i = 0; i < xTest.size1(); i++) this->_predic(i,0) = 0;

        for(int i = 0; i < xTest.size1(); i++){
            for(int j = 0; j < this->_coeff.size1(); j++){
                this->_predic(i,0) += this->_coeff(j,0) * _func(xTest(i,0),j); 
            }
        }  
        for(int i = 0; i < xTest.size1(); i++) this->_errorInPredic(i,0) = (this->_predic(i,0) - yTest(i,0));

    }
    void PolyRegression::train(Frame* xTrainData, Frame* yTrainData){
        auto x = Frame::cast(xTrainData->at(0));
        auto y = Frame::cast(yTrainData->at(0));
        
        this->_x.resize(x->size(), this->_dimension);
        this->_y.resize(y->size(), 1);

        convertToUblasMatrix(xTrainData,this->_x,false);
        convertToUblasMatrix(yTrainData,this->_y);

        this->fit();
        _isTrained = true;
    }
} // ML


#endif // POLY_REGRESSION_H
