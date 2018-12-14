#ifndef POLY_REGRESSION_H
#define POLY_REGRESSION_H

#include "core.hpp"

namespace ML{
    struct PolyRegression : Core{
    
    public:
        matrix<double>  _w;
        matrix<double>  _errorInPredic;
        using Core::train;
        using Core::predict;
        std::function<double(double,double)> _func = [](double exp, double pow){return std::pow(exp,pow);};

        PolyRegression(int dimension = 1,bool ortho = false):_dimension(dimension + 1),_ortho(ortho){_coeff.resize(dimension + 1, 1);}

        virtual double RSquared();
        virtual double adjRSquared();
        std::vector<double> getCoeff() const noexcept;
        void train(Frame* xTrainData, Frame* yTrainData) override;
        Frame* predict(Frame* xTrainData) override;

    protected:
        int             _dimension;
        bool            _ortho{false};
        bool            _isTrained{false};

        void fit();
        virtual void convertToUblasMatrix(Frame* v,matrix<double> &m,bool isY = true);
        void transposeMatrix(matrix<double>& m);
        bool inverseMatrix(matrix<double>& m, matrix<double>& inv);
        virtual void calWeights();
        virtual void calCoeff();
        virtual void squareDueRegression();
        virtual void squareDueResidual(matrix<double> &m);
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
            this->_headerY = v->_headers[0];
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
        for(int i = 0; i < this->_coeff.size1(); i++) v.at(i) = this->_coeff(i,0);
        return v;
    }

    double PolyRegression::RSquared(){
        this->squareDueRegression();
        return ((this->_squareDueRegression / this->_totalSumOfSquare));
    }
    
    double PolyRegression::adjRSquared(){
        double r = this->RSquared();
        int n = this->_y.size1();
        int k = this->_x.size2() - 1;
        return (1 - ((1 - r * r) * ( n - 1)) / (n - k - 1));
    }

    void PolyRegression::squareDueRegression(){
        this->_squareDueRegression = 0;
        matrix<double> temp = prod(this->_x,this->_coeff);
        for(int i = 0; i < temp.size1(); i++){
            this->_squareDueRegression += (temp(i,0) - this->_mean) * (temp(i,0) - this->_mean);
            this->_totalSumOfSquare += (this->_y(i,0) - this->_mean) * (this->_y(i,0) - this->_mean);
        }
        this->squareDueResidual(temp);
    }
    void PolyRegression::squareDueResidual(matrix<double> &m){
        m = this->_y - m;
        this->_squareDueResidual = prod(trans(m),m)(0,0);
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
    Frame* PolyRegression::predict(Frame* xTestData){
        if(!this->_isTrained) return nullptr;

        auto x = Frame::cast(xTestData->at(0));

        matrix<double> xTest(x->size(),1);
        convertToUblasMatrix(xTestData,xTest);

        this->_predicM.resize(xTest.size1(),1);
        this->_errorInPredic.resize(xTest.size1(),1);

        for(int i = 0; i < xTest.size1(); i++) this->_predicM(i,0) = 0;

        for(int i = 0; i < xTest.size1(); i++){
            for(int j = 0; j < this->_coeff.size1(); j++){
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

        convertToUblasMatrix(xTrainData,this->_x,false);
        convertToUblasMatrix(yTrainData,this->_y);
        this->_label = yTrainData->getLabel(0);
        this->_mean = yTrainData->mean(0);

        this->fit();
        _isTrained = true;
    }
} // ML


#endif // POLY_REGRESSION_H
