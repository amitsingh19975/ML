#ifndef POLY_DER_H
#define POLY_DER_H
#include "../gch/headers.hpp"

namespace ML{
    using namespace boost::numeric::ublas;
    struct Polynomial{
        std::vector<double> _coeff;
        size_t _degree{0};
        Polynomial(Eigen::MatrixXd& m){
            if(m.rows() > m.cols() && m.cols() == 1){
                _degree = m.rows();
                _coeff.resize(_degree);
                for(auto i = 0; i <_degree; i++){
                    _coeff[i] = m(i,0);
                }
            }else if(m.rows() < m.cols() && m.rows() == 1){
                _degree = m.rows();
                _coeff.resize(_degree);
                for(auto i = 0; i <_degree; i++){
                    _coeff[i] = m(0,i);
                }
            }else if(m.rows() == m.cols() && m.rows() == 1){
                _degree = 1;
                _coeff.resize(_degree);
                _coeff[0] = m(0,0);
            }
        }
        Polynomial(size_t degree, std::vector<double>& coeff):_degree(degree),_coeff(std::move(coeff)){}
        Polynomial(std::vector<double>& coeff):_coeff(std::move(coeff)){
            _degree = this->_coeff.size() - 1;
        }
        Polynomial(size_t degree):_degree(degree),_coeff(degree + 1){}
        Polynomial(){}
        size_t size() const noexcept{return _coeff.size();}
        size_t degree() const noexcept{return _degree;}
        void print() const noexcept{
            for(int i = 0; i < _coeff.size(); i++){
                if(i == 0){
                    if(_coeff.size() == 1){
                        printf("%f ",_coeff[i]);
                    }else if(_coeff[i] == 1){
                        printf("x^%lu ",_degree - i);
                    }else{
                        printf("%fx^%lu ",_coeff[i],_degree - i);
                    }
                }else if(i != _coeff.size() - 1){
                    if(_coeff[i] < 0){
                        printf("- %fx^%lu ",std::abs(_coeff[i]),_degree - i);
                    }else{
                        printf("+ %fx^%lu ",std::abs(_coeff[i]),_degree - i);
                    }
                }else{
                    if(_coeff[i] < 0){
                        printf("- %f ",std::abs(_coeff[i]));
                    }else{
                        printf("+ %f ",std::abs(_coeff[i]));
                    }
                }
            }
        }
        double& operator[](size_t i){
            return _coeff[i % size()];
        }
        std::complex<double> eval(std::complex<double> x){
            std::complex<double> c(0,0);
            for(int i = 0; i < size(); i++){
                c += _coeff[i] * std::pow(x,_degree - i);
            }
            return c;
        }
        double eval(double x){
            double sum = 0;
            for(int i = 0; i < size(); i++){
                sum += _coeff[i] * std::pow(x,_degree - i);
            }
            return sum;
        }
    };

    struct PolynomialDerivative{
        static void differentiate(Polynomial& poly){
            Polynomial p(poly.degree() - 1);
            for(int i = 0; i < poly.degree();i++){
                p[i] = (poly.degree() - i) * poly[i];
            }
            poly = std::move(p);
        }
    };
}


#endif // PolyDer