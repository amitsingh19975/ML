#ifndef POLY_ROOT_H
#define POLY_ROOT_H
#include "../gch/headers.hpp"
#include "PolynomialDerivative.hpp"

namespace ML{
    using namespace boost::numeric::ublas;
    template<typename T>
    void print(std::vector<T>& v){
        for(auto const& c : v){
            std::cout<<c<<'\n';
        }
    }
    void findRoots(Polynomial& poly){
        std::vector<std::complex<double>> roots(poly.degree());
        std::complex c = {0.4,0.9};
        for(int i = 0; i < roots.size(); i++){
            roots[i] = std::pow(c,i);
        }
        auto calDen = [](std::vector<std::complex<double>>& c,int i){
            std::complex p = {1.0,1.0};
            for(auto j = 0; j < c.size(); j++){
                if(i != j){
                    p *= (c[i] - c[j]);
                }
            }
            return p;
        };

        for(int i = 0; i < 100; i++){
            std::vector prev = roots;
            for(int  j = 0; j < roots.size(); j ++){
                auto temp = poly.eval(prev[j]);
                temp /= calDen(prev,j);
                roots[j] = prev[j] - temp;
            }
        }
        for(int i = 0; i < roots.size(); i++){
            if(std::abs(roots[i].imag()) < 1e-10) roots[i].imag(0);
            if(std::abs(roots[i].real()) < 1e-10) roots[i].real(0);
        }
        print(roots);
    }
}
#endif // PolyRoot