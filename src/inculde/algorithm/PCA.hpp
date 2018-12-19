#ifndef PCA_H
#define PCA_H

#include "../gch/headers.hpp"
#include "../Frame/frame.hpp"
#include "../PrettyPrint/prettyPrint.hpp"

namespace ML{

    double round_nplaces(double value, int to){
        double places = pow(10.0, to);
        return round(value * places) / places;
    }

    Eigen::MatrixXd removeColumn(Eigen::MatrixXd& matrix, unsigned int upper){
        unsigned int numRows = matrix.rows();
        unsigned int numCols = upper;
        Eigen::MatrixXd m(numRows,numCols);
        for(int i = 0; i < m.rows(); i++)
            for(int j = 0; j < m.cols();j++)
                m(i,j) = matrix(i,j);
        return m;
    }

    struct PCA{
        Eigen::EigenSolver<Eigen::MatrixXd> _eigenSolver;
        size_t _n_components{2};
        Eigen::MatrixXd _f;
        PCA(FrameShared& f,size_t n_components = 2):_n_components(n_components),_f(f->rowSize(),f->colSize()){
            f->Zscore();
            Eigen::MatrixXd sampleVar = f->cov();
            for(auto i = 0; i < _f.rows(); i++){
                for(auto j = 0; j < _f.cols(); j++){
                    _f(i,j) = f->at(i,j);
                }
            }
            _eigenSolver = Eigen::EigenSolver<Eigen::MatrixXd>(sampleVar);
            _n_components = n_components;
            for(int i = 0; i < f->colSize();i++){
                 _mean.push_back(f->mean(i));
                 _std.push_back(f->std(i));
            }
            this->init();
        }
        FrameShared getReducedFrame() noexcept;

    protected:
        double cov(Series* x, Series* y);
        Eigen::VectorXd _val;
        Eigen::MatrixXd _vec;
        std::vector<double> _mean;
        std::vector<double> _std;
        void init();
        void sort(int low, int high);
        int partition(int low, int high);
        void swapM(int i, int j);
    };

    FrameShared PCA::getReducedFrame() noexcept {
        FrameUnique f(new Frame(_f.rows(),_n_components));
        auto vec = removeColumn(_vec,_n_components);
        Eigen::MatrixXd t = _f * vec;
        for(auto i = 0; i < f->colSize();i++){
            std::string header = std::to_string(i + 1) +" P";
            SeriesUnique temp(new Vec<double>(header,"double"));
            for(auto j = 0; j < f->rowSize();j++){
                temp->push_d(t(j,i));
            }
            f->_headers.push_back(header);
            f->_data.push_back(std::move(temp));
        }
        return std::move(f);
    }
    void PCA::init(){
        auto val = _eigenSolver.eigenvalues();
        auto vec = _eigenSolver.eigenvectors();
        _vec.resize(vec.rows(),vec.cols());
        _val.resize(val.size());
        
        for(auto i = 0; i < val.size(); i++){
            _val(i) = val(i).real();
        }
        for(auto i = 0; i < vec.rows(); i++){    
            for(int j = 0; j < vec.cols(); j++){
                _vec(i,j) = vec(i,j).real();
            }
        }
        // _vec *= -1;
        sort(0,_val.size() - 1);
    }

    void PCA::sort(int low, int high){
        if (low < high) { 
            int pi = partition(low, high); 
            sort(low, pi - 1); 
            sort(pi + 1, high); 
        }
    }
    int PCA::partition (int low, int high) { 
        double pivot = _val(high);
        int i = (low - 1);
    
        for (int j = low; j <= high- 1; j++) { 
            if (_val(j) >= pivot) { 
                i++;
                std::swap(_val(i), _val(j)); 
                swapM(i, j); 
            } 
        } 
        std::swap(_val(i + 1), _val(high)); 
        swapM(i + 1, high); 
        return (i + 1); 
    } 
    void PCA::swapM(int i, int j){
        for(int k = 0; k < _vec.rows(); k++){
            std::swap(_vec(k,i),_vec(k,j));
        }
    }
}


#endif