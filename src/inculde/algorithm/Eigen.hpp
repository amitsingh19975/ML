#ifndef EIGEN_H
#define EIGEN_H
#include "../gch/headers.hpp"
#include <iomanip>

namespace ML{

    template<typename T>
    void print(boost::numeric::ublas::matrix<T>& m){
        for(int i = 0; i < m.size1(); i++){
            for(int j = 0; j < m.size2(); j++)
                std::cout<<std::setw(10)<<m(i,j)<<' ';
            puts("");
        }
    }
    // template<typename T>
    // void TransposeMultiply (const boost::numeric::ublas::vector<T>& vector, boost::numeric::ublas::matrix<T>& result,size_t size){
    //     result.resize (size,size);
    //     result.clear ();
    //     for(unsigned int row=0; row< vector.size(); ++row){
    //         for(unsigned int col=0; col < vector.size(); ++col)
    //             result(row,col) = vector(col) * vector(row);
    //         }
    // }
    // template<typename T>
    // void HouseholderCornerSubstraction (boost::numeric::ublas::matrix<T>& LeftLarge, const boost::numeric::ublas::matrix<T>& RightSmall){
    //     using namespace boost::numeric::ublas;
    //     using namespace std; 
    //     if( 
    //         !( 
    //             (LeftLarge.size1() >= RightSmall.size1())
    //         && (LeftLarge.size2() >= RightSmall.size2())
    //         ) 
    //         )
    //         {
    //         cerr << "invalid matrix dimensions" << endl;
    //         return;
    //         }  
    //     size_t row_offset = LeftLarge.size2() - RightSmall.size2();
    //     size_t col_offset = LeftLarge.size1() - RightSmall.size1();
    //     for(unsigned int row = 0; row < RightSmall.size2(); ++row )
    //         for(unsigned int col = 0; col < RightSmall.size1(); ++col )
    //             LeftLarge(col_offset+col,row_offset+row) -= RightSmall(col,row);
    // }
    // template<typename T>
    // void HouseholderQR (const boost::numeric::ublas::matrix<T>& M, boost::numeric::ublas::matrix<T>& Q, boost::numeric::ublas::matrix<T>& R){
    //     using namespace boost::numeric::ublas;
    //     using namespace std;  
    //     if( 
    //         !( 
    //             (M.size1() == M.size2())
    //         ) 
    //         )
    //         {
    //         cerr << "invalid matrix dimensions" << endl;
    //         return;
    //         }
    //     size_t size = M.size1();
    //     // init Matrices
    //     matrix<T> H, HTemp;
    //     HTemp = identity_matrix<T>(size);
    //     Q = identity_matrix<T>(size);
    //     R = M;
    //     // find Householder reflection matrices
    //     for(unsigned int col = 0; col < size-1; ++col){
    //         // create X vector
    //         boost::numeric::ublas::vector<T> RRowView = column(R,col);      
    //         vector_range< boost::numeric::ublas::vector<T> > X2 (RRowView, range (col, size));
    //         boost::numeric::ublas::vector<T> X = X2;
    //         // X -> U~
    //         if(X(0) >= 0)
    //             X(0) += norm_2(X);
    //         else
    //             X(0) += -1*norm_2(X);      
    //         HTemp.resize(X.size(),X.size(),true);
    //         TransposeMultiply(X, HTemp, X.size());
    //         // HTemp = the 2UUt part of H 
    //         HTemp *= ( 2 / inner_prod(X,X) );
    //         // H = I - 2UUt
    //         H = identity_matrix<T>(size);
    //         HouseholderCornerSubstraction(H,HTemp);
    //         // add H to Q and R
    //         Q = boost::numeric::ublas::prod(Q,H);
    //         R = boost::numeric::ublas::prod(H,R);
    //     }
    // }

    // void filter(boost::numeric::ublas::matrix<double>& m){
    //     for(int i = 0; i < m.size1(); i++){
    //         for(int j = 0; j < m.size2(); j++){
    //             if(std::abs(m(i,j)) < 1e-10) m(i,j) = 0;
    //         }
    //     }           
    // }

    // std::vector<double> eigenValues(boost::numeric::ublas::matrix<double>& m){
    //     using namespace boost::numeric::ublas;
    //     boost::numeric::ublas::matrix<double> Q,R;
    //     std::vector<double> eigenValuesVector;
    //     for(int i = 0; i < 50; i++){
    //         HouseholderQR(m,Q,R);

    //         R = trans(Q);
    //         m = boost::numeric::ublas::prod(R,m);
    //         m = boost::numeric::ublas::prod(m,Q);
    //     }
    //     filter(m);

    //     for(int i = 0; i < m.size1(); i++)
    //         eigenValuesVector.push_back(m(i,i));

    //     return eigenValuesVector;
    // }

    std::pair<std::vector<double>,std::vector<std::vector<double>>> EigenValuesVectors(Eigen::MatrixXd& M){
        using namespace Eigen;
        EigenSolver<MatrixXd> E(M);
        auto val = E.eigenvalues();
        auto vec = E.eigenvectors();
        std::vector<std::vector<double>> vectors(vec.rows(),std::vector<double>(vec.cols()));
        std::vector<double> values(val.size());
        
        for(int i = 0; i < val.size(); i++){
            values[i] = val(i).real();
        }
        for(int i = 0; i < vec.rows(); i++){    
            for(int j = 0; j < vec.cols(); j++){
                vectors[i][j] = vec(i,j).real();
            }
        }
        return {values,vectors};
    }

    std::pair<std::vector<double>,std::vector<std::vector<double>>> EigenValuesVectors(std::vector<std::vector<double>>& M){
        using namespace Eigen;
        MatrixXd m(M.size(),M.size());
        for(int i = 0; i < M.size(); i++){
            for(int j = 0; j < M.size(); j++){
                m(i,j) = M[i][j];
            }
        }
        auto [val,vec] = EigenValuesVectors(m);
        return {val,vec};
    }

    std::pair<std::vector<double>,std::vector<std::vector<double>>> EigenValuesVectors(boost::numeric::ublas::matrix<double>& M){
        using namespace Eigen;
        MatrixXd m(M.size1(),M.size1());
        for(int i = 0; i < M.size1(); i++){
            for(int j = 0; j < M.size1(); j++){
                m(i,j) = M(i,j);
            }
        }
        auto [val,vec] = EigenValuesVectors(m);
        return {val,vec};
    }
}

#endif //EIGEN_H