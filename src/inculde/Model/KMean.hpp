#ifndef K_MEAN_H
#define K_MEAN_H

#include "core.hpp"

namespace ML{
    struct KMean : Core{
        Eigen::MatrixXd  _clustors;
        Eigen::MatrixXd  _result;
        int              _k;
        int              _numberOfIterations;
        size_t           _random_seed;

        KMean(int k = 3, size_t random_seed = 42, int numberOfIterations = 10):_k(k),_random_seed(random_seed),_numberOfIterations(numberOfIterations){}

        void train(Frame* xTrainData) override;
        Frame* predict(Frame* xTestData) override;
    protected:
        int randomIdx(size_t random_seed) const noexcept;
        double mean();
        double distanceCal(Frame* f, Eigen::MatrixXd& clus, size_t j, size_t k){
            double y = 0;
            for(size_t i = 0; i < f->colSize(); i++){
                y += (f->at(j,i) - clus(k,i)) * (f->at(j,i) - clus(k,i));
            }
            return std::sqrt(y);
        }
        int nearest(Frame* f, Eigen::MatrixXd& clus,std::vector<size_t>& d);
    };
    void KMean::train(Frame* xTrainData){
        this->_clustors.resize(this->_k,xTrainData->colSize());
        _clustors.setZero();
        Eigen::MatrixXd m(this->_k,xTrainData->colSize());
        this->_label = xTrainData->getLabel(1);
        
        for(int i = 0; i < this->_k; i++) {
            int idx = this->randomIdx(_random_seed);
            for(int j = 0; j < xTrainData->colSize(); j++)
                m(i,j) = xTrainData->at(idx,j);
        }
        
        std::vector<size_t> assignment(xTrainData->rowSize(),0);

        for(size_t i = 0; i < this->_numberOfIterations; i++){
            for(size_t j = 0; j < xTrainData->rowSize(); j++){
                double bestDis = std::numeric_limits<double>::max();
                size_t bestClus = 0;

                for(size_t clus = 0; clus < this->_k;clus++){
                    double const distance = distanceCal(xTrainData,m,j,clus);

                    if(distance < bestDis){
                        bestDis = distance;
                        bestClus = clus;
                    }
                }
                assignment.at(j) = bestClus;
            }

            Eigen::MatrixXd new_mean(this->_k,xTrainData->colSize());
            new_mean.setZero();
            
            std::vector<size_t> counts(this->_k,0);

            for(size_t p = 0; p < xTrainData->rowSize(); p++){
                auto const clus = assignment.at(p);
                for(int i = 0; i < xTrainData->colSize(); i++){
                    new_mean(clus,i) += xTrainData->at(p,i); 
                    counts[clus] ++;
                }
            }

            // for(int i = 0; i < this->_k; i++){
            //     std::cout<<counts.at(i)<<'\n';
            // }

            for(size_t clus = 0; clus < this->_k; clus++){
                auto const count = std::max<size_t>(1, counts[clus]);
                for(int i = 0; i < xTrainData->colSize(); i++)
                    this->_clustors(clus,i) += xTrainData->at(clus,i)/count;  
            }
        }
        
        std::cout<<this->_clustors<<'\n';
    }
    Frame* KMean::predict(Frame* xTestData){
        

        return nullptr;

    }

    int KMean::randomIdx(size_t random_seed) const noexcept{
        // auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        // std::mt19937 gen(seed);
        // std::uniform_int_distribution<int> distribution(0,this->_clustors.cols() -1);
        std::mt19937 random_number_generator(random_seed);
        std::uniform_int_distribution<size_t> indices(0, this->_clustors.rows() - 1);
        return indices(random_number_generator);
    }

    int nearest(Frame* f, Eigen::MatrixXd& clus,std::vector<size_t>& d){
        // int min_i = 0;
        // for(int i = 0;)
        return 0;
    }


} // ML


#endif // K_MEAN_H
