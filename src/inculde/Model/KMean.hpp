#ifndef K_MEAN_H
#define K_MEAN_H

#include "core.hpp"

namespace ML{
    using namespace boost::numeric::ublas;
    struct KMean : Core{

        matrix<double>  _clustors;
        matrix<double>  _result;
        int             _k;
        int             _numberOfIterations;

        KMean(int k = 1, int numberOfIterations = 10):_k(k),_numberOfIterations(numberOfIterations){}

        void train(Frame* xTrainData, Frame* yTrainData) override;
        void test(Frame* xTestData, Frame* yTestData) override;
    protected:
        int randomIdx() const noexcept;
        double mean();
        double distanceCal(double x1, double y1, double x2, double y2){
            return std::pow((x1 -x2),2) + std::pow((y1 -y2),2);
        }
    };

    void KMean::train(Frame* xTrainData, Frame* yTrainData){
        this->_clustors.resize(this->_k,2);
        auto x = Frame::cast(xTrainData->at(0));
        auto y = Frame::cast(yTrainData->at(1));
        matrix<double> m(this->_k,2);
        
        for(int i = 0; i < this->_k; i++) {
            int idx = this->randomIdx();
            m(i,0) = x->_data[idx];
            m(i,1) = y->_data[idx];

            this->_clustors(i,0) = 0;
            this->_clustors(i,1) = 0;
        }
        
        std::vector<size_t> assignment(xTrainData->at(0)->size(),0);

        for(size_t i = 0; i < this->_numberOfIterations; i++){
            for(size_t j = 0; j < x->size(); j++){
                double bestDis = std::numeric_limits<double>::max();
                size_t bestClus = 0;

                for(size_t clus = 0; clus < this->_k;clus++){
                    double const distance = distanceCal(x->_data.at(j),
                    y->_data.at(j),m(clus,0),m(clus,1));

                    if(distance < bestDis){
                        bestDis = distance;
                        bestClus = clus;
                    }
                }
                assignment.at(j) = bestClus;
            }

            matrix<double> new_mean(this->_k,2);
            for(int i = 0; i < this->_k; i++){
                new_mean(i,0) = 0;
                new_mean(i,1) = 0;
            }
            
            std::vector<size_t> counts(this->_k,0);

            for(size_t p = 0; p < x->size(); p++){
                auto const clus = assignment.at(p);
                new_mean(clus,0) += x->_data.at(p); 
                new_mean(clus,1) += y->_data.at(p); 
                counts[clus] ++;
            }

            for(int i = 0; i < this->_k; i++){
                std::cout<<counts.at(i)<<'\n';
            }

            for(size_t clus = 0; clus < this->_k; clus++){
                auto const count = std::max<size_t>(1, counts[clus]);
                this->_clustors(clus,0) += x->_data[clus]/count; 
                this->_clustors(clus,1) += y->_data[clus]/count; 
            }
        }
        
        std::cout<<this->_clustors<<'\n';
    }
    void KMean::test(Frame* xTestData, Frame* yTestData){
        auto x = Frame::cast(xTestData->at(0));
        auto y = Frame::cast(yTestData->at(1));

        matrix<double> count(this->_k,1);

        for(int  i = 0 ; i < this->_k; i++) count(i,0) = 0;

        for(size_t j = 0; j < x->size(); j++){
                double bestDis = std::numeric_limits<double>::max();
                size_t bestClus = 0;

                for(size_t clus = 0; clus < this->_k;clus++){
                    double const distance = distanceCal(x->_data.at(j),
                    y->_data.at(j),this->_clustors(clus,0),this->_clustors(clus,1));

                    if(distance < bestDis){
                        bestDis = distance;
                        bestClus = clus;
                    }
                }
            count(bestClus,0)++;
        }

        std::cout<<count<<'\n';

    }

    int KMean::randomIdx() const noexcept{
        // auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        // std::mt19937 gen(seed);
        // std::uniform_int_distribution<int> distribution(0,this->_clustors.size2() -1);
        static std::random_device seed;
        static std::mt19937 random_number_generator(seed());
        std::uniform_int_distribution<size_t> indices(0, this->_clustors.size1() - 1);
        return indices(random_number_generator);
    }

} // ML


#endif // K_MEAN_H
