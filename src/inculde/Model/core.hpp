#ifndef CORE_H
#define CORE_H

#include "../Frame/frame.hpp"
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <unordered_set>
#include <chrono>
#include <random>
#include <iostream>

namespace ML{
    struct Core{
        virtual void train(Frame* xTrainData, Frame* yTrainData) = 0;
        virtual void test(Frame* xTestData, Frame* yTestData) = 0;
        size_t _trueP{0};
        size_t _falseP{0};
        size_t _trueN{0};
        size_t _falseN{0};
        void confusionMatrix() const noexcept;
    };

    void Core::confusionMatrix() const noexcept{
        for(int i = 0; i < 50; i++) std::cout<<'-';
        puts("");
        std::cout<<"True Postive: "<<_trueP<<'\n'<<"False Negative: "<<_falseN<<'\n';
        std::cout<<"True Negative: "<<_trueN<<'\n'<<"False Postive: "<<_falseP<<'\n';
        for(int i = 0; i < 50; i++) std::cout<<'-';
        puts("");
    }
}


#endif // CORE_H

