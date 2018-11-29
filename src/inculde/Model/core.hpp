#ifndef CORE_H
#define CORE_H

#include "../Frame/frame.hpp"
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <unordered_set>
#include <chrono>
#include <random>

namespace ML{
    struct Core{
        virtual void train(Frame* xTrainData, Frame* yTrainData) = 0;
        virtual void test(Frame* xTestData, Frame* yTestData) = 0;
    };
}


#endif // CORE_H

