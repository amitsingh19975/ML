#ifndef CORE_H
#define CORE_H

#include "../Frame/frame.hpp"

namespace ML{
    //Core methods and functions that every machine learning class or struct has to provide 
    struct Core{
        //regression coefficents
        matrix<double>  _coeff;
        double          _squareDueResidual{0};
        double          _squareDueRegression{0};
        double          _totalSumOfSquare{0};
        double          _mean{0};
        matrix<double>  _predicM;
        FrameUnique     _predic;
        virtual void train(Frame* xTrainData, Frame* yTrainData){}
        virtual void train(Frame* xTrainData){}
        //return predicted frame
        virtual Frame* predict(Frame* xTestData){return nullptr;}
        void train(std::shared_ptr<Frame>& xTrainData, std::shared_ptr<Frame>& yTrainData){this->train(xTrainData.get(),yTrainData.get());}
        void train(std::shared_ptr<Frame>& xTrainData){this->train(xTrainData.get());}
        //return predicted frame
        Frame* predict(std::shared_ptr<Frame>& xTestData){return this->predict(xTestData.get());}
        //init _predic
        void setPredict() noexcept;

    protected:
        matrix<double>                      _x;
        matrix<double>                      _y;
        std::string                         _headerY;
        std::unordered_map<std::string,int> _label;
    };
    
    void Core::setPredict() noexcept{
        FrameUnique t(new Frame(this->_predicM,_label,{_headerY}));
        this->_predic = std::move(t);
    }

    struct Metrics{
        matrix<size_t> _data;
        size_t _total{1};

        Metrics(Frame* y_test, Frame* y_predic){
            auto labelT = y_test->getLabel(0);
            if(labelT.size() == 0) throw std::runtime_error("Not a Classification Model!");
            
            auto labelP = y_predic->getLabel(0);
            _data.resize(labelT.size(),labelT.size());
            //set _data values to 0
            for(size_t i = 0; i < _data.size1(); i++)
                for(size_t j = 0; j < _data.size2(); j++)
                    _data(i,j) = 0;
            //set the confusion matrix
            for(int i = 0; i < y_test->_rows; i++){
                _data(y_test->at(i,0),y_predic->at(i,0))++;
            }
            //set the total row size
            _total = y_test->_rows;
        }

        void confusionMatrix() const noexcept;
        void listRates() const noexcept;
    };

    void Metrics::confusionMatrix() const noexcept{
            for(int i = 0; i < 50; i++) std::cout<<'-';
            puts("");
            puts("[ ");
            for(auto i = 0; i < _data.size1(); i++){
                for(auto j = 0; j < _data.size2(); j++){
                    std::cout<<"  "<<_data(i,j);
                    if(j != _data.size2() - 1) std::cout<<",";
                }
                puts("");
            }
            puts("]");
            for(int i = 0; i < 50; i++) std::cout<<'-';
            puts("");
    }
    void Metrics::listRates() const noexcept{
        puts("Rates");
        for(int i = 0; i < 50; i++) std::cout<<'-';
        puts("");
        double t = 0;
        for(auto i = 0; i < _data.size1(); i++) t += _data(i,i);
        double r = ((_data(0,0))/(t * 1.0));
        double p = ((_data(0,0) * 1.0)/(_data(0,0) + _data(1,0)));

        std::cout<<"Accuracy: " << t/(_total * 1.0)<<'\n';
        
        std::cout<<"Recall: " <<r<<'\n';
        for(auto i = 0; i < _data.size1(); i++) t += _data(i,i);

        std::cout<<"Sensitivity: " << ((_data(0,0) * 1.0)/(_data(1,0) + _data(0,0)))<<'\n';
        std::cout<<"False Positive Rate: " << ((_data(1,0) * 1.0)/(_data(1,0) + _data(1,1)))<<'\n';
        std::cout<<"Specificity: " << ((_data(1,1) * 1.0)/(_data(1,1) + _data(1,0)))<<'\n';
        std::cout<<"Precision: " <<p<<'\n';
        std::cout<<"F-measure: " << (2 * (r * p)/(r+p))<<'\n';
        // std::cout<<"Prevalence: " << ((_falseN + _trueP)/(_total * 1.0))<<'\n';
        for(int i = 0; i < 50; i++) std::cout<<'-';
        puts("");
    }
}


#endif // CORE_H

