#ifndef FRAME_HPP
#define FRAME_HPP

#include <iostream>
#include <string>
#include <vector>
#include "../Series/vec.hpp"
#include <chrono>
#include <random>

namespace ML{
    struct Frame{
        std::vector<std::unique_ptr<Series>> _data;
        std::vector<std::string> _headers;

        size_t _rows{0};
        size_t _cols{0};

        Frame(){}
        Frame(size_t row, size_t col):_rows(row),_cols(col){}
        Frame(std::vector<std::string>& h): _headers(h){}
        Frame(std::vector<std::unique_ptr<Series>> data):_cols(data.size()),_rows(data.at(0)->size()),_data(std::move(data)){}
        Frame(std::unique_ptr<Series> data, std::string header = ""){
            this->_data.push_back(std::move(data));
            this->_headers.push_back(header);
            this->_cols = this->_data.size();
            this->_rows = this->_data.at(0)->size();
        }
        Frame(std::vector<std::unique_ptr<Series>> data,
        std::vector<std::string> _header):_cols(data.size()),_rows(data.at(0)->size()),_data(std::move(data)),_headers(_header){}

        template<typename T>
        Frame(std::vector<std::vector<T>>& data);
        template<typename T>
        Frame(std::vector<T>& data, std::string header = "");
        template<typename T>
        Frame(std::initializer_list<std::initializer_list<T>> data);

        void print(int size = -1, int ind = 5) const noexcept;
        bool addSeries(std::unique_ptr<Series> series);
        bool addSeries(Series* series);
        template<typename T> bool addVec(Vec<T>* vec);
        template<typename U = double> U at(int i, int j);
        Series* at(int i);
        void labelToNumber(int idx);
        void numberToLabel(int idx);
        void apply(std::function<double(double)> func);

        Series* operator[](int i){return this->_data.at(i % this->_cols).get();}
        std::shared_ptr<Frame> operator[](std::initializer_list<std::string> l){
            return std::move(this->colSlice(l));
        }
        Series* operator[](std::string_view header){
            if( auto it = std::find(this->_headers.begin(),this->_headers.end(),header);
                it == this->_headers.end()){
                std::cerr<< "Line: " + std::to_string(__LINE__)<<'\n';
                exit(1);
            }else{
                auto idx = std::distance(this->_headers.begin(),it);
                return this->_data.at(idx).get();
            }
        }
        size_t size() const noexcept{return this->_data.size();}

        std::unique_ptr<Series> dropCol(int idx); 
        std::shared_ptr<Frame> colSlice(size_t start, size_t end = std::numeric_limits<size_t>::max()); 
        std::shared_ptr<Frame> colSlice(std::vector<int> l); 
        std::shared_ptr<Frame> colSlice(std::initializer_list<std::string> l); 
        std::vector<std::unique_ptr<Series>> dropRow(int idx); 
        std::pair<Frame*, Frame*> split(float percentage = 30, bool randomized = true); 
        static void randomize(Frame* frame, int iteration = 50); 

        double mean(int col){return this->at(col)->mean();}
        double std(int col){return this->at(col)->std();}
        double variance(int col){return this->at(col)->variance();}
        double median(int col){return this->at(col)->median();}
        void normalize(double val = 0);
        double corrcoef(int i, int j);
        
        template<typename T = double>
        std::unique_ptr<std::unordered_map<T, int>> unique(int col);

        template<typename U = double>
        static Vec<U>* cast(Series* data){
            return static_cast<Vec<U>*>(data);
        }
    protected:
        void indentUtil(int indent = 0) const noexcept{
            for(int i = 0; i< indent; i++){
                std::cout<<' ';
            }
        }
        void updateSize() noexcept{
            this->_cols = this->_data.size();
            if(this->_cols != 0)
                this->_rows = this->_data.at(0)->size();
            else
                this->_rows = 0;
        }
        bool isEmpty() const noexcept{
            if(this->_cols == 0 || this->_rows == 0) {
                std::string str = "Out of Bound!\nRow Size: " + std::to_string(this->_rows) + "\nCol Size: " + std::to_string(this->_cols);
                std::cerr<<str<<'\n';
                return true;
            }
            return false;
        }
        std::unique_ptr<Frame> _train;
        std::unique_ptr<Frame> _test;
    };


    template<typename U>
    U Frame::at(int i,int j){
        if(isEmpty() || (i >= this->_rows || i < 0) || (j >= this->_cols || j < 0)) {
            std::cerr<< "Error!\nLine: " + std::to_string(__LINE__)<<'\n';
            exit(1);
        }else
            return this->at(j)->at(i);
    }
    Series* Frame::at(int i){
        if(isEmpty() || this->_cols <= i) {
            std::cerr<< "Line: " + std::to_string(__LINE__)<<'\n';
            exit(1);
        }
        return this->_data.at(i).get();
    }

    template<>
    std::string Frame::at<std::string>(int i,int j){
        if(isEmpty() || (i >= this->_rows || i < 0) || (j >= this->_cols || j < 0)) {
            std::cerr<< "Line: " + std::to_string(__LINE__)<<'\n';
            exit(1);
        }else return this->at(j)->atS(i);
    }

    void Frame::print(int size, int ind) const noexcept {
        if(isEmpty()) {
            std::cerr<< "Line: " + std::to_string(__LINE__)<<'\n';
            return;
        }
        size = size < - 1 || size > this->_data.at(0)->size() ? this->_data.at(0)->size() : size;

        std::vector<int> lenghtOfHeaderString(this->_headers.size());
        int i = 0;
        for(auto h : this->_headers){
            printf("%s",h.c_str());
            this->indentUtil(ind);
            lenghtOfHeaderString[i++] = h.size();
        }
        
        std::cout<<'\n';
        for(int i = 0; i < this->_headers.size();i++) std::cout<<std::string(this->_headers.size() + 5,'-');
        std::cout<<'\n';

        for(int i = 0; i < size; i++){
            for(int j = 0; j < this->_data.size();j++){
                auto d = this->_data.at(j).get();
                if(d->_type == "string"){
                    Vec<std::string> t = *cast<std::string>(d);
                    std::cout<<t.at(i);
                    indent(t[i],lenghtOfHeaderString,j,ind);
                }else{
                    Vec<double> t = *cast<double>(d);
                    std::cout<<t.at(i);
                    indent(t[i],lenghtOfHeaderString,j,ind);
                }
            }
            std::cout<<'\n';
        }
    } 

    std::unique_ptr<Series> Frame::dropCol(int idx){
        if(isEmpty()|| this->_cols < idx|| idx < 0) {
            std::cerr<< "Line: " + std::to_string(__LINE__)<<'\n';
            exit(1);
        }
        this->_headers.erase(this->_headers.begin() + idx,this->_headers.begin() + idx + 1);
        auto t = std::move(this->_data.at(idx));
        this->_data.erase(this->_data.begin() + idx,this->_data.begin() + idx + 1);
        this->updateSize();
        return std::move(t);
    } 
    std::vector<std::unique_ptr<Series>> Frame::dropRow(int idx){
        if(isEmpty()) {
            std::cerr<< "Line: " + std::to_string(__LINE__)<<'\n';
            return {};
        }
        // std::vector<std::unique_ptr<Series>> temp;
        // for(int i = 0; i < this->_data.at(0)->size(); i++){
        //     std::unique_ptr<Series> s;
        //     if(this->_data.at(i)->_type == "string") {
        //         auto t = this->cast<std::string>(this->_data.at(i).get());
        //         temp.emplace_back(std::make_unique<Vec<std::string>>(t->_header,"string",t.at(i)));
        //     }else{
        //         auto t = this->cast(this->_data.at(i).get());
        //         temp.emplace_back(std::make_unique<Vec<double>>(t->_header,"double",t.at(i)));
        //     }  
        // }
        this->updateSize();
        return {};
    }

    bool Frame::addSeries(Series* series){
        if(isEmpty() || series->size() != this->_rows) {
            std::cerr<< "Line: " + std::to_string(__LINE__)<<'\n';
            return false;
        }
        if( series == nullptr
            ||((series)->_type != "string" && (series)->_type != "double")
            || (series)->size() == 0
            || (series)->_header == "") return false;
        if((series)->_type == "string" && Frame::cast<std::string>(series) == nullptr) return false;
        if((series)->_type == "double" && Frame::cast(series) == nullptr) return false;

        if(series->_type == "string"){
            std::unique_ptr<Series> s(new Vec<std::string>(series->_header,"string"));
            for(int i = 0; i < series->size(); i++){
                s->push_s(series->atS(i));
            }
            this->_headers.push_back((s)->_header);
            this->_data.push_back(std::move(s));
        }else{
            std::unique_ptr<Series> s(new Vec<double>(series->_header,"double"));
            for(int i = 0; i < series->size(); i++){
                s->push_d(series->at(i));
            }
            this->_headers.push_back((s)->_header);
            this->_data.push_back(std::move(s));
        }
        this->updateSize();
        return true;
    }
    bool Frame::addSeries(std::unique_ptr<Series> series){
        if(isEmpty()) {
            std::cerr<< "Line: " + std::to_string(__LINE__)<<'\n';
            return false;
        }
        if( series == nullptr
            ||((series)->_type != "string" && (series)->_type != "double")
            || (series)->size() == 0
            || (series)->_header == "") return false;
        if((series)->_type == "string" && Frame::cast<std::string>(series.get()) == nullptr) return false;
        if((series)->_type == "double" && Frame::cast(series.get()) == nullptr) return false;

        this->_headers.push_back((series)->_header);
        this->_data.push_back(std::move(series));
        this->updateSize();
        return true;
    }

    template<typename T>
    Frame::Frame(std::vector<T>& data, std::string header){
        if constexpr(std::is_same_v<T,std::string>){
            std::unique_ptr<Series> s(new Vec<T>(header,"string",data));
            this->_data.push_back(std::move(s));
            this->_headers.push_back((header));
        }else{
            std::unique_ptr<Series> s(new Vec<T>(header,"double",data));
            this->_data.push_back(std::move(s));
            this->_headers.push_back((header));
        }
        this->updateSize();
    }
    template<typename T>
    Frame::Frame(std::vector<std::vector<T>>& data){
        int i = 0;
        for(auto v : data){
            if constexpr(std::is_same_v<T,std::string>){
                std::unique_ptr<Series> s(new Vec<T>(std::to_string(i),"string",v));
                this->_data.push_back(std::move(s));
                this->_headers.push_back(std::to_string(i++));
            }else{
                std::unique_ptr<Series> s(new Vec<T>(std::to_string(i),"double",v));
                this->_data.push_back(std::move(s));
                this->_headers.push_back(std::to_string(i++));
            }
        }
        this->updateSize();
    }
    template<typename T>
    Frame::Frame(std::initializer_list<std::initializer_list<T>> data){

    }

    void Frame::labelToNumber(int idx){
        if(isEmpty()) {
            std::cerr<< "Line: " + std::to_string(__LINE__)<<'\n';
            return ;
        }
        if(this->_data[idx]->_type != "string") return;
        int i = 0, j = 0;
        std::unique_ptr<Series> newCol(new Vec<double>(this->_headers[idx],"double",this->_data[idx]->size()));
        auto newColVec = Frame::cast(newCol.get());
        for(auto key : Frame::cast<std::string>(this->_data[idx].get())->_data){
            if(newCol->_labelMap.find(key) == newCol->_labelMap.end()){
                newCol->_labelMap[key] = i++;
            }
            newColVec->_data[j++] = (newCol->_labelMap[key]);
        }

        this->_data[idx] = std::move(newCol);
    }
    void Frame::numberToLabel(int idx){
        if(isEmpty()) {
            std::cerr<< "Line: " + std::to_string(__LINE__)<<'\n';
            return;
        }
        if(this->_data[idx]->_type != "double" && this->_data[idx]->_labelMap.size() == 0) return;
        
        std::unique_ptr<Series> newCol(new Vec<std::string>(this->_headers[idx],"string",this->_data[idx]->size()));
        auto newColVec = Frame::cast<std::string>(newCol.get());
        auto oldColVec = Frame::cast(this->_data[idx].get());

        std::unordered_map<int,std::string> temp;

        for(auto [key, val] : this->_data[idx]->_labelMap){
            temp[val] = key;
        }

        int j = 0;
        for(size_t i = 0; i < this->_data[idx]->size(); i++){
            if(auto it = temp.find(oldColVec->_data.at(i)); it != temp.end()){
                newColVec->_data.at(i) = it->second; 
            }
        }
        this->_data[idx] = std::move(newCol);
    }

    void Frame::normalize(double val){
        if(isEmpty()) {
            std::cerr<< "Line: " + std::to_string(__LINE__)<<'\n';
            return;
        }
        if(val == 0){
            for(auto const& v: this->_data){
                double max = v->max();
                double min = v->min();
                v->apply([&](double el){
                    return (el - max) / (max - min);
                });
            }
        }else{
            for(auto const& v: this->_data){
                double max = v->max();
                if(max != 0){
                    v->apply([&](double el){
                        return el / val;
                    });
                }
            }
        }
    }

    std::pair<Frame*, Frame*> Frame::split(float percentage, bool randomized){
        if(isEmpty()) {
            std::cerr<< "Line: " + std::to_string(__LINE__)<<'\n';
            exit(1);
        }

        if(randomized) Frame::randomize(this);
        size_t idx = std::ceil((float)((percentage * this->_rows)/100.0));
        _train = std::make_unique<Frame>(this->_headers);
        _test = std::make_unique<Frame>(this->_headers);

        for(size_t i = 0; i < this->_cols; i++){
            if(this->_data.at(i)->_type == "string"){
                std::unique_ptr<Series> temp(new Vec<std::string>(this->_headers.at(i),"string"));
                for(size_t j = 0; j < idx; j++){
                    temp->push_s(this->at<std::string>(j,i));
                }
                temp->_labelMap = this->at(i)->_labelMap;
                _train->_data.push_back(std::move(temp));
            }else{
                std::unique_ptr<Series> temp(new Vec<double>(this->_headers.at(i),"double"));
                for(size_t j = 0; j < idx; j++){
                    temp->push_d(this->at(j,i));
                }
                temp->_labelMap = this->at(i)->_labelMap;
                _train->_data.push_back(std::move(temp));
            }
            _train->_cols = _train->_data.size();
            _train->_rows = _train->_data.at(0)->size();
        }
        for(size_t i = 0; i < this->_cols; i++){
            if(this->_data.at(i)->_type == "string"){
                std::unique_ptr<Series> temp(new Vec<std::string>(this->_headers.at(i),"string"));
                for(size_t j = idx; j < this->_rows; j++){
                    temp->push_s(this->at<std::string>(j,i));
                    // std::cout<<j<<'\n';
                }
                temp->_labelMap = this->at(i)->_labelMap;
                _test->_data.push_back(std::move(temp));
            }else{
                std::unique_ptr<Series> temp(new Vec<double>(this->_headers.at(i),"double"));
                for(size_t j = idx; j < this->_rows; j++){
                    temp->push_d(this->at(j,i));
                }
                temp->_labelMap = this->at(i)->_labelMap;
                _test->_data.push_back(std::move(temp));
            }
            _test->updateSize();
            _train->updateSize();
        }
        
        return {this->_train.get(),this->_test.get()};
    }

    void Frame::randomize(Frame* frame, int iteration){
        if(frame->isEmpty()) {
            std::cerr<< "Line: " + std::to_string(__LINE__)<<'\n';
            return;
        }
        auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        std::mt19937 gen(seed);
        std::uniform_int_distribution<size_t> distribution(0,frame->_rows - 1);
        auto idxGen = [&](){
            return distribution(gen);
        };

        for(size_t i = 0; i < iteration; i++){
            size_t idxA = idxGen();
            size_t idxB = idxGen();
            if(idxA == idxB) {
                i--;
                continue;
            }

            for(size_t j = 0; j < frame->_cols; j++){
                frame->at(j)->swap(idxA, idxB);
            }
        }
    }

    std::shared_ptr<Frame> Frame::colSlice(size_t start, size_t end){
        if(isEmpty() || this->_cols <= start || start < 0) {
            std::cerr<< "Line: " + std::to_string(__LINE__)<<'\n';
            exit(1);
        }
        std::unique_ptr<Frame> f(new Frame(this->_rows,std::min(end,this->_cols) - start));
        if(end == std::numeric_limits<size_t>::max()){
            size_t s = start;
            while(this->_cols > s ){
                if(!f->addSeries(std::move(this->dropCol(start)))) break;
            }
        }else{
            size_t s = start;
            while(s++ != end ){
                if(!f->addSeries(std::move(this->dropCol(start)))) break;
            }
        }
        return std::move(f);
    } 
    std::shared_ptr<Frame> Frame::colSlice(std::vector<int> l){
        
        for(auto num: l){
            if(num >= this->_cols || num <0){
                std::cerr<< "Line: " + std::to_string(__LINE__)<<'\n';
            }
        }
        if(isEmpty()) {
            std::cerr<< "Line: " + std::to_string(__LINE__)<<'\n';
            exit(1);
        }
        std::unique_ptr<Frame> f(new Frame(this->_rows, l.size()));
        for(auto i : l){
            f->addSeries(this->at(i));
        }
        std::sort(l.begin(),l.end(),std::greater<int>());

        int j = 0;
        for(auto i : l){
            this->dropCol(i);
        }
        
        return std::move(f);
    } 
    std::shared_ptr<Frame> Frame::colSlice(std::initializer_list<std::string> l){
        std::vector<int> v;
        for(std::string const& num: l){
            auto it = std::find(this->_headers.begin(),this->_headers.end(),num.c_str());
            if(it == this->_headers.end()){
                std::cerr<< "Line: " + std::to_string(__LINE__)<<'\n';
                exit(1);
            }else{
                v.push_back(std::distance(this->_headers.begin(),it));
            }
        }
        if(isEmpty()) {
            std::cerr<< "Line: " + std::to_string(__LINE__)<<'\n';
            exit(1);
        }
        return std::move(colSlice(v));
        return {};
    } 

    void Frame::apply(std::function<double(double)> func){
        if(isEmpty()) {
            std::cerr<< "Line: " + std::to_string(__LINE__)<<'\n';
            return;
        }
        for(int i = 0; i < this->_cols; i++){
            if(this->_data.at(i)->_type != "string")
                this->_data.at(i)->apply(func);
        }
    }

    double Frame::corrcoef(int i, int j){
        if(this->_cols <= i || this->_cols <= j ){
            std::cerr<< "OutOfBound: " + std::to_string(__LINE__)<<'\n';
            return 0;
        }
        double ux = this->_data.at(i)->mean();
        double uy = this->_data.at(j)->mean();
        double sx = this->_data.at(i)->stdS();
        double sy = this->_data.at(j)->stdS();
        double t = 0;
        for(int k = 0; k < this->_rows; k++){
            t += ((this->at(k,i) - ux)/sx) * ((this->at(k,j) - uy) / sy);
        }
        return t / (this->_rows - 1);
    }
}

#endif