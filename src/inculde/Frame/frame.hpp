#ifndef FRAME_HPP
#define FRAME_HPP

#include "../Series/vec.hpp"
#include <iomanip>

namespace ML{
    struct Frame;
    using Tuple         = std::tuple<Frame*, Frame*, Frame*, Frame*>;
    using Pair          = std::pair<Frame*, Frame*>;
    using FrameShared   = std::shared_ptr<Frame>;
    using FrameUnique   = std::unique_ptr<Frame>;
    using SeriesUnique  = std::unique_ptr<Series>;
    using MatrixXs      = Eigen::Matrix<std::string,Eigen::Dynamic,Eigen::Dynamic>;
    using MatrixXd      = Eigen::MatrixXd;
    using MatrixXsize   = Eigen::Matrix<size_t,Eigen::Dynamic,Eigen::Dynamic>;
    using MatrixXint    = Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic>;
    using MatrixXint_64 = Eigen::Matrix<int64_t,Eigen::Dynamic,Eigen::Dynamic>;
    //DataFrame similar to pandas dataframe 
    struct Frame{
        //stores the series
        std::vector<SeriesUnique> _data;
        std::vector<std::string> _headers;

        size_t _rows{0};
        size_t _cols{0};

        Frame(){}
        Frame(size_t row, size_t col):_rows(row),_cols(col){}
        Frame(std::vector<std::string>& h): _headers(h){}
        Frame(std::vector<SeriesUnique> data):_cols(data.size()),_rows(data.at(0)->size()),_data(std::move(data)){}
        Frame(SeriesUnique data, std::string header = ""){
            this->_data.push_back(std::move(data));
            this->_headers.push_back(header);
            this->_cols = this->_data.size();
            this->_rows = this->_data.at(0)->size();
        }
        Frame(std::vector<SeriesUnique> data,
        std::vector<std::string> _header):_cols(data.size()),_rows(data.at(0)->size()),_data(std::move(data)),_headers(_header){}

        template<typename T>
        Frame(std::vector<std::vector<T>>& data);
        template<typename T>
        Frame(std::vector<T>& data, std::string header = "");
        template<typename T>
        Frame(Eigen::MatrixBase<T>& data, std::vector<std::string> headers);
        template<typename T>
        Frame(Eigen::MatrixBase<T>& data,std::unordered_map<std::string,int>& label, std::vector<std::string> headers):Frame(data,headers){
            this->setLabel(label,0);
            this->_headers = headers;
            this->updateSize();
        }
        template<typename T>
        Frame(std::initializer_list<std::initializer_list<T>> data);
        //prints the whole frame
        void print(int size = -1, int ind = 5) const noexcept;
        //prints the headers
        void printHeader(int ind = 5) const noexcept;
        //prints the heading and data type
        void info() const noexcept;
        //use to add series to frame of type unique_ptr
        bool addSeries(SeriesUnique series);
        //use to add series to frame of type series
        bool addSeries(Series* series);
        //converts label to number
        void labelToNumber(int idx);
        //converts number to label
        void numberToLabel(int idx);
        //apply a function to frame
        void apply(std::function<double(double)> func);
        void apply(std::function<double(double,size_t i)> func);
        void apply(std::function<double(double,size_t i, size_t j)> func);
        //get series at a given postion
        Series* at(int i);
        //add Vec of given type
        template<typename T> bool addVec(Vec<T>* vec);
        template<typename U = double> U at(int i, int j);

        Series* operator[](int i){return this->_data.at(i % this->_cols).get();}
        FrameShared operator[](std::initializer_list<std::string> l){
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
    
        size_t rowSize() const noexcept{return _rows;}
        size_t colSize() const noexcept{return _cols;}
        //removes the column
        SeriesUnique dropCol(int idx); 
        //returns columns in a given range
        FrameShared colSlice(size_t start, size_t end = std::numeric_limits<size_t>::max()); 
        //returns columns in a given range using vector of indices
        FrameShared colSlice(std::vector<int> l); 
        //returns columns in a given range using column name
        FrameShared colSlice(std::initializer_list<std::string> l); 
        //TODO: implementing drop column
        std::vector<SeriesUnique> dropRow(int idx); 
        //splits the data set in a given percentage and using random seed
        //returns train, test
        Pair split(float trainPercentage = 30, int randomSeed = 42); 
        //splits the X and y in a given percentage and using random seed
        //returns X_train, X_test, y_train, y_test
        Tuple split(FrameShared& X, FrameShared& y, float trainPercentage = 30, int randomSeed = 42); 
        //randomizing rows
        static void randomize(Frame* frame, int seed = 42, int iteration = 50); 

        double mean(int col) noexcept {return this->at(col)->mean();}
        double sum(int col) noexcept {return this->at(col)->sum();}
        double std(int col) noexcept {return this->at(col)->std();}
        double variance(int col) noexcept {return this->at(col)->variance();}
        double median(int col) noexcept{return this->at(col)->median();}
        Eigen::VectorXd mean() noexcept;
        Eigen::VectorXd std() noexcept;
        Eigen::VectorXd variance() noexcept;
        Eigen::VectorXd median() noexcept;
        //normalize data between -1 and 1
        void normalize(double val = 0);
        void Zscore();
        //returns correlation coefficient
        double corrcoef(int i, int j) noexcept;
        //returns correlation coefficient matrix of every possible pair
        Eigen::MatrixXd corrcoefMatrix() noexcept;
        Eigen::MatrixXd cov() noexcept;
        //return the unique elements in a given column
        template<typename T = double>
        std::unique_ptr<std::unordered_map<T, int>> unique(int col);
        //use to cast series into Vec struct
        template<typename U = double>
        static Vec<U>* cast(Series* data){
            return static_cast<Vec<U>*>(data);
        }
        //get the label
        std::unordered_map<std::string,int> getLabel(int idx) noexcept{
            return this->_data[idx % this->_cols]->_labelMap;
        }
        //set the label
        void setLabel(std::unordered_map<std::string,int> label, int idx) noexcept{
            this->_data[idx % this->_cols]->_labelMap = label;
        }
    protected:

        void indentUtil(int indent = 0) const noexcept{
            for(int i = 0; i< indent; i++){
                std::cout<<' ';
            }
        }
        //updates the rows and cols
        void updateSize() noexcept{
            this->_cols = this->_data.size();
            if(this->_cols != 0)
                this->_rows = this->_data.at(0)->size();
            else
                this->_rows = 0;
        }
        // return true if frame is empty
        bool isEmpty() const noexcept{
            if(this->_cols == 0 || this->_rows == 0) {
                std::string str = "Out of Bound!\nRow Size: " + std::to_string(this->_rows) + "\nCol Size: " + std::to_string(this->_cols);
                std::cerr<<str<<'\n';
                return true;
            }
            return false;
        }
        FrameUnique _train;
        FrameUnique _test;
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

    Eigen::VectorXd Frame::mean() noexcept{
        Eigen::VectorXd m(colSize());
        for(int i = 0; i < colSize();i++){
            m(i) = mean(i);
        }
        return m;
    }
    Eigen::VectorXd Frame::std() noexcept{
        Eigen::VectorXd m(colSize());
        for(int i = 0; i < colSize();i++){
            m(i) = std(i);
        }
        return m;
    }
    Eigen::VectorXd Frame::median() noexcept{
        Eigen::VectorXd m(colSize());
        for(int i = 0; i < colSize();i++){
            m(i) = median(i);
        }
        return m;
    }
    Eigen::VectorXd Frame::variance() noexcept{
        Eigen::VectorXd m(colSize());
        for(int i = 0; i < colSize();i++){
            m(i) = variance(i);
        }
        return m;
    }

    template<>
    std::string Frame::at<std::string>(int i,int j){
        if(isEmpty() || (i >= this->_rows || i < 0) || (j >= this->_cols || j < 0)) {
            std::cerr<< "Line: " + std::to_string(__LINE__)<<'\n';
            exit(1);
        }else return this->at(j)->atS(i);
    }

    void Frame::printHeader(int ind) const noexcept {
        std::vector<int> lenghtOfHeaderString(this->_headers.size());
        int i = 0;
        for(auto h : this->_headers){
            printf("%s",h.c_str());
            this->indentUtil(ind);
            lenghtOfHeaderString[i++] = h.size();
        }
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
                    std::cout<<d->atS(i);
                    std::string temp = d->atS(i);
                    indent(temp,lenghtOfHeaderString,j,ind);
                }else{
                    std::cout<<d->at(i);
                    double temp = d->at(i);
                    indent(temp,lenghtOfHeaderString,j,ind);
                }
            }
            std::cout<<'\n';
        }
    } 

    SeriesUnique Frame::dropCol(int idx){
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
    std::vector<SeriesUnique> Frame::dropRow(int idx){
        if(isEmpty()) {
            std::cerr<< "Line: " + std::to_string(__LINE__)<<'\n';
            return {};
        }
        // std::vector<SeriesUnique> temp;
        // for(int i = 0; i < this->_data.at(0)->size(); i++){
        //     SeriesUnique s;
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
            SeriesUnique s(new Vec<std::string>(series->_header,"string"));
            for(int i = 0; i < series->size(); i++){
                s->push_s(series->atS(i));
            }
            s->_labelMap = series->_labelMap;
            this->_headers.push_back((s)->_header);
            this->_data.push_back(std::move(s));
        }else{
            SeriesUnique s(new Vec<double>(series->_header,"double"));
            for(int i = 0; i < series->size(); i++){
                s->push_d(series->at(i));
            }
            s->_labelMap = series->_labelMap;
            this->_headers.push_back((s)->_header);
            this->_data.push_back(std::move(s));
        }
        this->updateSize();
        return true;
    }
    bool Frame::addSeries(SeriesUnique series){
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
    Frame::Frame(Eigen::MatrixBase<T>& data, std::vector<std::string> headers){
        for(size_t i = 0; i < data.cols(); i++){
            if constexpr(std::is_same_v<T,MatrixXs>){
                std::vector<std::string> temp;
                for(size_t j = 0; j < data.rows(); j++){
                    temp.push_back(data(j,i));
                }
                SeriesUnique s(new Vec<std::string>(headers[i],"string",temp));
                this->_data.push_back(std::move(s));
            }else{
                std::vector<double> temp;
                for(size_t j = 0; j < data.rows(); j++){
                    temp.push_back(data(j,i));
                }
                SeriesUnique s(new Vec<double>(headers[i],"double",temp));
                this->_data.push_back(std::move(s));
            }
        }
        this->_headers = headers;
        this->updateSize();
    }

    template<typename T>
    Frame::Frame(std::vector<T>& data, std::string header){
        if constexpr(std::is_same_v<T,std::string>){
            SeriesUnique s(new Vec<T>(header,"string",data));
            this->_data.push_back(std::move(s));
            this->_headers.push_back((header));
        }else{
            SeriesUnique s(new Vec<T>(header,"double",data));
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
                SeriesUnique s(new Vec<T>(std::to_string(i),"string",v));
                this->_data.push_back(std::move(s));
                this->_headers.push_back(std::to_string(i++));
            }else{
                SeriesUnique s(new Vec<T>(std::to_string(i),"double",v));
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
        SeriesUnique newCol(new Vec<double>(this->_headers[idx],"double",this->_data[idx]->size()));
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
        
        SeriesUnique newCol(new Vec<std::string>(this->_headers[idx],"string",this->_data[idx]->size()));
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

    void Frame::Zscore(){
        if(isEmpty()) {
            std::cerr<< "Line: " + std::to_string(__LINE__)<<'\n';
            return;
        }
        for(auto const& v: this->_data){
            double mean = v->mean();
            double std = v->stdS();
            v->apply([&](double el){
                return (el - mean) / (std);
            });
        }
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

    Pair Frame::split(float trainPercentage, int randomSeed){
        if(isEmpty()) {
            std::cerr<< "Line: " + std::to_string(__LINE__)<<'\n';
            exit(1);
        }

        Frame::randomize(this,randomSeed);

        size_t idx = std::ceil((float)((trainPercentage * this->_rows)/100.0));
        _train = std::make_unique<Frame>(this->_headers);
        _test = std::make_unique<Frame>(this->_headers);

        for(size_t i = 0; i < this->_cols; i++){
            if(this->_data.at(i)->_type == "string"){
                SeriesUnique temp(new Vec<std::string>(this->_headers.at(i),"string"));
                for(size_t j = 0; j < idx; j++){
                    temp->push_s(this->at<std::string>(j,i));
                }
                temp->_labelMap = this->getLabel(i);
                _train->_data.push_back(std::move(temp));
            }else{
                SeriesUnique temp(new Vec<double>(this->_headers.at(i),"double"));
                for(size_t j = 0; j < idx; j++){
                    temp->push_d(this->at(j,i));
                }
                temp->_labelMap = this->getLabel(i);
                _train->_data.push_back(std::move(temp));
            }
            _train->_cols = _train->_data.size();
            _train->_rows = _train->_data.at(0)->size();
        }
        for(size_t i = 0; i < this->_cols; i++){
            if(this->_data.at(i)->_type == "string"){
                SeriesUnique temp(new Vec<std::string>(this->_headers.at(i),"string"));
                for(size_t j = idx; j < this->_rows; j++){
                    temp->push_s(this->at<std::string>(j,i));
                }
                temp->_labelMap = this->getLabel(i);
                _test->_data.push_back(std::move(temp));
            }else{
                SeriesUnique temp(new Vec<double>(this->_headers.at(i),"double"));
                for(size_t j = idx; j < this->_rows; j++){
                    temp->push_d(this->at(j,i));
                }
                temp->_labelMap = this->getLabel(i);
                _test->_data.push_back(std::move(temp));
            }
            _test->updateSize();
            _train->updateSize();
        }
        
        return {this->_train.get(),this->_test.get()};
    }

    Tuple Frame::split(FrameShared& X, FrameShared& y, float trainPercentage, int randomSeed){
        auto [X_train, X_test] = X->split(trainPercentage,randomSeed);
        auto [y_train, y_test] = y->split(trainPercentage,randomSeed);
        return {X_train, X_test, y_train, y_test};
    }

    void Frame::randomize(Frame* frame, int seed,int iteration){
        if(frame->isEmpty()) {
            std::cerr<< "Line: " + std::to_string(__LINE__)<<'\n';
            return;
        }

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

    FrameShared Frame::colSlice(size_t start, size_t end){
        if(isEmpty() || this->_cols <= start || start < 0) {
            std::cerr<< "Line: " + std::to_string(__LINE__)<<'\n';
            exit(1);
        }
        FrameUnique f(new Frame(this->_rows,std::min(end,this->_cols) - start));
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
    FrameShared Frame::colSlice(std::vector<int> l){
        
        for(auto num: l){
            if(num >= this->_cols || num <0){
                std::cerr<< "Line: " + std::to_string(__LINE__)<<'\n';
            }
        }
        if(isEmpty()) {
            std::cerr<< "Line: " + std::to_string(__LINE__)<<'\n';
            exit(1);
        }
        FrameUnique f(new Frame(this->_rows, l.size()));
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
    FrameShared Frame::colSlice(std::initializer_list<std::string> l){
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
    void Frame::apply(std::function<double(double,size_t)> func){
        if(isEmpty()) {
            std::cerr<< "Line: " + std::to_string(__LINE__)<<'\n';
            return;
        }
        for(int i = 0; i < this->_cols; i++){
            if(this->_data.at(i)->_type != "string")
                this->_data.at(i)->apply(func,i);
        }
    }
    void Frame::apply(std::function<double(double,size_t,size_t)> func){
        if(isEmpty()) {
            std::cerr<< "Line: " + std::to_string(__LINE__)<<'\n';
            return;
        }
        for(int i = 0; i < this->_cols; i++){
            if(this->_data.at(i)->_type != "string")
                this->_data.at(i)->apply(func,i);
        }
    }

    Eigen::MatrixXd Frame::corrcoefMatrix() noexcept{
        Eigen::MatrixXd m(colSize(),colSize());
        for(auto i = 0; i < colSize(); i++){
            for(auto j = 0; j < colSize(); j++){
                m(i,j)= this->corrcoef(i,j);
            }
        }
        return m;
    }
    Eigen::MatrixXd Frame::cov() noexcept{
        Eigen::MatrixXd m(colSize(),colSize());
        m.setZero();
        Eigen::VectorXd Mean = mean();
        for(auto i = 0; i < colSize(); i++){
            for(auto j = 0; j < colSize(); j++){
                for(auto k = 0; k < rowSize();k++){
                    m(i,j) += ((at(k,i) - Mean(i)) * (at(k,j) - Mean(j)));   
                }
                m(i,j) /= (rowSize() - 1);
            }
        }
        return m;
    }
    double Frame::corrcoef(int i, int j) noexcept{
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
    void Frame::info() const noexcept{
        for(int i = 0; i < 20; i++) std::cout<<'-';
        std::cout<<"INFO";
        for(int i = 0; i < 20; i++) std::cout<<'-';
        puts("");
        std::cout<<'('<<_rows<<','<<_cols<<')'<<'\n';
        size_t maxS = 0;
        for(auto i = 0; i < this->_headers.size(); i++) maxS = std::max(maxS,_headers[i].size());
        for(auto i = 0; i < this->_headers.size(); i++){
            std::cout   <<std::setw(std::to_string(_headers.size()).size())
                        <<i<<':'<<_headers[i]<<std::setw(maxS + 10 - _headers[i].size())
                        <<": "<<this->_data[i]->_type<<'\n';
        }
    }
}

#endif