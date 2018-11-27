#ifndef FRAME_HPP
#define FRAME_HPP

#include <iostream>
#include <string>
#include <vector>
#include "../Series/vec.hpp"

namespace ML{
    struct Frame{
        std::vector<std::unique_ptr<Series>> _data;
        std::vector<std::string> _headers;
        Frame(){}
        Frame(std::vector<std::unique_ptr<Series>> data):_data(std::move(data)){}
        Frame(std::unique_ptr<Series> data, std::string header = ""){
            this->_data.push_back(std::move(data));
            this->_headers.push_back(header);
        }
        Frame(std::vector<std::unique_ptr<Series>> data,
        std::vector<std::string> _header):_data(std::move(data)),_headers(_header){}
        
        template<typename T>
        Frame(std::vector<std::vector<T>>& data);
        template<typename T>
        Frame(std::vector<T>& data, std::string header = "");
        template<typename T>
        Frame(std::initializer_list<std::initializer_list<T>> data);

        void print(int size = -1, int ind = 5) const noexcept;
        bool addSeries(std::unique_ptr<Series> series);
        template<typename T> bool addVec(Vec<T>* vec);
        template<typename U> U& at(int i, int j);
        void convertToNumber(int idx);

        Series* operator[](int i){return this->_data[i].get();}

        std::unique_ptr<Series> dropCol(int idx); 
        std::vector<std::unique_ptr<Series>> dropRow(int idx); 

        double mean(int col);
        double std(int col);
        double variance(int col);
        double median(int col);
        void normalize(double val = 0);
        
        template<typename T = double>
        std::unique_ptr<std::unordered_map<T, int>> unique(int col);

        template<typename U = double>
        static Vec<U>* cast(Series* data){
            return dynamic_cast<Vec<U>*>(data);
        }
        protected:
            void indentUtil(int indent = 0) const noexcept{
                for(int i = 0; i< indent; i++){
                    std::cout<<' ';
                }
            }
    };

    template<typename U>
    U& Frame::at(int i,int j){
        return (*cast<double>(this->_data[j].get()))[i];
    }

    template<>
    std::string& Frame::at<std::string>(int i,int j){
        return (*cast<std::string>(this->_data[j].get()))[i];
    }

    void Frame::print(int size, int ind) const noexcept {

        size = size < - 1 || size > this->_data[0]->size() ? this->_data[0]->size() : size;

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
                auto d = this->_data[j].get();
                if(d->_type == "string"){
                    Vec<std::string> t = *cast<std::string>(d);
                    std::cout<<t[i];
                    indent(t[i],lenghtOfHeaderString,j,ind);
                }else{
                    Vec<double> t = *cast<double>(d);
                    std::cout<<t[i];
                    indent(t[i],lenghtOfHeaderString,j,ind);
                }
            }
            std::cout<<'\n';
        }
    } 

    std::unique_ptr<Series> Frame::dropCol(int idx){
        this->_headers.erase(this->_headers.begin() + idx,this->_headers.begin() + idx + 1);
        auto t = std::move(this->_data[idx]);
        (this->_data.erase(this->_data.begin() + idx,this->_data.begin() + idx + 1));
        return std::move(t);
    } 
    std::vector<std::unique_ptr<Series>> Frame::dropRow(int idx){
        // std::vector<std::unique_ptr<Series>> temp;
        // for(int i = 0; i < this->_data[0]->size(); i++){
        //     std::unique_ptr<Series> s;
        //     if(this->_data[i]->_type == "string") {
        //         auto t = this->cast<std::string>(this->_data[i].get());
        //         temp.emplace_back(std::make_unique<Vec<std::string>>(t->_header,"string",t[i]));
        //     }else{
        //         auto t = this->cast(this->_data[i].get());
        //         temp.emplace_back(std::make_unique<Vec<double>>(t->_header,"double",t[i]));
        //     }  
        // }
        return {};
    }

    bool Frame::addSeries(std::unique_ptr<Series> series){
        if( series == nullptr
            ||((series)->_type != "string" && (series)->_type != "double")
            || (series)->size() == 0
            || (series)->_header == "") return false;
        if((series)->_type == "string" && Frame::cast<std::string>(series.get()) == nullptr) return false;
        if((series)->_type == "double" && Frame::cast(series.get()) == nullptr) return false;

        this->_headers.push_back((series)->_header);
        this->_data.push_back(std::move(series));
        
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
    }
    template<typename T>
    Frame::Frame(std::initializer_list<std::initializer_list<T>> data){

    }

    void Frame::convertToNumber(int idx){
        if(this->_data[idx]->_type != "string") return;
        int i = 0, j = 0;
        std::unique_ptr<Series> newCol(new Vec<double>(this->_headers[idx],"double",this->_data[idx]->size()));
        auto newColVec = Frame::cast(newCol.get());
        for(auto key : Frame::cast<std::string>(this->_data[idx].get())->_data){
            if(this->_data[idx]->_stringToNumberMap.find(key) == this->_data[idx]->_stringToNumberMap.end()){
                this->_data[idx]->_stringToNumberMap[key] = i++;
            }
            newColVec->_data[j++] = (this->_data[idx]->_stringToNumberMap[key]);
        }
        this->_data[idx] = std::move(newCol);
    }

    void Frame::normalize(double val){
        if(val == 0){
            for(auto const& v: this->_data){
                double max = v->max();
                double min = v->min();
                if(max != 0){
                    v->apply([&](double el){
                        return (el - max) / (max - min);
                    });
                }
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
}

#endif