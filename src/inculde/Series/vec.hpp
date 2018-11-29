#ifndef VEC_H
#define VEC_H

#include <iostream>
#include <vector>
#include <cmath>
#include <unordered_map>
#include <initializer_list>
#include "../algorithm/median.hpp"

namespace ML{

    template<typename T = std::string>
    std::string to_string(std::string& num){
        return num;
    }

    template<typename T>
    std::string to_string(T& num){
        std::string temp = std::to_string(num);
        int i = temp.size() - 1;
        for(; i >= 0; i--) if(temp[i] != '0' || temp[i] == '.') break;
        return temp.substr(0,i);
    }

    template<typename T>
    void indent(T& num,std::vector<int>& lenghtOfHeaderString, int j, int in = 5){
        in += ((int)lenghtOfHeaderString[j] - (int)to_string(num).size());
        for(int i = 0; i < in; i++) std::cout<<' ';
    }

    struct Series{
        std::string _type;
        std::string _header{""};
        std::unordered_map<std::string,double> _labelMap;

        Series(){}
        Series(std::string tag,std::string header):_type(tag), _header(header){}
        size_t _size{0};
        size_t size() const noexcept{return _size;}
        virtual ~Series() = default;
        
        virtual double mean() = 0;
        virtual double variance() = 0;
        virtual double std() = 0;
        virtual double median() = 0;
        virtual double max() = 0;
        virtual double min() = 0;
        virtual std::unique_ptr<std::unordered_map<std::string, int>> unique() = 0;
        virtual void apply(std::function<double(double)> func) = 0;
        virtual void print(int numberOfData = -1) const = 0;
        virtual void swap(size_t i, size_t j) = 0;
        virtual void push_d(double val) = 0;
        virtual void push_s(std::string val) = 0;
        virtual double at(int i) = 0;
        virtual std::string atS(int i) = 0;

    };
    template <typename T>
    struct Vec : Series{
        std::vector<T> _data;
        Vec(){}
        Vec(std::string header, std::string type, std::vector<T>& vec):_data(std::move(vec)){
            this->_header = header;
            this->_type = type;
            this->_size = _data.size();
        }
        Vec(std::string header, std::string type){
            this->_header = header;
            this->_type = type;
            this->_size = _data.size();
        }
        Vec(std::string header, std::string type, size_t size):_data(size){
            this->_header = header;
            this->_type = type;
            this->_size = size;
        }
        Vec(std::string header, std::string type, std::initializer_list<T> l):_data(l){
            this->_header = header;
            this->_type = type;
            this->_size = _data.size();
        }
        
        void push_back(T data){
           this-> _data.push_back(data);
           this->_size = this->_data.size();
        }

        void push_d(double val) override{
            if constexpr(std::is_same_v<T,double>) this->push_back(val);
        }
        void push_s(std::string val) override{
            if constexpr(std::is_same_v<T,std::string>) this->push_back(val);
        }

        double at(int i) override{
            if constexpr(std::is_same_v<T,double>) return this->_data[i];
            else return 0;
        }
        std::string atS(int i)override{
            if constexpr(std::is_same_v<T,std::string>) return this->_data[i];
            else return "";
        }
        
        void print(int numberOfData = -1) const override{
            std::cout<<this->_header;
            std::cout<<'\n';
            std::cout<<std::string(this->_header.size() + 5,'-');
            std::cout<<'\n';
            int size = numberOfData < 0 || numberOfData >= this->size() - 1? this->size(): numberOfData;
            for(int i = 0; i < size; i++){
                std::cout<<this->_data[i]<<'\n';
            }
        }

        T& operator[](int i){
            return (this->_data[i]);
        }

        double mean() final override{
            double m = 0;
            if(this->_data.empty()) return 0;
            if constexpr(std::is_same_v<T, std::string>) return 0;
            else{
                for(auto num : this->_data){
                    m += num;
                }
                m /= this->size();
            }
            return m;
        }
        double max() final override{
            double m;
            if constexpr(std::is_same_v<T, std::string>) return std::numeric_limits<double>::min();
            else {
                m = std::numeric_limits<double>::min();
                for(int i = 0; i < this->_data.size();i++){
                    if(m < this->_data[i]) m = this->_data[i];
                }
            }
            return m;  
        }
        double min() final override{
            double m;
            if constexpr(std::is_same_v<T, std::string>) std::numeric_limits<double>::max();
            else {
                m = std::numeric_limits<double>::max();
                for(int i = 0; i < this->_data.size();i++){
                    if(m > this->_data[i]) m = this->_data[i];
                }
            }
            return m;
        }

        void apply(std::function<double(double)> func) final override{
            if(this->_data.empty()) return;
            if constexpr(std::is_same_v<T, std::string>) return; 
            else{
                for(int i = this->_data.size() - 1; i >= 0; i--) this->_data[i] = func(this->_data[i]);
            }
        }

        double variance() final override{
            double s = 0;
            if(this->_data.empty()) return 0;
            if constexpr(std::is_same_v<T, std::string>) return 0; 
            else{
                double m = this->mean();
                for(int i = this->_data.size() - 1; i >= 0; i--){
                    s += (this->_data[i] * this->_data[i]);
                }
                s /= this->size();
                s -= m * m; 
            }
            return s;
        }

        double std() final override{
            if(this->_data.empty()) return 0;
            if constexpr(std::is_same_v<T, std::string>) return 0; 
            else return std::sqrt(this->variance());
        }
        
        std::unique_ptr<std::unordered_map<std::string, int>> unique() final override{
            std::unique_ptr<std::unordered_map<std::string, int>> m{new std::unordered_map<std::string, int>()};
            std::string temp;
            for(auto el : this->_data){
                temp = to_string(el);
                if(auto it = m->find(temp); it != m->end()) it->second++;
                else (*m)[to_string(temp)] = 1;
            }
            return std::move(m);
        }

        double median() final override{
            double m = 0;
            if(this->_data.empty()) return 0;
            if constexpr(std::is_same_v<T, std::string>) return 0; 
            else m = medianAlgo<double>(this->_data,1,this->_data.size() - 1,(this->_data.size())/2);
            return m;
        }

        void swap(size_t i, size_t j) final override{
            std::swap(this->_data[i], this->_data[j]);
        }

    };

} // namespace ML

#endif