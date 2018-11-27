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
        std::unordered_map<std::string,double> _stringToNumberMap;

        Series(){}
        Series(std::string tag,std::string header):_type(tag), _header(header){}
        size_t rows{0};
        size_t size() const noexcept{return rows;}
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

    };
    template <typename T>
    struct Vec : Series{
        std::vector<T> _data{};
        Vec(){}
        Vec(std::string header, std::string type, std::vector<T>& vec):_data(std::move(vec)){
            this->_header = header;
            this->_type = type;
            this->rows = _data.size();
        }
        Vec(std::string header, std::string type){
            this->_header = header;
            this->_type = type;
            this->rows = _data.size();
        }
        Vec(std::string header, std::string type, size_t size):_data(size){
            this->_header = header;
            this->_type = type;
            this->rows = size;
        }
        Vec(std::string header, std::string type, std::initializer_list<T> l):_data(l){
            this->_header = header;
            this->_type = type;
            this->rows = _data.size();
        }
        
        void push_back(T data){
           this-> _data.push_back(data);
           this->rows = this->_data.size();
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

    };

} // namespace ML

#endif