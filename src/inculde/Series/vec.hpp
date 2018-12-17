#ifndef VEC_H
#define VEC_H
#include "../algorithm/median.hpp"
#include "../gch/headers.hpp"

namespace ML{
    using namespace boost::numeric::ublas;
    //Converting unknown type to string
    template<typename T = std::string>
    std::string to_string(std::string& num){
        return num;
    }

    template<typename T>
    std::string to_string(T& num){
        std::string temp = std::to_string(num);
        int i = temp.size() - 1;
        for(; i >= 0; i--) if(temp.at(i) != '0' || temp.at(i) == '.') break;
        return temp.substr(0,i);
    }
    //helper function for indent
    template<typename T>
    void indent(T& num,std::vector<int>& lenghtOfHeaderString, int j, int in = 5){
        in += ((int)lenghtOfHeaderString.at(j) - (int)to_string(num).size());
        for(int i = 0; i < in; i++) std::cout<<' ';
    }
    //base class for vector for dynamic casting
    struct Series{
        //store the type of vector
        std::string _type;
        std::string _header{""};
        //store label after converting string to number
        std::unordered_map<std::string,int> _labelMap;

        Series(){}
        Series(std::string tag,std::string header):_type(tag), _header(header){}
        size_t _size{0};
        size_t size() const noexcept{return _size;}
        virtual ~Series() = default;

        virtual double mean() = 0;
        //returns the population variance
        virtual double variance() = 0;
        //returns the sample variance 
        virtual double varianceS() = 0;
        //return the population standard deviation
        virtual double std() = 0;
        //return the sample standard deviation
        virtual double stdS() = 0;
        virtual double median() = 0;
        virtual double max() = 0;
        virtual double min() = 0;
        virtual double sum() = 0;
        //returns the number of unique elements in a vector
        virtual std::unique_ptr<std::unordered_map<std::string, int>> unique() = 0;
        //use to apply a function to whole vector
        virtual void apply(std::function<double(double)> func) = 0;
        //prints the vector
        virtual void print(int numberOfData = -1) const = 0;
        virtual void swap(size_t i, size_t j) = 0;
        //use to push double in vector
        virtual void push_d(double val) = 0;
        //pushing the string
        virtual void push_s(std::string val) = 0;
        //return the elements at giving postion i of type double
        virtual double at(int i) = 0;
        //return the elements at giving postion i of type string
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
        //Generic push_back api
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
            if constexpr(std::is_same_v<T,double>) return this->_data.at(i);
            else return 0;
        }
        std::string atS(int i)override{
            if constexpr(std::is_same_v<T,std::string>) return this->_data.at(i);
            else return "";
        }
        
        void print(int numberOfData = -1) const override{
            std::cout<<this->_header;
            std::cout<<'\n';
            std::cout<<std::string(this->_header.size() + 5,'-');
            std::cout<<'\n';
            int size = numberOfData < 0 || numberOfData >= this->size() - 1? this->size(): numberOfData;
            for(int i = 0; i < size; i++){
                std::cout<<this->_data.at(i)<<'\n';
            }
        }

        T& operator[](int i){
            return (this->_data.at(i));
        }

        double sum() final override{
            double m = 0;
            if(this->_data.empty()) return 0;
            if constexpr(std::is_same_v<T, std::string>) return 0;
            else{
                for(auto num : this->_data){
                    m += num;
                }
            }
            return m;
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
                    if(m < this->_data.at(i)) m = this->_data.at(i);
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
                    if(m > this->_data.at(i)) m = this->_data.at(i);
                }
            }
            return m;
        }

        void apply(std::function<double(double)> func) final override{
            if(this->_data.empty()) return;
            if constexpr(std::is_same_v<T, std::string>) return; 
            else{
                for(int i = this->_data.size() - 1; i >= 0; i--) this->_data.at(i) = func(this->_data.at(i));
            }
        }

        double variance() final override{
            double s = 0;
            if(this->_data.empty()) return 0;
            if constexpr(std::is_same_v<T, std::string>) return 0; 
            else{
                double m = this->mean();
                for(int i = this->_data.size() - 1; i >= 0; i--){
                    s += (this->_data.at(i) * this->_data.at(i));
                }
                s /= this->size();
                s -= m * m; 
            }
            return s;
        }
        double varianceS() final override{
            double s = 0;
            if(this->_data.empty()) return 0;
            if constexpr(std::is_same_v<T, std::string>) return 0; 
            else{
                double m = this->mean();
                for(int i = this->_data.size() - 1; i >= 0; i--){
                    s += (this->_data.at(i) - m) * (this->_data.at(i) - m);
                }
                s /= (this->size() - 1);
            }
            return s;
        }

        double std() final override{
            if(this->_data.empty()) return 0;
            if constexpr(std::is_same_v<T, std::string>) return 0; 
            else return std::sqrt(this->variance());
        }
        double stdS() final override{
            if(this->_data.empty()) return 0;
            if constexpr(std::is_same_v<T, std::string>) return 0; 
            else return std::sqrt(this->varianceS());
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
            std::swap(this->_data.at(i), this->_data.at(j));
        }

    };

} // namespace ML

#endif