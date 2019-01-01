#ifndef VEC_H
#define VEC_H
#include "../algorithm/median.hpp"
#include "../gch/headers.hpp"

namespace ML{
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
        size_t _totalNan{0};
        //store label after converting string to number
        std::unordered_map<std::string,int> _labelMap;

        Series(){}
        Series(std::string tag,std::string header):_type(tag), _header(header){}
        size_t _size{0};
        size_t size() const noexcept{return _size;}
        virtual ~Series() = default;

        virtual double mean() const noexcept = 0;
        //returns the population variance
        virtual double variance() const noexcept = 0;
        //returns the sample variance 
        virtual double varianceS() const noexcept = 0;
        //return the population standard deviation
        virtual double std() const noexcept = 0;
        //return the sample standard deviation
        virtual double stdS() const noexcept = 0;
        virtual double median() noexcept = 0;
        virtual double max() const noexcept = 0;
        virtual double min() const noexcept = 0;
        virtual double sum() const noexcept = 0;
        //returns the number of unique elements in a vector
        virtual std::unordered_map<std::string, int> unique() = 0;
        //use to apply a function to whole vector
        virtual void apply(std::function<double(double)> func) = 0;
        virtual void apply(std::function<double(double,size_t)> func,size_t i) = 0;
        virtual void apply(std::function<double(double,size_t,size_t)> func,size_t i) = 0;
        //prints the vector
        virtual void print(int numberOfData = -1) const = 0;
        virtual bool isnan() const noexcept = 0;
        virtual void fillnan(double val) noexcept = 0;
        virtual void swap(size_t i, size_t j) = 0;
        //use to push double in vector
        virtual void push_d(double val) = 0;
        //pushing the string
        virtual void push_s(std::string val) = 0;
        //return the elements at giving postion i of type double
        virtual double at(int i) const noexcept = 0;
        //return the elements at giving postion i of type string
        virtual std::string atS(int i) const noexcept = 0;

    };
    template <typename T>
    struct Vec : Series{
        std::vector<T> _data;
        Vec(){}
        Vec(std::string header, std::string type, std::vector<T>& vec):_data(std::move(vec)){
            this->_header = header;
            this->_type = type;
            this->_size = _data.size();
            if constexpr(std::is_same_v<T,double>){
                for(auto const& n : _data){
                    if(std::isnan(n)) _totalNan++;
                }
            }
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
            if constexpr(std::is_same_v<T,double>){
                for(auto const& n : _data){
                    if(std::isnan(n)) _totalNan++;
                }
            }
        }
        //Generic push_back api
        void push_back(T data){
           this-> _data.push_back(data);
           this->_size = this->_data.size();
        }

        void push_d(double val) override{
            if constexpr(std::is_same_v<double,T>) {
                if(std::isnan(val)) _totalNan++;
                this->push_back(val);
            }
        }
        void push_s(std::string val) override{
            if constexpr(std::is_same_v<std::string,T>) this->push_back(val);
        }

        double at(int i) const noexcept override{
            if constexpr(std::is_same_v<T,double>) return this->_data.at(i);
            else return 0;
        }
        std::string atS(int i)const noexcept override{
            if constexpr(std::is_same_v<T,std::string>) return this->_data.at(i);
            else return std::to_string(_data[i]);
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

        void ltrim(std::string& str) const noexcept{
            str.erase(str.begin(),std::find_if(str.begin(),str.end(),[](char c){
                return !std::isspace(c);
            }));
        }

        void rtrim(std::string &s) const noexcept{
            s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
                return !std::isspace(ch);
            }).base(), s.end());
        }

        void trim(std::string &s) const noexcept{
            ltrim(s);
            rtrim(s);
        }
        
        T& operator[](int i){
            return (this->_data.at(i));
        }

        double sum() const noexcept final override{
            double m = 0;
            if(this->_data.empty()) return 0;
            if constexpr(std::is_same_v<T, std::string>) return 0;
            else{
                for(auto num : this->_data){
                    if(!std::isnan(num)) m += num;
                }
            }
            return m;
        }
        double mean() const noexcept final override{
            double m = 0;
            if(this->_data.empty()) return 0;
            if constexpr(std::is_same_v<T, std::string>) return 0;
            else{
                int f = 0;
                for(auto num : this->_data){
                    if(std::isnan(num)) f++;
                    else m += num;
                }
                m /= (this->size() - f);
            }
            return m;
        }
        double max() const noexcept final override{
            double m;
            if constexpr(std::is_same_v<T, std::string>) return std::numeric_limits<double>::min();
            else {
                m = std::numeric_limits<double>::min();
                for(int i = 0; i < this->_data.size();i++){
                    if(m < this->_data.at(i) && !std::isnan(this->_data.at(i))) m = this->_data.at(i);
                }
            }
            return m;  
        }
        void fillnan(double val) noexcept final override{
            if constexpr(std::is_same_v<T, double>){
                for(int i = 0; i < this->_data.size();i++){
                    if(std::isnan(this->_data.at(i))) {
                        _data[i] = val; 
                        _totalNan--;
                    }
                }
            }
        }
        bool isnan() const noexcept final override{
            if constexpr(std::is_same_v<T, std::string>) return false;
            else {
                double m = std::numeric_limits<double>::max();
                for(int i = 0; i < this->_data.size();i++){
                    if(std::isnan(this->_data.at(i))) return true;
                }
                return false;
            }
        }
        double min() const noexcept final override{
            double m;
            if constexpr(std::is_same_v<T, std::string>) return std::numeric_limits<double>::max();
            else {
                m = std::numeric_limits<double>::max();
                for(int i = 0; i < this->_data.size();i++){
                    if(m > this->_data.at(i) && !std::isnan(this->_data.at(i))) m = this->_data.at(i);
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
        void apply(std::function<double(double,size_t)> func,size_t idx) final override{
            if(this->_data.empty()) return;
            if constexpr(std::is_same_v<T, std::string>) return; 
            else{
                for(int i = this->_data.size() - 1; i >= 0; i--) this->_data.at(i) = func(this->_data.at(i),idx);
            }
        }
        void apply(std::function<double(double,size_t i,size_t j)> func,size_t idx) final override{
            if(this->_data.empty()) return;
            if constexpr(std::is_same_v<T, std::string>) return; 
            else{
                for(int i = this->_data.size() - 1; i >= 0; i--) this->_data.at(i) = func(this->_data.at(i),idx,i);
            }
        }

        double variance() const noexcept final override{
            double s = 0;
            if(this->_data.empty()) return 0;
            if constexpr(std::is_same_v<T, std::string>) return 0; 
            else{
                double m = this->mean();
                int f = 0;
                for(int i = this->_data.size() - 1; i >= 0; i--){
                    if(!std::isnan(_data.at(i))) s += (this->_data.at(i) * this->_data.at(i));
                    else f++;
                }
                s /= this->size() - f;
                s -= m * m; 
            }
            return s;
        }
        double varianceS() const noexcept final override{
            double s = 0;
            if(this->_data.empty()) return 0;
            if constexpr(std::is_same_v<T, std::string>) return 0; 
            else{
                double m = this->mean();
                for(int i = this->_data.size() - 1; i >= 0; i--){
                    if(!std::isnan(_data.at(i))) s += (this->_data.at(i) - m) * (this->_data.at(i) - m);
                }
                s /= (this->size() - 1 - _totalNan);
            }
            return s;
        }

        double std() const noexcept final override{
            if(this->_data.empty()) return 0;
            if constexpr(std::is_same_v<T, std::string>) return 0; 
            else return std::sqrt(this->variance());
        }
        double stdS() const noexcept final override{
            if(this->_data.empty()) return 0;
            if constexpr(std::is_same_v<T, std::string>) return 0; 
            else return std::sqrt(this->varianceS());
        }
        
       std::unordered_map<std::string, int> unique() final override{
            std::unordered_map<std::string, int> m;
            std::string temp;
            for(auto el : this->_data){
                temp = to_string(el);
                trim(temp);
                if(auto it = m.find(temp); it != m.end()) it->second++;
                else m[to_string(temp)] = 1;
            }
            return m;
        }

        double median() noexcept final override{
            if constexpr(std::is_same_v<T, std::string>) return 0; 
            else{
                double m = 0;
                std::vector<double> v;
                for(int i = 0; i < _data.size();i++){
                    if(!std::isnan(at(i))) v.push_back(at(i));
                }
                std::sort(v.begin(),v.end());
                m = v[(v.size()/2)];
                return m;
            }
        }

        void swap(size_t i, size_t j) final override{
            std::swap(this->_data.at(i), this->_data.at(j));
        }

    };

} // namespace ML

#endif