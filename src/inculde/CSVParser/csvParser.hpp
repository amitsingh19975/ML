#ifndef CSV_PARSER_HPP
#define CSV_PARSER_HPP

#include "../Series/vec.hpp"
#include "../Frame/frame.hpp"

namespace ML{
    struct CSV : public Frame{
        CSV(const std::string fileName,bool parse = true);

            void parseToType();
        protected:
            void getHeader(std::string& headers);
            std::vector<std::string> parseData(std::string& data);
            void parse(std::vector<std::string>& data);
            void normalizeFileName(std::string& fileName);
            std::ifstream _f;
            std::vector<bool> isDouble;
            void trim(std::string& str) const noexcept{
                str.erase(str.begin(),std::find_if(str.begin(),str.end(),[](char c){
                    return !std::isspace(c);
                }));
            }
    };

    CSV::CSV(std::string fileName, bool parse){
        this->normalizeFileName(fileName);
        _f.open(fileName);
        if(!_f){
            throw std::runtime_error("File Not Found!");
        }

        std::string headers, data;
        getline(_f,headers);
        this->getHeader(headers);
        
        this->isDouble.resize(this->_headers.size());

        for(size_t i = 0; i < this->isDouble.size(); i++) this->isDouble[i] = true;

        while(getline(_f,data)) {
            auto d = this->parseData(data);
            this->parse(d);
        }
        this->updateSize();
        if(parse) this->parseToType();
    }

    void CSV::getHeader(std::string& headers){
        std::stringstream os(headers);
        std::string header;
        char c;
        while(!os.eof()){
            std::string temp;
            os>>c;
            if(c == '\"'){
                os>>c;
                while(c != '\"' && !os.eof()){
                    temp +=c;
                    os>>c;
                }
                if(!os.eof()) this->_headers.push_back(temp);
            }else if (c != ','){
                while(c != ',' && !os.eof()){
                    temp +=c;
                    os>>c;
                }
                this->_headers.push_back(temp);
            }
        }
    }

    std::vector<std::string> CSV::parseData(std::string& data){
        std::stringstream os(data),osT;
        std::vector<std::string> d;
        char c;
        bool isReading = true;
        std::stack<char> s;
        os>>std::noskipws;
        while(true){
            os>>c;
            std::string temp;
            if((c == '\"' || !s.empty())){
                if(!s.empty()) osT<<'\n'<<c;
                else s.push(c);
                os>>c;
                while(!os.eof()){
                    if(c == '\"' && os.peek() == ',') {
                        if(!s.empty()) s.pop();
                        temp = osT.str();
                        d.push_back(temp);
                        break;
                    }
                    else if(c == '\"' && os.peek() == ' '){
                        osT<<c;
                        os>>c;
                        int count = 1;
                        while(c == ' ' && !os.eof() ) {
                            osT<<c;
                            count++;
                            os>>c;
                        }
                        if(c == ',') {
                            std::string t = osT.str();
                            t = t.substr(0, t.size() - count);
                            osT.str(t);
                            osT.clear();
                            if(!s.empty()) s.pop();
                            temp = osT.str();
                            d.push_back(temp);
                            break;
                        };
                    }else{
                        osT<<c;
                        os>>c;
                    }
                }
            }else{
                bool exe = false;
                while(c != ',' && !os.eof()){
                    temp +=c;
                    os>>c;
                    exe = true;
                }
                if(exe || (c == ',' && os.peek() == ',')) d.push_back(temp);
            }
            if(os.eof()){
                if(!s.empty()){
                    getline(_f,data);
                    os.str(data);
                    os.clear();
                }else{
                    break;
                }
            }else{
                osT.str("");
                osT.clear();
            }
        }
        for(auto &str : d){
            auto pos = str.find("\"\"");
            while(pos != std::string::npos){
                str.replace(pos,2,"\"");
                pos = str.find("\"\"");
            }
        }
        return d;
    }

    void CSV::parseToType(){
        auto size = this->_headers.size();
        for(auto i = 0; i < isDouble.size(); i++){
            if(isDouble[i]){
                double temp;

                if(this->at<std::string>(0,i) == "") temp = std::numeric_limits<double>::quiet_NaN();
                else temp = std::stod(this->at<std::string>(0,i));
                SeriesUnique series{new Vec<double>(this->_headers[i],"double", {temp})};
                for(auto j = 1; j <this->_rows; j++){
                    if(this->at<std::string>(j,i) == "") temp = std::numeric_limits<double>::quiet_NaN();
                    else temp = std::stod(this->at<std::string>(j,i));
                    series->push_d(temp);
                }
                this->_data[i] = std::move(series);
            }
        }

    }
    void CSV::parse(std::vector<std::string>& data){
        auto size = this->_headers.size();
        if(data.size() != size) return;
        if(this->_data.empty()){
            for(size_t i = 0; i < size; i++){
                try{
                    (void)stod(data[i]);
                }catch(...){
                    trim(data[i]);
                    if(data[i] != "")
                        isDouble[i] = false;
                }
                SeriesUnique series{new Vec<std::string>(this->_headers[i],"string", {data[i]})};
                this->_data.push_back(std::move(series));
            }
        }else{
            for(size_t i = 0; i < size; i++){
                try{
                    (void)stod(data[i]);
                }catch(...){
                    trim(data[i]);
                    if(data[i] != "")
                        isDouble[i] = false;
                }
                this->_data[i]->push_s(data[i]);
            }
        }

    }

    void CSV::normalizeFileName(std::string& fileName){
        if(fileName.substr(fileName.size() - 4) != ".csv"){
            fileName += ".csv";
        }
    }
}

#endif