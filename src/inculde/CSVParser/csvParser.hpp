#ifndef CSV_PARSER_HPP
#define CSV_PARSER_HPP

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include "../Series/vec.hpp"
#include "../Frame/frame.hpp"

namespace ML{
    struct CSV : public Frame{
        CSV(const std::string fileName);

        protected:
            void getHeader(std::string& headers);
            void getData(std::string& data);
            void normalizeFileName(std::string& fileName);
    };

    CSV::CSV(std::string fileName){
        this->normalizeFileName(fileName);
        std::ifstream f(fileName);
        if(!f){
            throw std::runtime_error("File Not Found!");
        }

        std::string headers, data;
        getline(f,headers);
        this->getHeader(headers);
        
        while(getline(f,data)) {
            this->getData(data);
        }
        this->_cols = this->_data.size();
        this->_rows = this->_data[0]->size();
    }

    void CSV::getHeader(std::string& headers){
        std::stringstream os(headers);
        std::string header;

        while(getline(os,header,',')){
            if(header == "\"")
                this->_headers.push_back(header.substr(1,header.size()-2));
            else
                this->_headers.push_back(header);
        }
    }

    void CSV::getData(std::string& data){
        std::stringstream os(data);
        std::string d;
        if(this->_data.size() == 0){
            int i = 0;
            while(getline(os,d,',')){
                if(d[0] == '\"'){
                    std::unique_ptr<Series> series{new Vec<std::string>(this->_headers[i++],"string", {d.substr(1,d.size()-2)})};
                    this->_data.push_back(std::move(series));
                }else{
                    std::unique_ptr<Series> series{new Vec<double>(this->_headers[i++],"double", {std::stod(d)})};
                    this->_data.push_back(std::move(series));
                }
            }
        }else{
            int i = 0;
            while(getline(os,d,',')){
                if(d[0] == '\"'){
                    static_cast<Vec<std::string>*>(this->_data[i++].get())->push_back(d.substr(1,d.size()-2));
                }else{
                    static_cast<Vec<double>*>(this->_data[i++].get())->push_back(std::stod(d));
                }
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