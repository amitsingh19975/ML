#ifndef PRETTY_PRINT_H
#define PRETTY_PRINT_H

#include "../gch/headers.hpp"
#include "../Frame/frame.hpp"
#include "terminal_raw.hpp"

namespace ML{
    
    enum class FG : size_t{
        BLACK           = 30,
        RED,
        GREEN,
        YELLOW,
        BLUE,
        MAGENTA,
        CYAN,
        WHITE,
        BRIGHT_BLACK    = 90,
        BRIGHT_RED,
        BRIGHT_GREEN,
        BRIGHT_YELLOW,
        BRIGHT_BLUE,
        BRIGHT_MAGENTA,
        BRIGHT_CYAN,
        BRIGHT_WHITE,
        NONE = 39,
    };
    enum class BG : size_t{
        BLACK           = 40,
        RED,
        GREEN,
        YELLOW,
        BLUE,
        MAGENTA,
        CYAN,
        WHITE,
        BRIGHT_BLACK    = 100,
        BRIGHT_RED,
        BRIGHT_GREEN,
        BRIGHT_YELLOW,
        BRIGHT_BLUE,
        BRIGHT_MAGENTA,
        BRIGHT_CYAN,
        BRIGHT_WHITE,
        NONE = 49,
    };

    std::string getFGCode(FG f){
       return std::to_string((size_t)f);
    }
    std::string getBGCode(BG f){
       return std::to_string((size_t)f);
    }

    struct PrettyPrintData{
        template <typename T>
        using Vmatrix               = std::vector<std::vector<T>>;
        template <typename T>
        using VecArray              = std::vector<T>;
        Vmatrix<std::string>        _body;
        VecArray<std::string>       _headers;
        VecArray<size_t>            _width;
        PrettyPrintData(){}
        PrettyPrintData(Frame* frame);
        PrettyPrintData(Series* series);
        template<typename T>
        PrettyPrintData(Vmatrix<T>& m);
        template<typename T>
        PrettyPrintData(VecArray<T>& m);
        template<typename T>
        PrettyPrintData(Eigen::MatrixBase<T>& m);
    };

    struct Color{
        u_char  _r{0};
        u_char  _g{0};
        u_char  _b{0};
        FG      _fg{FG::NONE};
        BG      _bg{BG::NONE};
        bool    _isRGB{false};
        bool    _isEmpty{false};
        Color(u_char r, u_char g, u_char b):_r(r),_g(g),_b(b),_isRGB(true){}
        Color(FG fg):_fg(fg){}
        Color(BG bg):_bg(bg){}
        Color():_isEmpty(true){}
        std::string getFGRGBColor() const noexcept{
            std::string const rgb = "38;2;" + std::to_string(_r) 
                        + std::to_string(_g) + std::to_string(_b);
            return rgb;
        }
        std::string getBGRGBColor() const noexcept{
            std::string const rgb = "48;2;" + std::to_string(_r) 
                        + std::to_string(_g) + std::to_string(_b);
            return rgb;
        }

        std::string getFGColor() const noexcept{
            return getFGCode(_fg);
        }

        std::string getBGColor() const noexcept{
            return getBGCode(_bg);
        }
    };

    std::string getColor(Color f, Color b){
        std::string c = "\x1b[";
        if(f._isRGB && b._isRGB){
            if(!b._isEmpty && !f._isEmpty){
                c += f.getFGRGBColor() + ";" + b.getBGRGBColor();
            }else if(f._isEmpty){
                c += b.getBGRGBColor();
            }else {
                c += f.getFGRGBColor();
            }
        }else if(!f._isRGB && !b._isRGB){
            c += f.getFGColor() + ";" +b.getBGColor();
        }else if (!f._isRGB){
            if(b._isEmpty){
                c += f.getFGColor();
            }else {
                c +=f.getFGColor() + ";" + b.getBGRGBColor();
            }
            
        }else{
            if(f._isEmpty){
                c += b.getBGColor();
            }else {
                c +=f.getFGRGBColor() + ";" + b.getBGColor();
            }
        }
        return c;
    }

    struct PPrint{
        template <typename T>
        using Vmatrix      = std::vector<std::vector<T>>; 
        template <typename T>
        using VecArray     = std::vector<T>;
        static  Color   ColorHeaderFG;
        static  Color   ColorHeaderBG;
        static  Color   ColorBody;
        static  auto    print(Frame* frame, uint32_t indent = 5
                            , uint32_t const maxLine = 2, uint32_t maxWidth = 20) -> void;
        static  auto    print(Frame& frame, uint32_t indent = 5
                            , uint32_t const maxLine = 2, uint32_t maxWidth = 20) -> void;
        static  auto    print(FrameShared& frame, uint32_t indent = 5
                            , uint32_t const maxLine = 2, uint32_t maxWidth = 20) -> void;
        static  auto    print(Series* series, uint32_t indent = 5
                            , uint32_t const maxLine = 2, uint32_t maxWidth = 20) -> void;
        template<typename T>
        static  auto    print(Vmatrix<T>& , uint32_t indent = 5
                            , uint32_t const maxLine = 2, uint32_t maxWidth = 20) -> void;
        template<typename T>
        static  auto    print(VecArray<T>& , uint32_t indent = 5
                            , uint32_t const maxLine = 2, uint32_t maxWidth = 20) -> void;
        template<typename T>
        static  auto    print(Eigen::MatrixBase<T>& , uint32_t indent = 5
                            , uint32_t const maxLine = 2, uint32_t maxWidth = 20) -> void;
    protected:
        static  auto    printColumn(PrettyPrintData& p, size_t idx, size_t colPos, size_t rowPos,size_t min, size_t max,
                                 uint32_t indent, uint32_t const maxLine = 2, uint32_t maxWidth = 20) -> int;
        static  auto    wordWrap(std::vector<std::string>& w, size_t maxW) -> std::vector<std::pair<size_t,size_t>>;
        static  auto    split(std::string& line,std::vector<std::string>& w) -> void;
        static  auto    printHelper(PrettyPrintData& p, uint32_t indent = 5
                            , uint32_t const maxLine = 2, uint32_t maxWidth = 20) -> void;
    };

    Color PPrint::ColorHeaderBG = Color(BG::GREEN);
    Color PPrint::ColorHeaderFG = Color(FG::BLACK);
    Color PPrint::ColorBody = Color(FG::NONE);

    PrettyPrintData::PrettyPrintData(Series* series):_width(1){
        _headers.push_back(series->_header);
        _width[0] = (_headers.back().size());
        std::string str;
        VecArray<std::string> temp(series->size());
        for(auto j = 0; j < series->size(); j++){
            if(series->_type == "string"){
                _width[0] = std::max(_width[0],series->atS(j).size());
                temp[j] = series->atS(j);
            }else{
                double a = series->at(j);
                if(a == static_cast<int>(a) && !std::isnan(a)){
                    str = std::to_string(static_cast<int>(a));
                }else{
                    str = std::to_string(a);
                }
                _width[0] = std::max(_width[0],str.size());
                temp[j] = str;
            }
        }
        _body.push_back(temp);
    }
    PrettyPrintData::PrettyPrintData(Frame* frame):_width(frame->colSize()){
        _headers = frame->_headers;
        std::string str;
        for(auto i = 0; i < _headers.size(); i++) _width[i] = _headers[i].size();

        for(auto i = 0; i < frame->colSize(); i++){
            VecArray<std::string> temp(frame->rowSize());
            for(auto j = 0; j < frame->rowSize(); j++){
                if(frame->at(i)->_type == "string"){
                    _width[i] = std::max(_width[i],frame->at(i)->atS(j).size());
                    temp[j] = frame->at(i)->atS(j);
                }else{
                    double a = frame->at(i)->at(j);
                    if(a == static_cast<int>(a) && !std::isnan(a)){
                        str = std::to_string(static_cast<int>(a));
                    }else{
                        str = std::to_string(a);
                    }
                    _width[i] = std::max(_width[i],str.size());
                    temp[j] = str;
                }
            }
            _body.push_back(temp);
        }
    }
    template<typename T>
    PrettyPrintData::PrettyPrintData(Vmatrix<T>& m):_width(m.size()),_headers(m.size()){
        for(int i = 0; i < m.size(); i++) _headers[i] = "";
        _width.resize(_headers.size());
        for(int i = 0; i < m.size(); i++) _width[i] = _headers[i].size();

        for(int i = 0; i < m.size(); i++){
            VecArray<std::string> temp;
            for(int j = 0; j < m[0].size(); j++){
                if constexpr(std::is_same_v<T,std::string>){
                    temp.push_back(m[i][j]);
                }else{
                    temp.push_back(std::to_string(m[i][j]));
                }
                _width[i] = std::max(temp.back().size(),_width[i]);
            }
            _body.push_back(temp);
        }
    }
    template<typename T>
    PrettyPrintData::PrettyPrintData(VecArray<T>& m):_width(1),_headers(1){
        _headers[0] = "";
        _width[0] = _headers[0].size();

        VecArray<std::string> temp;
        for(int j = 0; j < m.size(); j++){
            if constexpr(std::is_same_v<T,std::string>){
                temp.push_back(m[j]);
            }else{
                temp.push_back(std::to_string(m[j]));
            }
            _width[0] = std::max(temp.back().size(),_width[0]);
        }
        _body.push_back(temp);
    }

    template<typename T>
    PrettyPrintData::PrettyPrintData(Eigen::MatrixBase<T>& m):_width(m.cols()),_headers(m.cols()){
       for(int i = 0; i < m.cols(); i++) _headers[i] = "";
        _width.resize(_headers.size());
        for(int i = 0; i < m.cols(); i++) _width[i] = _headers[i].size();

        for(int i = 0; i < m.cols(); i++){
            VecArray<std::string> temp;
            for(int j = 0; j < m.rows(); j++){
                if constexpr(std::is_same_v<T,std::string>){
                    temp.push_back(m(j,i));
                }else{
                    temp.push_back(std::to_string(m(j,i)));
                }
                _width[i] = std::max(temp.back().size(),_width[i]);
            }
            _body.push_back(temp);
        } 
    }

    auto PPrint::printHelper(PrettyPrintData& p, uint32_t indent, uint32_t const maxLine, uint32_t maxWidth) -> void{
        Terminal::init();
        auto [row, col] = Terminal::getWindowSize();
        int szOfCol = (maxWidth + 5);
        size_t max = (row - maxLine - 1)/(maxLine + 1);
        size_t totalHeight = max;
        int totalWidth = ((col - 10.0) / szOfCol) ;
        size_t j = 0,min = 0;
        Terminal::clearScreen();
        std::string mess = "Press q to quit!";
        while(1){
            int k = 0;
            int y = 0;
            for(int i = j; i < std::min(j + totalWidth,p._body.size()); i++){
                y = printColumn(p,i,k++,0,min,max,indent,maxLine,maxWidth);
            }
            Terminal::setCursor(0,y);
            write(STDOUT_FILENO, mess.c_str(),mess.size());
            Terminal::setCursor(0,y + 1);
            int c = Terminal::keyEvent();
            switch(c){
                case 'q':
                    write(STDOUT_FILENO, "\x1b[2J", 4);
                    write(STDOUT_FILENO, "\x1b[H", 3);
                    Terminal::disable();
                    return;
                case ARROW_LEFT:
                    if(j > 0){
                        j--;
                        Terminal::clearScreen();
                    }
                    break;
                case ARROW_RIGHT:
                    if(j + totalWidth < p._body.size()){
                        j++;
                        Terminal::clearScreen();
                    }
                    break;
                case ARROW_DOWN:
                    if(min < p._body[0].size() - totalHeight){
                        min++;
                        max++;
                        Terminal::clearScreen();
                    }
                    break;
                case ARROW_UP:
                    if(min >0){
                        min--;
                        max--;
                        Terminal::clearScreen();
                    }
                    break;
            }
        }
        Terminal::disable();
    }
    template<typename T>
    auto PPrint::print(Eigen::MatrixBase<T>& m, uint32_t indent, uint32_t const maxLine, uint32_t maxWidth) -> void{
        PrettyPrintData p(m);
        PPrint::printHelper(p, indent,maxLine,maxWidth);
    }
    
    auto PPrint::print(Frame* frame, uint32_t indent, uint32_t const maxLine, uint32_t maxWidth) -> void{
        PrettyPrintData p(frame);
        PPrint::printHelper(p, indent,maxLine,maxWidth);
    }
    
    auto PPrint::print(Frame& frame, uint32_t indent, uint32_t const maxLine, uint32_t maxWidth) -> void{
        PrettyPrintData p(&frame);
        PPrint::printHelper(p, indent,maxLine,maxWidth);
    }
    
    auto PPrint::print(FrameShared& frame, uint32_t indent, uint32_t const maxLine, uint32_t maxWidth) -> void{
        PrettyPrintData p(frame.get());
        PPrint::printHelper(p, indent,maxLine,maxWidth);
    }
    
    auto PPrint::print(Series* series, uint32_t indent, uint32_t const maxLine, uint32_t maxWidth) -> void{
        PrettyPrintData p(series);
        PPrint::printHelper(p, indent,maxLine,maxWidth);
    }
    template<typename T>
    auto PPrint::print(Vmatrix<T>& m, uint32_t indent, uint32_t const maxLine, uint32_t maxWidth) -> void{
        PrettyPrintData p(m);
        PPrint::printHelper(p, indent,maxLine,maxWidth);
    }
    template<typename T>
    auto PPrint::print(VecArray<T>& m, uint32_t indent, uint32_t const maxLine, uint32_t maxWidth) -> void{
        PrettyPrintData p(m);
        PPrint::printHelper(p, indent,maxLine,maxWidth);
    }

    auto PPrint::printColumn(PrettyPrintData& p, size_t idx, size_t colPos, size_t rowPos,
                size_t min, size_t max, uint32_t indent,uint32_t const maxLine, uint32_t maxWidth) -> int{
        int x = (maxWidth + indent) * colPos + 10 ,y{0};
        std::vector<std::string> lines;
        std::string line;
        
        if(p._headers[idx] == ""){
            Terminal::setCursor(x,y);
            std::string t = std::to_string(idx);
            auto ls = maxWidth > t.size() ? maxWidth - t.size() : 0;
            ls /= 2;
            std::string c = getColor(PPrint::ColorHeaderFG, PPrint::ColorHeaderBG) + ";1m";
            t = c + std::string(ls,' ') + t + std::string(ls,' ') + std::string("\x1b[0m");
            write(STDOUT_FILENO,t.c_str(),t.size());
            y = rowPos + maxLine + 1;
        }else{
            size_t diff = p._headers[idx].size() > p._width[idx] ? p._headers[idx].size() - p._width[idx] : p._width[idx] - p._headers[idx].size() ;
            std::string temp = std::to_string(idx) + " " + p._headers[idx];
            lines.clear();
            split(temp,lines);
            auto wordW = wordWrap(lines,maxWidth);
            int i = 0;
            for(auto [f,l] : wordW){
                temp = "";
                if( i == maxLine) break;
                for(; f <= l; f++){
                    if(f != l){
                        temp += lines[f - 1] +' ';
                    }else{
                        temp += lines[f - 1];
                    }
                }
                size_t ls = maxWidth > temp.size() ? maxWidth - temp.size() : 0;
                ls /=2;
                temp = std::string(ls,' ') + temp + std::string(ls,' ');
                if(temp.size() >= maxWidth ){
                    if(i == maxLine - 1){
                        temp[temp.size() - 3] = '.';
                        temp[temp.size() - 2] = '.';
                        temp[temp.size() - 1] = '.';
                    }else{
                        temp = temp.substr(0,maxWidth);
                    }
                }
                Terminal::setCursor(x,y);
                std::string c = getColor(PPrint::ColorHeaderFG, PPrint::ColorHeaderBG) + ";1m";
                temp = c + temp + "\x1b[0m"; 
                write(STDOUT_FILENO,temp.c_str(),temp.size());
                y++;
                i++;
            }
            y = rowPos + maxLine + 1;
            
        }
        for(auto i = min; i < max;i++){
            if(p._body[idx][i] == ""){
                Terminal::setCursor(x,y);
                // if(p._width[idx] > maxWidth) printf("%s",std::string(p._width[idx],' ').substr(0,maxWidth).c_str());
                // else printf("%s",std::string(p._width[idx],' ').c_str());
                write(STDOUT_FILENO," ",1);
                y += maxLine + 1;
            }else{
                std::string temp = p._body[idx][i];
                lines.clear();
                split(temp,lines);
                auto wordW = wordWrap(lines,maxWidth);
                int j = 0;
                int yTemp = y;
                for(auto [f,l] : wordW){
                    temp = "";
                    if( j == maxLine) break;
                    for(; f <= l; f++){
                        if(f != l){
                            temp += lines[f - 1] +' ';
                        }else{
                            temp += lines[f - 1];
                        }
                    }
                    size_t ls = maxWidth > temp.size() ? maxWidth - temp.size() : 0;
                    ls /=2;
                    if(temp.size() >= maxWidth ){
                        temp = temp.substr(0,maxWidth);
                        if(i == maxLine - 1){
                            temp[temp.size() - 3] = '.';
                            temp[temp.size() - 2] = '.';
                            temp[temp.size() - 1] = '.';
                        }
                    }
                    std::string c = getColor(PPrint::ColorBody, Color()) + "m";
                    temp = std::string(ls,' ') + temp + std::string(ls,' ');
                    Terminal::setCursor(x,y);
                    write(STDOUT_FILENO,temp.c_str(),temp.size());
                    y++;
                    j++;
                }
                y = yTemp + maxLine + 1;
            }
        }
        y = 3;
        for(int i = min; i < max; i++){
            Terminal::setCursor(0,y);
            std::string t = std::to_string(i);
            auto ls = 5 > t.size() ? 5 - t.size() : 0;
            ls /= 2;
            std::string c = getColor(PPrint::ColorHeaderFG, PPrint::ColorHeaderBG) + ";1m";
            t = c + std::string(ls,' ') + t + std::string(ls,' ') + std::string("\x1b[0m");
            write(STDOUT_FILENO,t.c_str(),t.size());
            y += rowPos + maxLine + 1;
        }
        return y;
    }

    auto PPrint::wordWrap(std::vector<std::string>& line, size_t maxW) -> std::vector<std::pair<size_t,size_t>>{
        std::vector<size_t> w;
        for(auto const& word:line) w.push_back(word.size());
        int i, j; 
        int currlen; 
        int cost; 
        int dp[w.size()]; 
        int ans[w.size()]; 
        dp[w.size() - 1] = 0; 
        ans[w.size() - 1] = w.size() - 1; 
    
        for (i = w.size() - 2; i >= 0; i--) { 
            currlen = -1; 
            dp[i] = std::numeric_limits<int>::max(); 
            for (j = i; j < w.size(); j++) { 
                currlen += (w[j] + 1); 
                if (currlen > maxW) 
                    break; 
                if (j == w.size() - 1) 
                    cost = 0; 
                else
                    cost = (maxW - currlen) * (maxW - currlen) + dp[j + 1]; 
                if (cost < dp[i]) { 
                    dp[i] = cost; 
                    ans[i] = j; 
                } 
            } 
        } 
        std::vector<std::pair<size_t,size_t>> temp;
        i = 0; 
        while (i < w.size()) { 
            temp.push_back({i+1,ans[i] + 1});
            i = ans[i] + 1; 
        } 
        return temp;
    }
    auto PPrint::split(std::string& line,std::vector<std::string>& w) -> void{
        std::stringstream ss(line);
        std::string str;
        while(std::getline(ss,str,' ')) w.push_back(str);
    }
}


#endif // PRETTY_PRINT_H
