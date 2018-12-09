#ifndef MEDIAN_H
#define MEDIAN_H

#include<algorithm>
#include<vector>

template<typename T>
int partition(std::vector<T>& vec,int low, int high){
    int p = low, r = high, x = vec.at(r), i = p - 1;
    for(int j = p; j <= r - 1; j++){
        if(vec.at(j) <= x){
            i++;
            std::swap(vec.at(i),vec.at(j));
        }
    }
    std::swap(vec[i + 1],vec.at(r));
    return i + 1;
}

template<typename T>
int medianAlgo(std::vector<T>& vec,int left, int right,int kth){
    while(true){
        int pivotIdx = partition<T>(vec,left,right);
        int len = pivotIdx - left + 1;

        if(kth == len) return vec[pivotIdx];
        else if (kth < len) right = pivotIdx - 1;
        else{
            kth -= len;
            left = pivotIdx + 1;
        }
    }
}

#endif