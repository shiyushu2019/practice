#include<vector>
#include<iostream>
#include <stdexcept>

class Tensor {
    private:
        std::vector<int> data_;
        std::vector<size_t> shape_;
    public:

        Tensor(std::vector<int> data,std::vector<size_t> shape)
            : data_(std::move(data)), shape_(std::move(shape)) {}
        Tensor(std::vector<int> data)
            : data_(std::move(data)), shape_(std::vector<size_t>{data_.size()}) {}

        std::vector<size_t>& shape(){
            return shape_;
        }
        std::vector<int>& data(){
            return data_;
        }

        void reshape(std::vector<size_t>& new_shape){
            size_t new_size = 1;
            for(auto dim : new_shape){
                new_size *= dim;
            }
            if(new_size != data_.size()){
                throw std::invalid_argument("New shape size does not match data size.");
            }
            shape_ = std::move(new_shape);
        }
};

Tensor matmul(Tensor& A, Tensor& B) {
    if (A.shape().size()==2 && B.shape().size()==2){

        if (A.shape()[1] != B.shape()[0]) {
            throw std::invalid_argument("Inner dimensions must match for matrix multiplication.");
        }
        /* 
        A: (3,6)
        00 01 02 03 04 05
        10 11 12 13 14 15
        20 21 22 23 24 25
        B: (6,3)
        00 01 02
        10 11 12
        20 21 22
        30 31 32
        40 41 42
        50 51 52
       */
        std::vector<int> result_data(A.shape()[0] * B.shape()[1], 0);
        for (size_t i = 0; i < A.shape()[0]; ++i) {
            for (size_t j = 0; j < B.shape()[1]; ++j) {
                for (size_t k = 0; k < A.shape()[1]; ++k) {
                    result_data[i * B.shape()[1] + j] += A.data()[i * A.shape()[1] + k] * B.data()[k * B.shape()[1] + j];
                }
            }
        }
        std::vector<size_t> result_shape = {A.shape()[0], B.shape()[1]};
        return Tensor(result_data, result_shape);

    }else if (A.shape().size()==1 && B.shape().size()==1){
        if (A.shape()[0] != B.shape()[0]) {
            throw std::invalid_argument("Vectors must be of the same length for dot product.");
        }
        int result = 0;
        for (size_t i = 0; i < A.shape()[0]; ++i) {
            result += A.data()[i] * B.data()[i];
        }
        return Tensor(std::vector<int>{result}, std::vector<size_t>{1});
    }else if (A.shape().size()==2 && B.shape().size()==1){
        if (A.shape()[1] != B.shape()[0]) {
            throw std::invalid_argument("Inner dimensions must match for matrix-vector multiplication.");
        }
        std::vector<int> result_data(A.shape()[0], 0);
        for (size_t i = 0; i < A.shape()[0]; ++i) {
            for (size_t j = 0; j < A.shape()[1]; ++j) {
                result_data[i] += A.data()[i * A.shape()[1] + j] * B.data()[j];
            }
        }
        std::vector<size_t> result_shape = {A.shape()[0]};
        return Tensor(result_data, result_shape);
    }else if (A.shape().size()==1 && B.shape().size()==2){
        if (A.shape()[0] != B.shape()[0]) {
            throw std::invalid_argument("Inner dimensions must match for vector-matrix multiplication.");
        }
        std::vector<int> result_data(B.shape()[1], 0);
        for (size_t i = 0; i < B.shape()[1]; ++i) {
            for (size_t j = 0; j < A.shape()[0]; ++j) {
                result_data[i] += A.data()[j] * B.data()[j * B.shape()[1] + i];
            }
        }
        std::vector<size_t> result_shape = {B.shape()[1]};
        return Tensor(result_data, result_shape);
    }else{
        // 算最后两维度
        if (A.shape()[A.shape().size()-1] != B.shape()[B.shape().size()-2]) {
            throw std::invalid_argument("Inner dimensions must match for tensor multiplication.");
        }
        int step_A=A.shape()[A.shape().size()-1]*A.shape()[A.shape().size()-2];
        int step_B=B.shape()[B.shape().size()-1]*B.shape()[B.shape().size()-2];
        if (std::vector<size_t>(A.shape().begin(), A.shape().end()-2) == std::vector<size_t>(B.shape().begin(), B.shape().end()-2)){
            int total_steps=1;
            for (size_t i = 0; i < A.shape().size()-2; ++i) {
                total_steps *= A.shape()[i];
            }
            std::vector<int> result_data(total_steps*A.shape()[A.shape().size()-2]*B.shape()[B.shape().size()-1], 0);
            for (int step = 0; step < total_steps; ++step) {
                for (size_t i = 0; i < A.shape()[A.shape().size()-2]; ++i) {
                    for (size_t j = 0; j < B.shape()[B.shape().size()-1]; ++j) {
                        for (size_t k = 0; k < A.shape()[A.shape().size()-1]; ++k) {
                            result_data[step*A.shape()[A.shape().size()-2]*B.shape()[B.shape().size()-1] + i * B.shape()[B.shape().size()-1] + j] += A.data()[step*step_A + i * A.shape()[A.shape().size()-1] + k] * B.data()[step*step_B + k * B.shape()[B.shape().size()-1] + j];
                        }
                    }
                }
            }
            std::vector<size_t> result_shape(A.shape().begin(), A.shape().end()-2);
            result_shape.push_back(A.shape()[A.shape().size()-2]);
            result_shape.push_back(B.shape()[B.shape().size()-1]);
            return Tensor(result_data, result_shape);
        }else{
            // TODO : 广播？
            ;
        }
    }
}