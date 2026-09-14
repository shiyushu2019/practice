#include<vector>
#include<iostream>
#include <stdexcept>

#define DataType double

class Tensor {
    private:
        std::vector<DataType> data_;
        std::vector<size_t> shape_;
    public:

        Tensor(std::vector<DataType> data,std::vector<size_t> shape)
            : data_(std::move(data)), shape_(std::move(shape)) {
                size_t total = 1;
                for (auto d : shape_) total *= d;
                if (total != data_.size())
                    throw std::invalid_argument("shape does not match data size");
            }
        explicit Tensor(std::vector<DataType> data)
            : data_(std::move(data)), shape_(std::vector<size_t>{data_.size()}) {}

        const std::vector<size_t>& shape() const { return shape_; }
        const std::vector<DataType>&    data()  const { return data_;  }

        void reshape(std::vector<size_t> new_shape){
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

Tensor matmul(const Tensor& A, const Tensor& B) {
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
        std::vector<DataType> result_data(A.shape()[0] * B.shape()[1], 0);
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
        DataType result = 0;
        for (size_t i = 0; i < A.shape()[0]; ++i) {
            result += A.data()[i] * B.data()[i];
        }
        return Tensor(std::vector<DataType>{result}, std::vector<size_t>{1});
    }else if (A.shape().size()==2 && B.shape().size()==1){
        if (A.shape()[1] != B.shape()[0]) {
            throw std::invalid_argument("Inner dimensions must match for matrix-vector multiplication.");
        }
        std::vector<DataType> result_data(A.shape()[0], 0);
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
        std::vector<DataType> result_data(B.shape()[1], 0);
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
        size_t step_A=A.shape()[A.shape().size()-1]*A.shape()[A.shape().size()-2];
        size_t step_B=B.shape()[B.shape().size()-1]*B.shape()[B.shape().size()-2];
        if (std::vector<size_t>(A.shape().begin(), A.shape().end()-2) == std::vector<size_t>(B.shape().begin(), B.shape().end()-2)){
            size_t total_steps=1;
            for (size_t i = 0; i < A.shape().size()-2; ++i) {
                total_steps *= A.shape()[i];
            }
            std::vector<DataType> result_data(total_steps*A.shape()[A.shape().size()-2]*B.shape()[B.shape().size()-1], 0);
            for (size_t step = 0; step < total_steps; ++step) {
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