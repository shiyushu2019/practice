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

        void transpose(int dim1, int dim2)  {
            int ndim = static_cast<int>(shape_.size());
            if (dim1 < 0 || dim1 >= ndim || dim2 < 0 || dim2 >= ndim) {
                throw std::invalid_argument("Dimension indices are out of bounds.");
            }
            if (dim1 == dim2) {
                return;
            }

            std::vector<size_t> new_shape = shape_;
            std::swap(new_shape[dim1], new_shape[dim2]);

            std::vector<size_t> old_strides(ndim, 1);
            for (int i = ndim - 2; i >= 0; --i) {
                old_strides[i] = old_strides[i + 1] * shape_[i + 1];
            }

            std::vector<size_t> new_strides(ndim, 1);
            for (int i = ndim - 2; i >= 0; --i) {
                new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
            }

            std::vector<DataType> new_data(data_.size());
            std::vector<size_t> coords(ndim);

            for (size_t i = 0; i < data_.size(); ++i) {
                size_t idx = i;
                for (int j = 0; j < ndim; ++j) {
                    coords[j] = idx / old_strides[j];
                    idx %= old_strides[j];
                }

                std::swap(coords[dim1], coords[dim2]);

                size_t new_index = 0;
                for (int j = 0; j < ndim; ++j) {
                    new_index += coords[j] * new_strides[j];
                }

                new_data[new_index] = data_[i];
            }

            data_ = std::move(new_data);
            shape_ = std::move(new_shape);
        }

        void sum(int dim) {
            int ndim = static_cast<int>(shape_.size());
            if (dim < 0 || dim >= ndim) {
                throw std::invalid_argument("Dimension index is out of bounds.");
            }

            std::vector<size_t> new_shape = shape_;
            new_shape.erase(new_shape.begin() + dim);

            size_t new_size = 1;
            for (auto d : new_shape) new_size *= d;

            std::vector<DataType> new_data(new_size, 0);

            std::vector<size_t> old_strides(ndim, 1);
            for (int i = ndim - 2; i >= 0; --i) {
                old_strides[i] = old_strides[i + 1] * shape_[i + 1];
            }

            std::vector<size_t> new_strides(new_shape.size(), 1);
            for (int i = static_cast<int>(new_shape.size()) - 2; i >= 0; --i) {
                new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
            }

            std::vector<size_t> coords(ndim);

            for (size_t i = 0; i < data_.size(); ++i) {
                size_t idx = i;
                for (int j = 0; j < ndim; ++j) {
                    coords[j] = idx / old_strides[j];
                    idx %= old_strides[j];
                }

                size_t new_index = 0;
                for (int j = 0, k = 0; j < ndim; ++j) {
                    if (j != dim) {
                        new_index += coords[j] * new_strides[k++];
                    }
                }

                new_data[new_index] += data_[i];
            }

            data_= std::move(new_data);
            shape_= std::move(new_shape);
        }

        Tensor operator+(const Tensor& other) const {
            if (shape_ != other.shape_) {
                throw std::invalid_argument("Shapes must match for addition.");
            }
            std::vector<DataType> result_data(data_.size());
            for (size_t i = 0; i < data_.size(); ++i) {
                result_data[i] = data_[i] + other.data_[i];
            }
            return Tensor(std::move(result_data), shape_);
        }

        static Tensor zeros(std::vector<size_t> shape) {
            size_t total = 1;
            for (size_t d : shape) total *= d;
            return Tensor(std::vector<DataType>(total, 0), std::move(shape));
        }
};

Tensor matmul(const Tensor& A, const Tensor& B) {
    const auto& a_shape = A.shape();
    const auto& b_shape = B.shape();
    if (a_shape.empty() || b_shape.empty()) {
        throw std::invalid_argument("matmul requires tensors with at least one dimension.");
    }

    const bool a_vector = a_shape.size() == 1;
    const bool b_vector = b_shape.size() == 1;
    const size_t a_rank = a_shape.size();
    const size_t b_rank = b_shape.size();
    const size_t a_prefix_rank = a_vector ? 0 : a_rank - 2;
    const size_t b_prefix_rank = b_vector ? 0 : b_rank - 2;
    const size_t m = a_vector ? 1 : a_shape[a_rank - 2];
    const size_t k = a_shape[a_rank - 1];
    const size_t n = b_vector ? 1 : b_shape[b_rank - 1];

    if (k != (b_vector ? b_shape[0] : b_shape[b_rank - 2])) {
        throw std::invalid_argument("Inner dimensions must match for matrix multiplication.");
    }

    std::vector<size_t> a_prefix(a_shape.begin(), a_shape.begin() + a_prefix_rank);
    std::vector<size_t> b_prefix(b_shape.begin(), b_shape.begin() + b_prefix_rank);
    const size_t prefix_rank = std::max(a_prefix.size(), b_prefix.size());
    std::vector<size_t> result_prefix(prefix_rank, 1);
    for (size_t axis = 0; axis < prefix_rank; ++axis) {
        const size_t a_dim = axis < prefix_rank - a_prefix.size()
            ? 1 : a_prefix[axis - (prefix_rank - a_prefix.size())];
        const size_t b_dim = axis < prefix_rank - b_prefix.size()
            ? 1 : b_prefix[axis - (prefix_rank - b_prefix.size())];
        if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
            throw std::invalid_argument("Broadcast dimensions do not match for matrix multiplication.");
        }
        result_prefix[axis] = a_dim == 1 ? b_dim : a_dim;
    }

    auto batch_offset = [](const std::vector<size_t>& shape,
                           const std::vector<size_t>& result_shape,
                           size_t batch_index) {
        std::vector<size_t> padded(result_shape.size(), 1);
        const size_t padding = result_shape.size() - shape.size();
        for (size_t axis = 0; axis < shape.size(); ++axis) {
            padded[padding + axis] = shape[axis];
        }

        size_t offset = 0;
        size_t stride = 1;
        for (size_t axis = result_shape.size(); axis > 0; --axis) {
            const size_t current = axis - 1;
            const size_t coordinate = batch_index % result_shape[current];
            batch_index /= result_shape[current];
            if (padded[current] != 1) {
                offset += coordinate * stride;
            }
            stride *= padded[current];
        }
        return offset;
    };

    size_t batch_count = 1;
    for (size_t dim : result_prefix) {
        batch_count *= dim;
    }
    std::vector<DataType> result_data(batch_count * m * n, 0);
    const size_t a_matrix_size = m * k;
    const size_t b_matrix_size = k * n;
    for (size_t batch = 0; batch < batch_count; ++batch) {
        const size_t a_offset = a_vector ? 0 : batch_offset(a_prefix, result_prefix, batch) * a_matrix_size;
        const size_t b_offset = b_vector ? 0 : batch_offset(b_prefix, result_prefix, batch) * b_matrix_size;
        for (size_t row = 0; row < m; ++row) {
            for (size_t column = 0; column < n; ++column) {
                for (size_t inner = 0; inner < k; ++inner) {
                    result_data[batch * m * n + row * n + column] +=
                        A.data()[a_offset + row * k + inner] *
                        B.data()[b_offset + inner * n + column];
                }
            }
        }
    }

    std::vector<size_t> result_shape = result_prefix;
    if (!a_vector) {
        result_shape.push_back(m);
    }
    if (!b_vector) {
        result_shape.push_back(n);
    }
    return Tensor(std::move(result_data), std::move(result_shape));
}