#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    size_t iters = m / batch; // 迭代次数
    for(size_t i = 0; i < iters; i++){
        // 计算矩阵相乘
        float* Z = new float[batch * k]; // 中间矩阵 (batch, k)
        for(size_t r = i * batch; r < (i+1) * batch; r++){
            for(size_t c = 0; c < k; c++){
                float sum = 0.0f;
                for(size_t inter = 0; inter < n; inter++){
                    sum += X[r*n + inter] * theta[inter * k + c];
                }
                Z[(r - i*batch)*k + c] = sum;
            }
        }
        float* ZSum = new float[batch];
        for(int iz = 0; iz < batch; iz++){
            for(int j = 0; j < k; j++){
                ZSum[iz] += std::expf(Z[iz*k + j]);
            }
        }
        // normalize
        for(int iz = 0; iz < batch; iz++){
            for(int j = 0; j < k; j++){
                Z[iz*k + j] = std::exp(Z[iz*k + j]) / ZSum[iz];
            }
        }
        // Z - Iy
        for(int iz = 0; iz < batch; iz++){
            int yIndex = y[i*batch + iz];
            Z[iz * k + yIndex] -= 1.0f;
        }
        // update theta
        for(size_t iz = 0; iz < n; iz++){
            for(size_t jz = 0; jz < k; jz++){
                float sum = 0.0f;
                for(size_t im = 0; im < batch; im++){
                    sum += X[im*n + i*batch*n + iz] * Z[im * k + jz];

                }
                theta[iz*k + jz] -= (lr / batch * sum );
            }
        }
        delete [] ZSum;
        delete [] Z;
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
