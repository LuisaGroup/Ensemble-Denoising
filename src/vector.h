//
// Created by Mike Smith on 2020/11/10.
//

#pragma once

#include <Eigen/Dense>

template<size_t M, size_t N>
using Matrix = Eigen::Matrix<double, M, N>;

template<size_t N>
using Vector = Matrix<N, 1>;
