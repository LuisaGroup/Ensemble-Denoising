//
// Created by Mike Smith on 2020/11/10.
//

#pragma once

#include <opencv2/opencv.hpp>
#include "vector.h"

[[nodiscard]] inline auto blend(const std::vector<cv::Mat>& I, const std::vector<cv::Mat>& weights) -> cv::Mat {
    
    cv::Mat blended{I[0].rows, I[0].cols, CV_32FC3, cv::Scalar::all(0.0)};
    for (auto i = 0; i < I[0].rows * I[0].cols; i++) {
        Vector<3> color = Vector<3>::Zero();
        Vector<3> fallback_color = Vector<3>::Zero();
        Vector<3> sum_weights = Vector<3>::Zero();
        for (auto m = 0; m < I.size(); m++) {
            auto array_to_vector = [](std::array<float, 3> x) { return Vector<3>{x[0], x[1], x[2]}; };
            auto c = array_to_vector(reinterpret_cast<const std::array<float, 3> *>(I[m].data)[i]);
            Vector<3> w = array_to_vector(reinterpret_cast<const std::array<float, 3> *>(weights[m].data)[i]).cwiseMin(2.0).cwiseMax(-1.0);
            color.array() += c.array() * w.array();
            fallback_color.array() += c.array();
            sum_weights += w;
        }
        if (auto s = sum_weights.array().abs().sum(); s < 1e-4 || std::isnan(s)) {
            color = fallback_color * (1.0 / I.size());
        } else {
            color.array() /= sum_weights.array();
        }
        reinterpret_cast<std::array<float, 3> *>(blended.data)[i] = {
            static_cast<float>(color[0]),
            static_cast<float>(color[1]),
            static_cast<float>(color[2])};
    }
    
    return blended;
}
