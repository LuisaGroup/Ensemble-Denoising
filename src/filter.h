//
// Created by Mike Smith on 2020/11/10.
//

#pragma once

#include <opencv2/imgproc.hpp>
#include <thread>
#include <atomic>

#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/opencv.hpp>
#include "vector.h"

template<int radius>
[[nodiscard]] inline cv::Mat filter_bilateral(
    const cv::Mat &target_image,
    const cv::Mat &target_variance,
    float kc,
    float sigma_color,
    float sigma_spatial,
    const std::vector<cv::Mat> &joint_images,
    const std::vector<float> &joint_sigmas) {
    
    auto width = target_image.cols;
    auto height = target_image.rows;
    
    using Pixel = Vector<3>;
    
    auto load = [width, height](const cv::Mat &image, int px, int py) noexcept {
        
        auto wrap = [width, height](int x, int y) noexcept {
            x = x < 0 ? -x : (x >= width ? 2 * width - 2 - x : x);
            y = y < 0 ? -y : (y >= height ? 2 * height - 2 - y : y);
            return std::make_pair(x, y);
        };
        
        auto[x, y] = wrap(px, py);
        auto color = reinterpret_cast<const std::array<float, 3> *>(image.data)[y * width + x];
        return Pixel{color[0], color[1], color[2]};
    };
    
    cv::Mat filtered{height, width, CV_32FC3, cv::Scalar::all(0.0f)};
    for (auto py = 0; py < height; py++) {
        for (auto px = 0; px < width; px++) {
            Pixel sum_color{0.0, 0.0, 0.0};
            auto sum_weight = 0.0;
            for (auto dy = -radius; dy <= radius; dy++) {
                for (auto dx = -radius; dx <= radius; dx++) {
                    auto qx = px + dx;
                    auto qy = py + dy;
                    auto spatial_distance = (dx * dx + dy * dy) / (2.0 * sigma_spatial * sigma_spatial);
                    Pixel color_p = load(target_image, px, py);
                    Pixel color_q = load(target_image, qx, qy);
                    Pixel var_p = load(target_variance, px, py);
                    Pixel var_q = load(target_variance, qx, qy);
                    Pixel var_cancel = kc * (var_p + var_p.cwiseMin(var_q));
                    Pixel var_norm = (var_p + var_q).cwiseMax(1e-4) * (sigma_color * sigma_color);
                    Pixel noisy_distance = ((color_p - color_q).array().square() - var_cancel.array()).max(0.0) / var_norm.array();
                    auto scaled_squared_diff = spatial_distance + noisy_distance.sum();
                    for (auto i = 0; i < joint_images.size(); i++) {
                        auto f_p = load(joint_images[i], px, py);
                        auto f_q = load(joint_images[i], qx, qy);
                        auto sigma = joint_sigmas[i];
                        auto squared_diff = (f_p.array() - f_q.array()).square().sum();
                        scaled_squared_diff += squared_diff / (2.0 * sigma * sigma);
                    }
                    auto w = std::min(std::exp(-scaled_squared_diff), 10.0);
                    sum_color += w * color_q;
                    sum_weight += w;
                }
            }
            Pixel color = sum_color / std::max(sum_weight, 1e-6);
            auto data = reinterpret_cast<std::array<float, 3> *>(filtered.data);
            data[py * width + px] = {static_cast<float>(color[0]), static_cast<float>(color[1]), static_cast<float>(color[2])};
        }
    }
    
    return filtered;
}

template<int radius>
[[nodiscard]] inline cv::Mat remove_outliers(const cv::Mat &image, float tol = 1.5) noexcept {
    
    auto width = image.cols;
    auto height = image.rows;

    static constexpr auto f = 2 * radius + 1;
    
    cv::Mat result;
    cv::GaussianBlur(image, result, {f, f}, 0.0);
    
    auto src_pixels = reinterpret_cast<const std::array<float, 3> *>(image.data);
    auto dst_pixels = reinterpret_cast<std::array<float, 3> *>(result.data);
    for (auto y = 0; y < height; y++) {
        for (auto x = 0; x < width; x++) {
            
            std::array<float, 3> sum{0.0f, 0.0f, 0.0f};
            std::array<float, 3> sum_sqr{0.0f, 0.0f, 0.0f};
            
            for (auto dy = -radius; dy <= radius; dy++) {
                for (auto dx = -radius; dx <= radius; dx++) {
                    
                    if (dx == 0 && dy == 0) { continue; }
                    
                    auto px = cv::borderInterpolate(x + dx, width, cv::BORDER_REFLECT101);
                    auto py = cv::borderInterpolate(y + dy, height, cv::BORDER_REFLECT101);
                    auto p = src_pixels[py * width + px];
    
                    for (auto channel = 0; channel < 3; channel++) {
                        auto c = p[channel];
                        sum[channel] += c;
                        sum_sqr[channel] += c * c;
                    }
                }
            }
            
            static constexpr auto n = f * f - 1;
            static constexpr auto inv_n = 1.0 / n;
            auto c = src_pixels[y * width + x];
            auto good = true;
            std::array<float, 3> mean_color{};
            for (auto channel = 0; channel < 3; channel++) {
                auto mean = sum[channel] * inv_n;
                auto mean_sqr = sum_sqr[channel] * inv_n;
                auto sigma = std::sqrt((1.0f - inv_n) * (mean_sqr - mean * mean));
                mean_color[channel] = mean;
                if (std::abs(c[channel] - mean) >= tol * sigma) { good = false; }
            }
            if (good) { dst_pixels[y * width + x] = c; }
        }
    }
    
    return result;
}

template<int radius, size_t J>
[[nodiscard]] inline auto filter_cross_bilateral(
    const std::vector<cv::Mat> &target_images,
    const cv::Mat &guide_image,
    const cv::Mat &guide_variance,
    float sigma_guide,
    float sigma_spatial,
    const std::array<cv::Mat, J> &joint_images,
    const std::array<float, J> &joint_sigmas) {
    
    auto width = guide_image.cols;
    auto height = guide_image.rows;
    
    auto read = [](const cv::Mat &image, int index) noexcept -> Vector<3> {
        auto color = reinterpret_cast<const std::array<float, 3> *>(image.data)[index];
        return {color[0], color[1], color[2]};
    };

    auto N = target_images.size();
    std::vector<cv::Mat> filtered_images(N);
    for (auto &&image : filtered_images) {
        image.create(height, width, CV_32FC3);
    }

    auto k_spatial = -0.5 / (sigma_spatial * sigma_spatial);
    auto k_guide = -0.5 / (sigma_guide * sigma_guide);

    auto num_workers = std::thread::hardware_concurrency();
    std::vector<std::thread> workers;
    workers.reserve(num_workers);
    std::atomic<int> curr_y{0};

    for (auto wid = 0; wid < num_workers; wid++) {
        workers.emplace_back(std::thread{[&](int id) {
            std::vector<Vector<3>> sum_colors(N);
            for (auto py = curr_y++; py < height; py = curr_y++) {
                // if (id == 0) { std::cout << "Progress: " << py + 1 << "/" << height << std::endl; }
                for (auto px = 0; px < width; px++) {

                    auto sum_weight = 0.0;
                    for (auto &&c : sum_colors) { c = Vector<3>::Zero(); }

                    auto index_p = py * width + px;
                    std::array<Vector<3>, J> joint_p{};
                    std::array<double, J> joint_k{};
                    for (auto j = 0; j < J; j++) {
                        joint_p[j] = read(joint_images[j], index_p);
                        joint_k[j] = -0.5 / joint_sigmas[j];
                    }
                    Vector<3> guide_p = read(guide_image, index_p);
                    Vector<3> var_p = read(guide_variance, index_p);
                    for (auto dy = -radius; dy <= radius; dy++) {
                        for (auto dx = -radius; dx <= radius; dx++) {
                            auto weight = std::exp((dx * dx + dy * dy) * k_spatial);
                            auto qx = cv::borderInterpolate(px + dx, width, cv::BORDER_REFLECT101);
                            auto qy = cv::borderInterpolate(py + dy, height, cv::BORDER_REFLECT101);
                            auto index_q = qy * width + qx;
                            Vector<3> guide_q = read(guide_image, index_q);
                            Vector<3> var_q = read(guide_variance, index_q);
                            weight *= std::exp(((guide_p - guide_q).array().square() / (var_p + var_q).array().max(1e-6)).sum() * k_guide);
                            for (auto i = 0; i < J; i++) {
                                auto joint_q = read(joint_images[i], index_q);
                                weight *= std::exp((joint_p[i] - joint_q).array().square().sum() * joint_k[i]);
                            }
                            for (auto i = 0; i < N; i++) { sum_colors[i] += weight * read(target_images[i], index_q); }
                            sum_weight += weight;
                        }
                    }

                    auto inv_sum = 1.0 / sum_weight;
                    for (auto i = 0; i < N; i++) {
                        auto color = inv_sum * sum_colors[i];
                        reinterpret_cast<std::array<float, 3> *>(filtered_images[i].data)[index_p] = {
                            static_cast<float>(color[0]),
                            static_cast<float>(color[1]),
                            static_cast<float>(color[2])};
                    }
                }
            }
        }, wid});
    }

    for (auto &&worker : workers) { worker.join(); }
    return filtered_images;
}
