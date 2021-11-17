//
// Created by Mike Smith on 2020/11/10.
//

#pragma once

#include <array>
#include <iostream>
#include <thread>
#include <atomic>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/hal/interface.h>
#include <string>
#include <filesystem>
#include <type_traits>
#include <random>
#include <opencv2/opencv.hpp>
#include <cxxopts.hpp>

#include "vector.h"

#ifdef ENABLE_FBSTAB
#include <fbstab/fbstab_dense.h>

template<size_t N>
[[nodiscard]] inline auto fbstab_solve(Matrix<N, N> H, Vector<N> f) noexcept -> Vector<N> {
    
    using namespace fbstab;
    
    static constexpr auto n = static_cast<int>(N);
    static constexpr auto m = static_cast<int>(1);
    static constexpr auto q = static_cast<int>(N);
    
    static thread_local FBstabDense::Variable x0{n, m, q};
    
    static thread_local auto &&data = []() -> auto & {
        static thread_local FBstabDense::ProblemData data{n, m, q};
        data.G = Matrix<1, N>::Ones();
        data.h = Matrix<1, 1>::Ones();
        data.A = -Matrix<N, N>::Identity();
        data.b = Vector<N>::Zero();
        return data;
    }();
    
    static thread_local auto &&solver = []() -> auto & {
        static thread_local FBstabDense solver(n, m, q);
        FBstabDense::Options opts = FBstabDense::DefaultOptions();
        opts.display_level = Display::OFF;
        opts.abs_tol = 1e-10;
        solver.UpdateOptions(opts);
        return solver;
    }();
    
    data.H = H;
    data.f = -f;
    if (auto out = solver.Solve(data, &x0); out.eflag == ExitFlag::SUCCESS) {
        Vector<N> w = x0.z.array().min(1.0).max(0.0);
        return w * (1.0 / w.sum());
    }
    return Vector<N>::Ones() * (1.0 / N);
}

template<size_t N>
[[nodiscard]] inline auto optimize_fbstab(Vector<N> IA, Vector<N> IB, double I0A, double I0B, bool solve_full) noexcept -> std::tuple<Vector<N>, Vector<N>, Vector<N>> {
    
    Matrix<N, N> AA = IA * IA.transpose();
    Matrix<N, N> AB = IB * IB.transpose();
    Vector<N> bA = I0B * IA;
    Vector<N> bB = I0A * IB;
    
    if (solve_full) {
        auto w = fbstab_solve<N>(AA + AB, bA + bB);
        return {w, w, w};
    }
    
    auto wA = fbstab_solve<N>(AA, bA);
    auto wB = fbstab_solve<N>(AB, bB);
    return {wA, wA, wB};
}

#endif

template<size_t N, size_t num_iter = 1024>
[[nodiscard]] inline auto optimize_brute(Vector<N> IA, Vector<N> IB, double I0A, double I0B, bool solve_full) -> std::tuple<Vector<N>, Vector<N>, Vector<N>, int> {
    
    static thread_local std::default_random_engine random{std::random_device{}()};
    auto select = [] { return std::uniform_int_distribution<uint32_t>{0, N - 1}(random); };
    auto uniform = [] { return std::uniform_real_distribution<double>{0.0, 1.0}(random); };
    
    static constexpr auto alpha_eps = 1e-10;
    static constexpr auto a_eps = 1e-6;
    
    auto rand_w = [uniform] {
        Vector<N> w;
        for (auto i = 0; i < N; i++) { w[i] = uniform(); }
        w *= 1.0 / w.sum();
        return w;
    };
    
    if (solve_full) {
        Vector<N> w = rand_w();
        auto iter_count = 0;
        for (auto i = 0, ineffective_count = 0; i < num_iter && ineffective_count <= 2 * N; i++) {
            auto sel = select();
            auto I1A = w.dot(IA);
            auto I1B = w.dot(IB);
            auto I2A = IA[sel];
            auto I2B = IB[sel];
            auto a = std::max((I1A - I2A) * (I1A - I2A) + (I1B - I2B) * (I1B - I2B), a_eps);
            auto b = (I1A - I0B) * (I1A - I2A) + (I1B - I0A) * (I1B - I2B);
            auto alpha = std::clamp(b / a, 0.0, 1.0);
            ineffective_count = alpha >= alpha_eps ? 0 : ineffective_count + 1;
            w *= 1.0 - alpha;
            w[sel] += alpha;
            w *= 1.0 / w.sum();
            iter_count++;
        }
        return {w, w, w, iter_count};
    }
    
    Vector<N> wA = rand_w();
    for (auto i = 0, ineffective_count = 0; i < num_iter && ineffective_count <= 2 * N; i++) {
        auto sel = select();
        auto I1A = wA.dot(IA);
        auto I2A = IA[sel];
        auto a = std::max((I1A - I2A) * (I1A - I2A), a_eps);
        auto b = (I1A - I0B) * (I1A - I2A);
        auto alpha = std::clamp(b / a, 0.0, 1.0);
        ineffective_count = alpha >= alpha_eps ? 0 : ineffective_count + 1;
        wA *= 1.0 - alpha;
        wA[sel] += alpha;
        wA *= 1.0 / wA.sum();
    }
    
    Vector<N> wB = rand_w();
    for (auto i = 0, ineffective_count = 0; i < num_iter && ineffective_count <= 2 * N; i++) {
        auto sel = select();
        auto I1B = wB.dot(IB);
        auto I2B = IB[sel];
        auto a = std::max((I1B - I2B) * (I1B - I2B), a_eps);
        auto b = (I1B - I0A) * (I1B - I2B);
        auto alpha = std::clamp(b / a, 0.0, 1.0);
        ineffective_count = alpha >= alpha_eps ? 0 : ineffective_count + 1;
        wB *= 1.0 - alpha;
        wB[sel] += alpha;
        wB *= 1.0 / wB.sum();
    }
    return {wA, wA, wB, 0};
}

enum class Solver {
    BRUTE,
#ifdef ENABLE_FBSTAB
    FBSTAB
#endif
};

template<size_t N>
[[nodiscard]] inline std::tuple<std::vector<cv::Mat>, std::vector<cv::Mat>, std::vector<cv::Mat>, cv::Mat>
solve_weights_specialized(const cv::Mat &I0A, const cv::Mat &I0B,
                          const std::vector<cv::Mat> &IA,
                          const std::vector<cv::Mat> &IB,
                          Solver solver,
                          bool solve_full,
                          double beta = 0.0) {
    
    std::vector<cv::Mat> weights;
    std::vector<cv::Mat> weightsA;
    std::vector<cv::Mat> weightsB;
    
    for (auto i = 0; i < N; i++) {
        if (solve_full) {
            weights.emplace_back(I0A.rows, I0A.cols, CV_32FC3, cv::Scalar::all(0.0));
        } else {
            weightsA.emplace_back(I0A.rows, I0A.cols, CV_32FC3, cv::Scalar::all(0.0));
            weightsB.emplace_back(I0A.rows, I0A.cols, CV_32FC3, cv::Scalar::all(0.0));
        }
    }
    
    cv::Mat statistics;
    if (solver == Solver::BRUTE) { statistics.create(I0A.rows, I0B.cols, CV_32FC3); }
    
    auto num_workers = std::thread::hardware_concurrency();
    std::vector<std::thread> workers;
    workers.reserve(num_workers);
    std::atomic<int> curr_row{0};
    
    auto h = I0A.rows;
    auto w = I0A.cols;
    
    for (auto wid = 0; wid < num_workers; wid++) {
        workers.emplace_back(std::thread{[&](int id) {
            auto pitch = w * 3;
            for (auto row = curr_row++; row < h; row = curr_row++) {
                for (auto i = row * pitch; i < row * pitch + pitch; i++) {
                    auto i0a = reinterpret_cast<const float *>(I0A.data)[i];
                    auto i0b = reinterpret_cast<const float *>(I0B.data)[i];
                    Vector<N> ia;
                    Vector<N> ib;
                    for (auto m = 0u; m < N; m++) {
                        ia[m] = reinterpret_cast<const float *>(IA[m].data)[i];
                        ib[m] = reinterpret_cast<const float *>(IB[m].data)[i];
                    }
                    
                    Vector<N> w;
                    Vector<N> wA;
                    Vector<N> wB;
                    auto iter_count = 0;
                    
                    switch (solver) {
                        case Solver::BRUTE:
                            std::tie(w, wA, wB, iter_count) = optimize_brute<N>(ia, ib, i0a, i0b, solve_full);
                            break;
#ifdef ENABLE_FBSTAB
                        case Solver::FBSTAB:
                            std::tie(w, wA, wB) = optimize_fbstab<N>(ia, ib, i0a, i0b, solve_full);
                            break;
#endif
                        default:
                            break;
                    }
                    
                    for (auto m = 0u; m < N; m++) {
                        if (solve_full) {
                            reinterpret_cast<float *>(weights[m].data)[i] = w[m];
                            if (solver == Solver::BRUTE) {
                                reinterpret_cast<float *>(statistics.data)[i] = static_cast<float>(iter_count);
                            }
                        } else {
                            reinterpret_cast<float *>(weightsA[m].data)[i] = wA[m];
                            reinterpret_cast<float *>(weightsB[m].data)[i] = wB[m];
                        }
                    }
                }
            }
        }, wid});
    }
    
    for (auto &&worker : workers) { worker.join(); }
    return {weights, weightsA, weightsB, statistics};
}
