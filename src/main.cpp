#include "cxxopts.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <solve.h>
#include <filter.h>
#include <blend.h>
#include <string>

int main(int argc, char *argv[]) {

    cxxopts::Options options{"EnsembleDenoising"};
    options.add_options()
               ("i,input-dir", "Directory containing input buffers", cxxopts::value<std::filesystem::path>(), "<path>")
               ("o,output-dir", "Directory for writing weights and variances of weights", cxxopts::value<std::filesystem::path>(), "<path>")
               ("m,methods", "Base denoising methods for blending, separated by comma without blanks (e.g., -m oidn,optix,mcgan)", cxxopts::value<std::vector<std::string>>(), "<name>[,<name>[,...]]")
               ("s,solver", "Select solver to use (Candidates: brute, fbstab)", cxxopts::value<std::string>()->default_value("brute"), "<name>")
               ("d,debug", "Debug mode, outputs intermediate buffers", cxxopts::value<bool>())
               ("q,quiet", "Output nothing but blending", cxxopts::value<bool>())
               ("statistics", "Enable statistics", cxxopts::value<bool>())
               ("p,passes", "Number of passes in weight solving", cxxopts::value<int>()->default_value("1"), "<n>")
               ("h,help", "Show help and exit", cxxopts::value<bool>());

    auto arguments = options.parse(argc, argv);
    if (arguments["help"].count()) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    auto standardize = [](const cv::Mat &image) -> cv::Mat {
        cv::Scalar mean;
        cv::Scalar stddev;
        cv::meanStdDev(image, mean, stddev);
        stddev[0] = std::max(stddev[0], 1e-3);
        stddev[1] = std::max(stddev[1], 1e-3);
        stddev[2] = std::max(stddev[2], 1e-3);
        cv::Mat result = (image - mean) / stddev;
        return cv::min(cv::max(result, cv::Scalar::all(-3.0)), cv::Scalar::all(3.0));
    };

    auto folder = arguments["i"].as<std::filesystem::path>();
    auto imread = [&folder, &standardize](const std::filesystem::path &name, bool should_standardize = false) -> cv::Mat {
        auto canonical_path = std::filesystem::canonical(folder / name);
        auto image = cv::imread(canonical_path.string(), cv::IMREAD_UNCHANGED);
        if (image.empty()) {
            std::cout << "Empty: " << canonical_path << std::endl;
            exit(-1);
        }
        if (image.channels() == 4) { cv::cvtColor(image, image, cv::COLOR_BGRA2BGR); }
        else if (image.channels() == 1) { cv::cvtColor(image, image, cv::COLOR_GRAY2BGR); }
        auto pixels = reinterpret_cast<float *>(image.data);
        auto lb = name.string().find("normal") != std::string::npos ? -1.0f : 0.0f;
        for (auto i = 0u; i < image.rows * image.cols * image.channels(); i++) {
            auto value = std::max(pixels[i], lb);
            if (std::isnan(value)) { value = 0.0f; }
            pixels[i] = value;
        }
        if (should_standardize) { return standardize(image); }
        return image;
    };

    auto imwrite = [](const std::filesystem::path &name, const cv::Mat &image) noexcept {
        cv::imwrite(name.string(), image);
    };

    auto debug_mode = arguments["debug"].as<bool>();
    auto quiet = arguments["quiet"].as<bool>();
    auto statistics = arguments["statistics"].as<bool>();

    auto output_folder = arguments["o"].as<std::filesystem::path>();
    if (!std::filesystem::exists(output_folder)) { std::filesystem::create_directories(output_folder); }

    auto I0A = imread("colorA.exr");
    auto I0B = imread("colorB.exr");

    auto methods = arguments["m"].as<std::vector<std::string>>();

    auto N = methods.size();
    std::vector<cv::Mat> I(N);
    std::vector<cv::Mat> IA(N);
    std::vector<cv::Mat> IB(N);

    for (auto i = 0u; i < N; i++) {
        auto &&method = methods[i];
        auto prefix = std::string{method}.append("/").append(method);
        I[i] = imread(prefix + ".exr", false);
        IA[i] = imread(prefix + "A.exr", false);
        IB[i] = imread(prefix + "B.exr", false);
    }

    auto albedo = imread("albedo.exr");
    auto normal = imread("normal.exr", true);

    std::vector<cv::Mat> weights;
    std::vector<cv::Mat> weightsA;
    std::vector<cv::Mat> weightsB;
    cv::Mat stats;

    auto solver_str = arguments["solver"].as<std::string>();
    auto solver = [solver_str] {
        if (solver_str == "brute") { return Solver::BRUTE; }
#ifdef ENABLE_FBSTAB
        if (solver_str == "fbstab") { return Solver::FBSTAB; }
#endif // ENABLE_FBSTAB
        std::cerr << "WARNING: Unknown solver: '" << solver_str << "', using 'brute' as fallback..." << std::endl;
        return Solver::BRUTE;
    }();

    auto t0 = std::chrono::high_resolution_clock::now();

    auto beta = arguments["beta"].as<double>();
    auto num_passes = arguments["passes"].as<int>();
    cv::Mat noisy;
    cv::Mat noisyA;
    cv::Mat noisyB;
    for (auto i = 0; i < num_passes; i++) {
        if (!quiet) { std::cout << "Pass " << i + 1 << "/" << num_passes << std::endl; }
        auto solve_full = i == num_passes - 1;
        if (N == 1) { std::tie(weights, weightsA, weightsB, stats) = solve_weights_specialized<1>(I0A, I0B, IA, IB, solver, solve_full, beta); }
        else if (N == 2) { std::tie(weights, weightsA, weightsB, stats) = solve_weights_specialized<2>(I0A, I0B, IA, IB, solver, solve_full, beta); }
        else if (N == 3) { std::tie(weights, weightsA, weightsB, stats) = solve_weights_specialized<3>(I0A, I0B, IA, IB, solver, solve_full, beta); }
        else if (N == 4) { std::tie(weights, weightsA, weightsB, stats) = solve_weights_specialized<4>(I0A, I0B, IA, IB, solver, solve_full, beta); }
        else if (N == 5) { std::tie(weights, weightsA, weightsB, stats) = solve_weights_specialized<5>(I0A, I0B, IA, IB, solver, solve_full, beta); }
        else if (N == 6) { std::tie(weights, weightsA, weightsB, stats) = solve_weights_specialized<6>(I0A, I0B, IA, IB, solver, solve_full, beta); }
        else if (N == 7) { std::tie(weights, weightsA, weightsB, stats) = solve_weights_specialized<7>(I0A, I0B, IA, IB, solver, solve_full, beta); }
        else if (N == 8) { std::tie(weights, weightsA, weightsB, stats) = solve_weights_specialized<8>(I0A, I0B, IA, IB, solver, solve_full, beta); }
        else if (N == 9) { std::tie(weights, weightsA, weightsB, stats) = solve_weights_specialized<9>(I0A, I0B, IA, IB, solver, solve_full, beta); }
        else if (N == 10) { std::tie(weights, weightsA, weightsB, stats) = solve_weights_specialized<10>(I0A, I0B, IA, IB, solver, solve_full, beta); }
        else { throw std::runtime_error{"Too many base denoisers!"}; }
        if (solve_full) {
            I0A = blend(IA, weights);
            I0B = blend(IB, weights);
        } else {
            I0A = blend(IA, weightsA);
            I0B = blend(IB, weightsB);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    if (statistics) {
        using namespace std::chrono_literals;
        auto dt = (t1 - t0) / 1ns * 1e-9;
        std::cout << "Time: " << dt << "s" << std::endl;
        std::ofstream time_file{output_folder / "time"};
        time_file << dt << std::endl;
        if (solver == Solver::BRUTE) { imwrite(output_folder / "iterations.exr", stats); }
    }

    if (!quiet) { std::cout << "Applying noisy weights..." << std::endl; }
    cv::Mat result = blend(I, weights);
    if (!quiet || debug_mode) { imwrite(output_folder / "blend-raw.exr", result); }

    if (!quiet) { std::cout << "Filtering weights..." << std::endl; }
    std::array<cv::Mat, 2> joint_images{normal, albedo};
    std::array<float, 2> joint_sigmas{0.15f, 0.25f};
    cv::Mat var_guide = 0.5 * (I0A - I0B).mul(I0A - I0B);
    cv::GaussianBlur(var_guide, var_guide, {9, 9}, 0.0);
    if (debug_mode) { imwrite(folder / "guide-var.exr", var_guide); }
    auto filtered_weights = filter_cross_bilateral<7>(weights, result, var_guide, 2.0f, 3.0f, joint_images, joint_sigmas);
    for (auto &&f : filtered_weights) { f = remove_outliers<3>(f, 2.5); }
    if (!quiet || debug_mode) {
        for (auto i = 0; i < N; i++) {
            auto m = methods[i];
            imwrite(output_folder / ("weight-" + m + "-filtered.exr"), filtered_weights[i]);
        }
    }

    if (!quiet) { std::cout << "Applying filtered weights..." << std::endl; }
    imwrite(output_folder / "blend.exr", blend(I, filtered_weights));
}
