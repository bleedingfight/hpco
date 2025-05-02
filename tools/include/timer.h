#pragma once
#include <chrono>
#include <functional>
#include <iostream>

class Timer {
  public:
    Timer() : start_time_point(std::chrono::high_resolution_clock::now()) {}

    void reset() {
        start_time_point = std::chrono::high_resolution_clock::now();
    }

    double elapsed_seconds() const {
        auto end_time_point = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration =
            end_time_point - start_time_point;
        return duration.count();
    }

    long int elapsed_nanoseconds() const {
        auto end_time_point = std::chrono::high_resolution_clock::now();
        auto duration = end_time_point - start_time_point;
        auto d = std::chrono::duration_cast<std::chrono::nanoseconds>(duration);
        return d.count();
    }

  private:
    std::chrono::high_resolution_clock::time_point start_time_point;
};

#define FUNC_COST_TIME(func, ...)                                              \
    {                                                                          \
        auto start = std::chrono::high_resolution_clock::now();                \
        auto timer = Timer();                                                  \
        func(__VA_ARGS__);                                                     \
        std::cout << #func << " took " << timer.elapsed_nanoseconds()          \
                  << " seconds." << std::endl;                                 \
    }
template <typename ReturnType, typename... Args>
ReturnType function_cost_time(ReturnType (*func)(Args...), Args &&...args) {
    // 记录开始时间
    auto start = std::chrono::high_resolution_clock::now();

    // 调用原始函数，完美转发参数
    ReturnType result = func(std::forward<Args>(args)...);

    // 记录结束时间
    auto end = std::chrono::high_resolution_clock::now();

    // 计算并输出执行时间
    std::chrono::duration<double> duration = end - start;
    std::cout << "Function execution took: " << duration.count() << " seconds."
              << std::endl;

    return result;
}
