#pragma once
// LOCAL DEBUG: LEONIDA_DEBUG_TIME_SHADER=<part of a compute program name> times, on the GPU, every dispatch of the
// compute pipelines whose program name contains it: a timestamp after the work before it and one after the dispatch
// (both BOTTOM_OF_PIPE, so the gap is the dispatch alone on its queue; other queues still run beside it). The results
// are read later without waiting, only once both are available, and the count, mean and longest are logged every 10 s
// and in total. A pair whose command buffer never ran stays pending and is lost (1024 pairs).
#include <vulkan/vulkan.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace leonida::debug_shader_timer {

inline const char* wanted() {
    static const char* name = [] {
        const char* v = std::getenv("LEONIDA_DEBUG_TIME_SHADER");
        return v && *v ? v : static_cast<const char*>(nullptr);
    }();
    return name;
}

inline bool matches(const std::string& program) {
    const char* name = wanted();
    return name && program.find(name) != std::string::npos;
}

struct Timer {
    static constexpr std::uint32_t pairs = 1024;
    VkDevice device = VK_NULL_HANDLE;
    VkQueryPool pool = VK_NULL_HANDLE;
    PFN_vkCmdResetQueryPool reset = nullptr;
    PFN_vkCmdWriteTimestamp write = nullptr;
    PFN_vkGetQueryPoolResults get = nullptr;
    double period_ns = 1.0;
    std::mutex mutex;
    std::vector<std::uint8_t> pending = std::vector<std::uint8_t>(pairs, 0);
    std::uint32_t next = 0;
    std::uint64_t count = 0, sum_ns = 0, max_ns = 0, total_count = 0, total_sum_ns = 0, total_max_ns = 0, lost = 0;
    std::chrono::steady_clock::time_point last_log = std::chrono::steady_clock::now();

    // The pair index to pass to end(), or -1 when every pair is still waiting for its results.
    int begin(VkCommandBuffer command_buffer) {
        std::lock_guard lock(mutex);
        for (std::uint32_t tried = 0; tried < pairs; ++tried) {
            const std::uint32_t slot = (next + tried) % pairs;
            if (pending[slot]) continue;
            pending[slot] = 1;
            next = (slot + 1) % pairs;
            reset(command_buffer, pool, slot * 2, 2);
            write(command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, slot * 2);
            return int(slot);
        }
        ++lost;
        return -1;
    }
    void end(VkCommandBuffer command_buffer, int slot) {
        if (slot >= 0) write(command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, std::uint32_t(slot) * 2 + 1);
    }

    void poll() {
        std::lock_guard lock(mutex);
        for (std::uint32_t slot = 0; slot < pairs; ++slot) {
            if (!pending[slot]) continue;
            std::uint64_t data[4] = {};  // value, availability, value, availability
            const VkResult r = get(device, pool, slot * 2, 2, sizeof(data), data, sizeof(std::uint64_t) * 2,
                                   VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
            if ((r != VK_SUCCESS && r != VK_NOT_READY) || !data[1] || !data[3]) continue;
            pending[slot] = 0;
            if (data[2] < data[0]) continue;
            const std::uint64_t ns = std::uint64_t(double(data[2] - data[0]) * period_ns);
            if (ns > 600ull * 1000000000ull) continue;  // not a measurement
            ++count;
            sum_ns += ns;
            max_ns = std::max(max_ns, ns);
        }
        const auto now = std::chrono::steady_clock::now();
        if (now - last_log < std::chrono::seconds(10) || !count) return;
        total_count += count;
        total_sum_ns += sum_ns;
        total_max_ns = std::max(total_max_ns, max_ns);
        std::fprintf(stderr,
                     "[rt] shader-time %s: %llu dispatches in 10 s, mean %.3f ms, max %.3f ms; total %llu, mean %.3f ms, "
                     "max %.3f ms, %llu not timed\n",
                     wanted(), static_cast<unsigned long long>(count), sum_ns / 1e6 / double(count), max_ns / 1e6,
                     static_cast<unsigned long long>(total_count), total_sum_ns / 1e6 / double(total_count),
                     total_max_ns / 1e6, static_cast<unsigned long long>(lost));
        count = sum_ns = max_ns = 0;
        last_log = now;
    }
};

inline std::atomic<Timer*>& instance() {
    static std::atomic<Timer*> timer{nullptr};
    return timer;
}

// Creates the timer once (from the submitting thread, before its first poll). Returns null when not asked for or
// when the device cannot time.
inline Timer* create(VkDevice device, PFN_vkGetDeviceProcAddr get_device, PFN_vkCreateQueryPool create_pool,
                     PFN_vkCmdResetQueryPool reset, PFN_vkCmdWriteTimestamp write, double period_ns) {
    if (!wanted() || instance().load()) return instance().load();
    static std::once_flag once;
    std::call_once(once, [&] {
        auto get = get_device ? reinterpret_cast<PFN_vkGetQueryPoolResults>(get_device(device, "vkGetQueryPoolResults"))
                              : nullptr;
        if (!get || !create_pool || !reset || !write || period_ns <= 0) {
            std::fprintf(stderr, "[rt] shader-time: cannot time on this device\n");
            return;
        }
        VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        info.queryType = VK_QUERY_TYPE_TIMESTAMP;
        info.queryCount = Timer::pairs * 2;
        auto* timer = new Timer;
        if (create_pool(device, &info, nullptr, &timer->pool) != VK_SUCCESS) {
            delete timer;
            std::fprintf(stderr, "[rt] shader-time: no query pool\n");
            return;
        }
        timer->device = device;
        timer->reset = reset;
        timer->write = write;
        timer->get = get;
        timer->period_ns = period_ns;
        instance().store(timer);  // kept for the process's life
        std::fprintf(stderr, "[rt] shader-time: timing dispatches of programs named *%s* (%.3f ns per tick)\n",
                     wanted(), period_ns);
    });
    return instance().load();
}

}  // namespace leonida::debug_shader_timer
