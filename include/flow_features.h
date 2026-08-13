#ifndef DPI_FLOW_FEATURES_H
#define DPI_FLOW_FEATURES_H

#include <vector>
#include <string>
#include <cstddef>
#include <chrono>

namespace DPI {

// ============================================================================
// Traffic Category Classification Enum
// ============================================================================
enum class TrafficCategory {
    UNKNOWN = 0,
    WEB_BROWSING,
    VIDEO_STREAMING,
    VOIP_CALL,
    FILE_TRANSFER,
    APP_COUNT
};

std::string categoryToString(TrafficCategory category);
TrafficCategory stringToCategory(const std::string& str);

// ============================================================================
// Flow Features Struct for ML Inference
// ============================================================================
struct FlowFeatures {
    float mean_packet_size = 0.0f;
    float stddev_packet_size = 0.0f;
    float mean_iat_ms = 0.0f;        // Inter-packet arrival time in ms
    float stddev_iat_ms = 0.0f;
    float bytes_ratio = 1.0f;        // Inbound / Outbound bytes ratio
    
    // Payload lengths of the first 5 packets
    float pkt_len_1 = 0.0f;
    float pkt_len_2 = 0.0f;
    float pkt_len_3 = 0.0f;
    float pkt_len_4 = 0.0f;
    float pkt_len_5 = 0.0f;

    // Convert features into a float array for ML model input
    std::vector<float> toVector() const;

    // Helper to compute FlowFeatures from raw packet sizes and timestamps
    static FlowFeatures extract(
        const std::vector<size_t>& packet_sizes,
        const std::vector<std::chrono::steady_clock::time_point>& packet_times,
        uint64_t bytes_in,
        uint64_t bytes_out
    );
};

} // namespace DPI

#endif // DPI_FLOW_FEATURES_H
