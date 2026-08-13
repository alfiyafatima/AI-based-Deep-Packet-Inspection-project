#include "flow_features.h"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <cctype>

namespace DPI {

std::string categoryToString(TrafficCategory category) {
    switch (category) {
        case TrafficCategory::WEB_BROWSING:     return "WEB_BROWSING";
        case TrafficCategory::VIDEO_STREAMING:  return "VIDEO_STREAMING";
        case TrafficCategory::VOIP_CALL:        return "VOIP_CALL";
        case TrafficCategory::FILE_TRANSFER:    return "FILE_TRANSFER";
        case TrafficCategory::UNKNOWN:
        default:                                return "UNKNOWN";
    }
}

TrafficCategory stringToCategory(const std::string& str) {
    std::string s = str;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::toupper(c); });
    
    if (s == "WEB_BROWSING" || s == "WEB" || s == "BROWSING") return TrafficCategory::WEB_BROWSING;
    if (s == "VIDEO_STREAMING" || s == "VIDEO" || s == "STREAMING") return TrafficCategory::VIDEO_STREAMING;
    if (s == "VOIP_CALL" || s == "VOIP" || s == "CALL") return TrafficCategory::VOIP_CALL;
    if (s == "FILE_TRANSFER" || s == "FILE" || s == "TRANSFER") return TrafficCategory::FILE_TRANSFER;
    return TrafficCategory::UNKNOWN;
}

std::vector<float> FlowFeatures::toVector() const {
    return {
        mean_packet_size,
        stddev_packet_size,
        mean_iat_ms,
        stddev_iat_ms,
        bytes_ratio,
        pkt_len_1,
        pkt_len_2,
        pkt_len_3,
        pkt_len_4,
        pkt_len_5
    };
}

FlowFeatures FlowFeatures::extract(
    const std::vector<size_t>& packet_sizes,
    const std::vector<std::chrono::steady_clock::time_point>& packet_times,
    uint64_t bytes_in,
    uint64_t bytes_out) {

    FlowFeatures f;

    // 1. Packet size mean and stddev
    if (!packet_sizes.empty()) {
        double sum = 0.0;
        for (size_t sz : packet_sizes) {
            sum += static_cast<double>(sz);
        }
        f.mean_packet_size = static_cast<float>(sum / packet_sizes.size());

        double var_sum = 0.0;
        for (size_t sz : packet_sizes) {
            double diff = static_cast<double>(sz) - f.mean_packet_size;
            var_sum += diff * diff;
        }
        f.stddev_packet_size = static_cast<float>(std::sqrt(var_sum / packet_sizes.size()));
    }

    // 2. Inter-arrival time (IAT) mean and stddev
    if (packet_times.size() >= 2) {
        std::vector<double> iats_ms;
        iats_ms.reserve(packet_times.size() - 1);

        for (size_t i = 1; i < packet_times.size(); ++i) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                packet_times[i] - packet_times[i - 1]).count();
            iats_ms.push_back(static_cast<double>(duration) / 1000.0); // convert microseconds to milliseconds
        }

        double sum_iat = std::accumulate(iats_ms.begin(), iats_ms.end(), 0.0);
        f.mean_iat_ms = static_cast<float>(sum_iat / iats_ms.size());

        double var_iat = 0.0;
        for (double iat : iats_ms) {
            double diff = iat - f.mean_iat_ms;
            var_iat += diff * diff;
        }
        f.stddev_iat_ms = static_cast<float>(std::sqrt(var_iat / iats_ms.size()));
    }

    // 3. Bytes ratio (Inbound / Outbound)
    if (bytes_out > 0) {
        f.bytes_ratio = static_cast<float>(bytes_in) / static_cast<float>(bytes_out);
    } else if (bytes_in > 0) {
        f.bytes_ratio = static_cast<float>(bytes_in);
    } else {
        f.bytes_ratio = 1.0f;
    }

    // 4. First 5 packet payload / frame lengths
    f.pkt_len_1 = (packet_sizes.size() > 0) ? static_cast<float>(packet_sizes[0]) : 0.0f;
    f.pkt_len_2 = (packet_sizes.size() > 1) ? static_cast<float>(packet_sizes[1]) : 0.0f;
    f.pkt_len_3 = (packet_sizes.size() > 2) ? static_cast<float>(packet_sizes[2]) : 0.0f;
    f.pkt_len_4 = (packet_sizes.size() > 3) ? static_cast<float>(packet_sizes[3]) : 0.0f;
    f.pkt_len_5 = (packet_sizes.size() > 4) ? static_cast<float>(packet_sizes[4]) : 0.0f;

    return f;
}

} // namespace DPI
