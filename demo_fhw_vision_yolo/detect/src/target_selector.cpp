#include "target_selector.h"

#include <algorithm>

void TargetSelector::init(const cv::Size& frame_size) {
    frame_size_ = frame_size;
}

FrameResult TargetSelector::update(const std::vector<ArmorObject>& detections) {
    FrameResult result;

    std::vector<ArmorObject> sorted = detections;
    std::sort(sorted.begin(), sorted.end(), [](const ArmorObject& a, const ArmorObject& b) {
        return a.confidence > b.confidence;
    });

    for (const auto& det : sorted) {
        if (result.detected_count >= MAX_ARMOR_COUNT) break;
        if (det.center.x < 0 || det.center.y < 0 ||
            det.center.x >= frame_size_.width || det.center.y >= frame_size_.height) {
            continue;
        }

        result.centers[result.detected_count] = det.center;
        result.detected_count++;
    }

    return result;
}
