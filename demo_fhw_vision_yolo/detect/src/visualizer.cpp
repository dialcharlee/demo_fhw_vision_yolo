#include "visualizer.h"

#include <cmath>
#include <iomanip>
#include <sstream>

namespace {
cv::Scalar colorForClass(int class_id) {
    static const cv::Scalar colors[] = {
        cv::Scalar(255, 80, 80),   // blue3
        cv::Scalar(255, 180, 60),  // blue1
        cv::Scalar(255, 80, 220),  // bluesb
        cv::Scalar(80, 80, 255),   // red1
        cv::Scalar(60, 180, 255),  // red3
        cv::Scalar(80, 220, 255)   // redsb
    };
    return colors[std::abs(class_id) % 6];
}

std::string className(int class_id) {
    static const std::string names[] = {
        "blue3", "blue1", "bluesb",
         "red1", "red3","redsb"
    };
    return names[std::abs(class_id) % 6];
}
} 

void Visualizer::drawDetections(cv::Mat& frame,
                                const std::vector<ArmorObject>& detections) {
    for (const auto& det : detections) {
        cv::Scalar color = colorForClass(det.class_id);
        cv::Rect box(cv::Point(static_cast<int>(std::round(det.bbox.x)),
                               static_cast<int>(std::round(det.bbox.y))),
                     cv::Point(static_cast<int>(std::round(det.bbox.x + det.bbox.width)),
                               static_cast<int>(std::round(det.bbox.y + det.bbox.height))));
        cv::rectangle(frame, box, color, 2);

        std::ostringstream label;
        label << className(det.class_id) << " " 
              << std::fixed << std::setprecision(2) << det.confidence;

        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::Point origin(static_cast<int>(det.bbox.x), std::max(0, static_cast<int>(det.bbox.y) - 6));
        cv::Rect bg(origin.x, std::max(0, origin.y - text_size.height - baseline),
                    text_size.width + 6, text_size.height + baseline + 4);
        cv::rectangle(frame, bg, color, cv::FILLED);
        cv::putText(frame, label.str(), cv::Point(bg.x + 3, bg.y + text_size.height + 1),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(20, 20, 20), 1, cv::LINE_AA);
    }
}

void Visualizer::drawCenters(cv::Mat& frame, const FrameResult& result) {
    for (int i = 0; i < result.detected_count && i < MAX_ARMOR_COUNT; ++i) {
        const cv::Point2f& p = result.centers[i];
        cv::circle(frame, p, 5, cv::Scalar(0, 255, 255), cv::FILLED);
        
        std::ostringstream text;
        text << "(" << static_cast<int>(std::round(p.x)) << "," << static_cast<int>(std::round(p.y)) << ")";
        cv::putText(frame, text.str(), p + cv::Point2f(8, -8),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
    }
}

void Visualizer::drawHUD(cv::Mat& frame, float fps, int detected_count) {
    std::ostringstream line1;
    line1 << "FPS: " << std::fixed << std::setprecision(1) << fps;
    std::ostringstream line2;
    line2 << "Armors: " << detected_count;

    cv::rectangle(frame, cv::Rect(8, 8, 170, 58), cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(frame, line1.str(), cv::Point(16, 30), cv::FONT_HERSHEY_SIMPLEX,
                0.6, cv::Scalar(80, 255, 80), 2, cv::LINE_AA);
    cv::putText(frame, line2.str(), cv::Point(16, 56), cv::FONT_HERSHEY_SIMPLEX,
                0.6, detected_count > 0 ? cv::Scalar(0, 255, 255) : cv::Scalar(160, 160, 160),
                2, cv::LINE_AA);
}