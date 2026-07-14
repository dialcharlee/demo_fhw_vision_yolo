#include "detector.h"
#include "target_selector.h"
#include "visualizer.h"

#ifdef ENABLE_JUDGE
#include "judge.h"
#endif

#ifdef ENABLE_EVALUATE
#include "evaluate.h"
#endif

#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

int main() {
    std::string model_path = "assets/best.onnx"; 
    std::string input_path = "assets/RM_TestVideo.mp4"; 
    float conf_thresh = 0.5f;
    float nms_thresh = 0.45f;
    int input_size = 640;

#ifdef ENABLE_JUDGE
    Judge judge;
    judge.init("assets/log");
#endif

#ifdef ENABLE_EVALUATE
    Evaluator evaluator;
    evaluator.init("assets/std.csv");
#endif

    ArmorDetector detector;
    TargetSelector selector;
    Visualizer vis;

    //加载模型
    detector.init(model_path, conf_thresh, nms_thresh, cv::Size(input_size, input_size));

    // ===== 打开输入源 =====
    cv::VideoCapture cap;
    if (input_path == "0" ) {
        cap.open(0);   // 默认摄像头
    } else {
        cap.open(input_path);
    }

    // ===== 获取第一帧尺寸并初始化 selector =====
    cv::Mat frame;
    cap.read(frame);
    selector.init(frame.size());

    // ===== 主循环 =====
    int total_frames = 0;
    int detected_frames = 0;
    float total_time_ms = 0.0f;

    while (true) {
        if (!cap.read(frame) || frame.empty()) break;
        total_frames++;

        auto t0 = std::chrono::high_resolution_clock::now();

        std::vector<ArmorObject> detections;
        detector.detect(frame, detections);
        FrameResult result = selector.update(detections);

        auto t1 = std::chrono::high_resolution_clock::now();
        float frame_time_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        total_time_ms += frame_time_ms;

        if (result.detected_count > 0) detected_frames++;

#ifdef ENABLE_JUDGE
        judge.log(total_frames, result);
#endif

#ifdef ENABLE_EVALUATE
        evaluator.submit(total_frames, result);
#endif

        float instant_fps = (frame_time_ms > 0.0f) ? (1000.0f / frame_time_ms) : 0.0f;
        vis.drawDetections(frame, detections);
        vis.drawCenters(frame, result);
        vis.drawHUD(frame, instant_fps, result.detected_count);

        cv::imshow("armor_detect", frame);
        if (cv::waitKey(1) == 'q') break;   // 按 q 退出
    }

    cap.release();
    cv::destroyAllWindows();

#ifdef ENABLE_JUDGE
    judge.close();
#endif

    // ===== 统计输出 =====
    float detection_rate = (total_frames > 0) ? static_cast<float>(detected_frames) / total_frames : 0.0f;
    float avg_fps = (total_time_ms > 0) ? (1000.0f * total_frames / total_time_ms) : 0.0f;

    std::cout << "\n====== Result ======" << std::endl;
    std::cout << "Total frames:    " << total_frames << std::endl;
    std::cout << "Detected frames: " << detected_frames << std::endl;
    std::cout << "Detection rate:  " << std::fixed << std::setprecision(4) << detection_rate << std::endl;
    std::cout << "Average FPS:     " << std::fixed << std::setprecision(2) << avg_fps << std::endl;
    std::cout << "====================" << std::endl;

#ifdef ENABLE_EVALUATE
    evaluator.printResult(avg_fps);
#endif

    return 0;
}