#include "detector.h"

#include <algorithm>
#include <cmath>
#include <iostream>

bool ArmorDetector::init(const std::string& model_path,
                         float conf_threshold,
                         float nms_threshold,
                         const cv::Size& input_size) {
    conf_threshold_ = conf_threshold;
    nms_threshold_ = nms_threshold;
    input_size_ = input_size;
    class_names_ = {"blue3", "blue1", "bluesb", "red3", "red1", "redsb"};

    try {
        net_ = cv::dnn::readNetFromONNX(model_path);
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
    } catch (const cv::Exception& e) {
        std::cerr << "[Detector] Failed to load ONNX model: " << model_path << "\n"
                  << e.what() << std::endl;
        return false;
    }

    if (net_.empty()) {
        std::cerr << "[Detector] Empty network: " << model_path << std::endl;
        return false;
    }

    std::cout << "[Detector] Model loaded: " << model_path << std::endl;
    return true;
}

bool ArmorDetector::detect(const cv::Mat& frame, std::vector<ArmorObject>& results) {
    results.clear();
    if (frame.empty() || net_.empty()) return false;

    cv::Mat blob = preprocess(frame);
    net_.setInput(blob);

    std::vector<cv::Mat> outputs;
    net_.forward(outputs, net_.getUnconnectedOutLayersNames());
    if (outputs.empty()) return false;

    postprocess(outputs[0], frame.size(), results);
    return !results.empty();
}

cv::Mat ArmorDetector::preprocess(const cv::Mat& frame) {
    float r_w = static_cast<float>(input_size_.width) / static_cast<float>(frame.cols);
    float r_h = static_cast<float>(input_size_.height) / static_cast<float>(frame.rows);
    scale_ = std::min(r_w, r_h);

    int resized_w = static_cast<int>(std::round(frame.cols * scale_));
    int resized_h = static_cast<int>(std::round(frame.rows * scale_));
    pad_x_ = (input_size_.width - resized_w) / 2;
    pad_y_ = (input_size_.height - resized_h) / 2;

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(resized_w, resized_h));

    cv::Mat canvas(input_size_, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(pad_x_, pad_y_, resized_w, resized_h)));

    cv::Mat blob;
    cv::dnn::blobFromImage(canvas, blob, 1.0 / 255.0, input_size_, cv::Scalar(), true, false);
    return blob;
}

void ArmorDetector::postprocess(const cv::Mat& output,
                                const cv::Size& frame_size,
                                std::vector<ArmorObject>& results) {
    if (output.empty()) return;

    cv::Mat predictions;
    if (output.dims == 3) {
        int dim1 = output.size[1];
        int dim2 = output.size[2];
        predictions = cv::Mat(dim1, dim2, CV_32F,
                              const_cast<float*>(reinterpret_cast<const float*>(output.data)));
        if (dim1 < dim2) {
            cv::transpose(predictions, predictions);
        }
    } else if (output.dims == 2) {
        predictions = output;
    } else {
        std::cerr << "[Detector] Unsupported output dims: " << output.dims << std::endl;
        return;
    }

    const int num_classes = static_cast<int>(class_names_.size());
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    for (int i = 0; i < predictions.rows; ++i) {
        const float* row = predictions.ptr<float>(i);
        const int attrs = predictions.cols;
        if (attrs < 5) continue;

        bool has_objectness = (attrs == 5 + num_classes);
        int class_offset = has_objectness ? 5 : 4;
        int available_classes = attrs - class_offset;
        if (available_classes <= 0) continue;

        int best_class = 0;
        float best_score = row[class_offset];
        for (int c = 1; c < available_classes; ++c) {
            if (row[class_offset + c] > best_score) {
                best_score = row[class_offset + c];
                best_class = c;
            }
        }

        float confidence = has_objectness ? row[4] * best_score : best_score;
        if (confidence < conf_threshold_) continue;

        float cx = row[0];
        float cy = row[1];
        float w = row[2];
        float h = row[3];
        if (std::max({std::abs(cx), std::abs(cy), std::abs(w), std::abs(h)}) <= 2.0f) {
            cx *= input_size_.width;
            w *= input_size_.width;
            cy *= input_size_.height;
            h *= input_size_.height;
        }

        float x1 = (cx - w * 0.5f - static_cast<float>(pad_x_)) / scale_;
        float y1 = (cy - h * 0.5f - static_cast<float>(pad_y_)) / scale_;
        float x2 = (cx + w * 0.5f - static_cast<float>(pad_x_)) / scale_;
        float y2 = (cy + h * 0.5f - static_cast<float>(pad_y_)) / scale_;

        x1 = std::clamp(x1, 0.0f, static_cast<float>(frame_size.width - 1));
        y1 = std::clamp(y1, 0.0f, static_cast<float>(frame_size.height - 1));
        x2 = std::clamp(x2, 0.0f, static_cast<float>(frame_size.width - 1));
        y2 = std::clamp(y2, 0.0f, static_cast<float>(frame_size.height - 1));
        if (x2 <= x1 || y2 <= y1) continue;

        boxes.emplace_back(cv::Rect(cv::Point(static_cast<int>(std::round(x1)), static_cast<int>(std::round(y1))),
                                    cv::Point(static_cast<int>(std::round(x2)), static_cast<int>(std::round(y2)))));
        confidences.push_back(confidence);
        class_ids.push_back(best_class);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold_, nms_threshold_, indices);

    results.reserve(indices.size());
    for (int idx : indices) {
        ArmorObject obj;
        obj.bbox = cv::Rect2f(boxes[idx]);
        obj.class_id = class_ids[idx];
        obj.confidence = confidences[idx];
        obj.center = cv::Point2f(obj.bbox.x + obj.bbox.width * 0.5f,
                                 obj.bbox.y + obj.bbox.height * 0.5f);
        results.push_back(obj);
    }

    std::sort(results.begin(), results.end(), [](const ArmorObject& a, const ArmorObject& b) {
        return a.confidence > b.confidence;
    });
}
