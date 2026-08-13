#ifndef DPI_ML_CLASSIFIER_H
#define DPI_ML_CLASSIFIER_H

#include "flow_features.h"
#include <string>
#include <memory>

namespace DPI {

class MLClassifier {
public:
    explicit MLClassifier(const std::string& model_path = "traffic_model.onnx");
    ~MLClassifier();

    // Predict traffic category from extracted flow features
    TrafficCategory predict(const FlowFeatures& features) const;

    // Check if ONNX model was loaded successfully
    bool isModelLoaded() const { return model_loaded_; }

private:
    std::string model_path_;
    bool model_loaded_ = false;

    // Lightweight Decision Tree / Random Forest evaluator fallback
    TrafficCategory predictFallback(const FlowFeatures& features) const;
};

} // namespace DPI

#endif // DPI_ML_CLASSIFIER_H
