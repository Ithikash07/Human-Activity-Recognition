# Human-Activity-Recognition

##Project Overview:
Developed a comprehensive multi-person human activity recognition (HAR) system capable of detecting and classifying human actions in complex, crowded videos. The system fuses state-of-the-art detection, tracking, and video classification methods for robust performance in real-world scenarios.

##Core Components and Technologies:

Action Recognition Backbone:
Initially employed TensorFlow’s MoviNet, a cutting-edge, efficient video classification model tailored for spatiotemporal feature learning, achieving high accuracy on the UCF101 dataset. Further exploration included fine-tuning or complementary architectures based on EfficientNet with temporal convolutions or LSTM layers for enhanced temporal modeling.

Datasets and Training:
Utilized the UCF101 dataset, a widely-used benchmark of over 10,000 labeled videos spanning 101 action categories. Implemented data preprocessing pipelines including frame extraction, resizing, augmentation, and dataset balancing to prepare input batches.

Person Detection:
Integrated YOLOv8, a leading object detector optimized for high recall and precision, especially effective in crowded and dynamic scenes, ensuring reliable person localization.

Multi-object Tracking:
Applied ByteTrack multi-object tracker to maintain consistent identities across frames, enabling temporal tracking of multiple people for individualized action analysis.

Per-Person Clip Extraction and Classification:
Constructed temporal clips from tracked bounding boxes by cropping and resizing, facilitating action classification on per-individual motion segments using the trained deep learning pipeline.

Visualization and Interface:
Designed a real-time or offline video processing pipeline visualized through a Streamlit app, displaying detected persons, tracked IDs, and predicted actions with confidence metrics overlayed on video frames.

##Technologies Involved:

TensorFlow (MoviNet, EfficientNet), YOLOv8 (Ultralytics), ByteTrack, OpenCV, Streamlit, UCF101 Dataset.
