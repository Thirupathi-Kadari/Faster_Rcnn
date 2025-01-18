# Multiclass Food Detection with Faster R-CNN and Detectron2 🍔🍱
**"Accurate detection and classification of food items for advanced AI applications"**

<div>
  <img src="/images_app/food-detection-banner.png" alt="Food Detection" />
</div>

> A Sync AI Inc. Research Initiative for Food Recognition

---

[![GitHub Stars](https://img.shields.io/github/stars/your-username/faster-rcnn-food-detection?style=social)](https://github.com/your-username/faster-rcnn-food-detection/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/your-username/faster-rcnn-food-detection?style=social)](https://github.com/your-username/faster-rcnn-food-detection/network/members)
[![GitHub Issues](https://img.shields.io/github/issues/your-username/faster-rcnn-food-detection)](https://github.com/your-username/faster-rcnn-food-detection/issues)
[![GitHub Last Commit](https://img.shields.io/github/last-commit/your-username/faster-rcnn-food-detection)](https://github.com/your-username/faster-rcnn-food-detection)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## 🔬 Key Features
- ✅ **Accurate Detection**: Detect and classify 345 unique food items with high precision.
- ✅ **Custom Dataset**: Uses a COCO-format dataset for detailed food annotations.
- ✅ **Dockerized Setup**: Ensures consistent environments for training and testing.
- ✅ **Performance Metrics**: Evaluated with mAP@50 and mAP@50-95 for model accuracy.
- ✅ **Future LLM Integration**: Planned support for multimodal insights using language models.

---

## 👥 Research Team at Sync AI Inc.
- **Thirupathi Kadari** - Research Lead, AI Development
- **Syed Raheel Hussain** - Research Contributor, Healthcare Integration
- **Tushar Sinha** - Technical Advisor, Product Strateg

---

## 🛠 Technical Architecture

### Model Details
- **Model Used**: Faster R-CNN with ResNet-50 backbone and FPN.
- **Framework**: [Detectron2](https://github.com/facebookresearch/detectron2) powered by [PyTorch](https://pytorch.org/).
- **Dataset**: Custom COCO-format dataset with 345 classes.
- **Deployment**: Dockerized environment for robust and reproducible experiments.

### Application Features
1. Multiclass food detection and classification.
2. Integration with COCO evaluation metrics for precise validation.
3. Training logs and checkpoints for iterative improvement.
4. Dockerized environment for seamless development and deployment.

---
## 🔧 Installation

### Prerequisites
- **Hardware**: NVIDIA GPU with at least 8GB VRAM.
- **Software**:
  - Docker
  - NVIDIA Container Toolkit

### Steps to Install and Run
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/faster-rcnn-food-detection.git
   cd faster-rcnn-food-detection
---

### Build the Docker Image
  ```

docker build -t detectron2:latest -f docker/Dockerfile .

```

### Run the Docker Container
  ```
  docker run --gpus all -it --rm --name detectron2-container -v $(pwd):/workspace detectron2:latest bash
```

---

## 📊 Performance Metrics

| Model          | Dataset       | Epochs | **mAP@0.5** | **mAP@0.5:0.95** |
|-----------------|---------------|--------|-------------|------------------|
| Faster R-CNN   | Food Dataset  | 20     | 🟢 0.51    | 🟠 0.38         |

---

## 🍽 Dataset
The training dataset consists of food images annotated in COCO format:
- **Training Annotations**: `coco_annotations_train.json`
- **Validation Annotations**: `coco_annotations_val.json`
- **Images**: Located in `/workspace/images`.

For details on dataset preparation, refer to the [Dataset Preparation Guide](docs/dataset-preparation.md).

---

## 🩺 Applications
- **Dietary Tracking**: Automate meal logging and classify food types.
- **Health Monitoring**: Supports future integration with nutritional analysis tools.
- **AI Training**: Offers a benchmark dataset for food detection experiments.
- **Mobile Apps**: Can be extended to smartphone-based applications for real-time food detection.

---

## Application Screenshots
<div>
  <img src="/images_app/appscreen1.png" alt="Sample Application Screenshot 1" />
</div>
<div>
  <img src="/images_app/appscreen2.png" alt="Sample Application Screenshot 2" />
</div>

---

## 🔗 Future Integration
1. **Advanced Models**: Extend to support larger datasets and higher accuracy.
2. **LLM Integration**: Leverage language models for generating nutritional insights.
3. **Mobile Optimization**: Build lightweight mobile models for real-time use.
4. **Integration with Healthcare Platforms**: Enable seamless connectivity with health monitoring systems.

---

## 🤝 Contributing  
We welcome contributions! Here's how you can get involved:  
1. Fork the repository.  
2. Create a new branch: `git checkout -b feature/your-feature`.  
3. Commit your changes: `git commit -m 'Add some feature'`.  
4. Push to the branch: `git push origin feature/your-feature`.  
5. Submit a pull request. 

## 📫 Contact
For research collaboration or healthcare integration inquiries:
- 📧 **Thirupathi Kadari**: [Email](mailto:thirupathi.kadari986@gmail.com)
- 📧 **Sayed Raheel Hussain**: [Email](mailto:Sayedraheel1995@gmail.com)
- 📧 **Tushar Sinha**: [Email](mailto:tsr@justsync.ai)

## 📃 License
Copyright © 2024 Sync AI Inc. All rights reserved.

---
<p align="center">
Developed by Sync AI Inc. for advancing diabetes care through intelligent nutrition monitoring
</p>
