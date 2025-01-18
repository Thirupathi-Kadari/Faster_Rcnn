import os
import logging
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger
from detectron2.data import DatasetMapper, detection_utils as utils

# Setup logger
setup_logger()

# Define paths
train_json = "/workspace/json_files/coco_annotations_train.json"
val_json = "/workspace/json_files/coco_annotations_val.json"
images_path = "/workspace/images"

# Register datasets
register_coco_instances("food_train", {}, train_json, os.path.join(images_path, "train"))
register_coco_instances("food_val", {}, val_json, os.path.join(images_path, "val"))

# Custom Dataset Mapper to Handle Size Mismatches
class CustomDatasetMapper(DatasetMapper):
    def __call__(self, dataset_dict):
        try:
            # Call the default behavior
            return super().__call__(dataset_dict)
        except utils.SizeMismatchError as e:
            # Log and skip the problematic image
            logging.warning(f"Skipping problematic image: {dataset_dict['file_name']} due to size mismatch.")
            return None

# Configure the model
cfg = get_cfg()
cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Base config
cfg.DATASETS.TRAIN = ("food_train",)
cfg.DATASETS.TEST = ("food_val",)
cfg.DATALOADER.NUM_WORKERS = 4

# Pre-trained weights
cfg.MODEL.WEIGHTS = "./output/model_final.pth"  # Use your trained model weights path
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 345  # Update with your dataset's number of classes
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set a threshold for predictions

# Output directory
cfg.OUTPUT_DIR = "./output"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Evaluate the model
print("Starting evaluation on validation dataset...")

# Create COCO evaluator
evaluator = COCOEvaluator("food_val", cfg, False, output_dir=cfg.OUTPUT_DIR)

# Build the validation data loader using the custom mapper
val_loader = build_detection_test_loader(
    cfg, "food_val", mapper=CustomDatasetMapper(cfg, is_train=False)
)

# Load the trained model
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=True)

# Run the evaluation
evaluation_results = inference_on_dataset(trainer.model, val_loader, evaluator)

# Print evaluation results
print("Evaluation Results (mAP and other metrics):")
print(evaluation_results)

# Save evaluation results to a JSON file
import json
with open(os.path.join(cfg.OUTPUT_DIR, "evaluation_results.json"), "w") as f:
    json.dump(evaluation_results, f, indent=4)

print(f"Evaluation results saved to {os.path.join(cfg.OUTPUT_DIR, 'evaluation_results.json')}")
