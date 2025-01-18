import os
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import logging

# Setup logger
setup_logger()

# Define paths
train_json = "/workspace/json_files/coco_annotations_train.json"
val_json = "/workspace/json_files/coco_annotations_val.json"
images_path = "/workspace/images"

# Register datasets
register_coco_instances("food_train", {}, train_json, os.path.join(images_path, "train"))
register_coco_instances("food_val", {}, val_json, os.path.join(images_path, "val"))

# Verify dataset registration
train_metadata = MetadataCatalog.get("food_train")
val_metadata = MetadataCatalog.get("food_val")
print("Train Metadata:", train_metadata)
print("Val Metadata:", val_metadata)

# Custom Dataset Mapper to Handle Size Mismatches
from detectron2.data import detection_utils as utils
from detectron2.structures import BoxMode

class CustomDatasetMapper(DatasetMapper):
    def __call__(self, dataset_dict):
        try:
            # Call the default behavior
            return super().__call__(dataset_dict)
        except utils.SizeMismatchError as e:
            # Log and skip the problematic image
            logging.warning(f"Skipping problematic image: {dataset_dict['file_name']} due to size mismatch.")
            return None

# Custom Trainer to Use the Custom Mapper
class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(
            cfg, mapper=CustomDatasetMapper(cfg, is_train=True)
        )

# Configure model and training
cfg = get_cfg()
cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Base config
cfg.DATASETS.TRAIN = ("food_train",)
cfg.DATASETS.TEST = ("food_val",)
cfg.DATALOADER.NUM_WORKERS = 4

# Pre-trained weights
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"

# Updated Training Parameters
cfg.SOLVER.IMS_PER_BATCH = 8  # Batch size per iteration
cfg.SOLVER.BASE_LR = 0.001  # Learning rate
cfg.SOLVER.MAX_ITER = 100000  # Total iterations for approximately 1 epoch (2.2x dataset coverage for batch size = 8)
cfg.SOLVER.STEPS = (70000, 90000)  # Learning rate decay steps
cfg.SOLVER.GAMMA = 0.1  # Reduce LR by a factor of 10
cfg.SOLVER.CHECKPOINT_PERIOD = 5000  # Save checkpoints every 5000 iterations
cfg.TEST.EVAL_PERIOD = 5000  # Evaluate the model every 5000 iterations

# ROI Head Configuration
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # Number of proposals per image
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 345  # Update for your dataset's number of classes

# Output directory
cfg.OUTPUT_DIR = "./output"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Train the model
trainer = CustomTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Evaluate the model
evaluator = COCOEvaluator("food_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "food_val")
print("Evaluating validation dataset...")
evaluation_results = inference_on_dataset(trainer.model, val_loader, evaluator)

# Print mAP score
print("Evaluation Results (mAP and other metrics):")
print(evaluation_results)
