import fiftyone as fo
from fiftyone.types import COCODetectionDataset

# Path to the dataset
dataset_dir = "./coco_data/train_data"
data_path = f"{dataset_dir}/images"
labels_path = f"{dataset_dir}/labels.json"

# Load the dataset
dataset = fo.Dataset.from_dir(
    dataset_type=COCODetectionDataset,
    data_path=data_path,
    labels_path=labels_path,
    dataset_name="coco_dataset"
)

# Launch FiftyOne app
session = fo.launch_app(dataset)
session.wait()

