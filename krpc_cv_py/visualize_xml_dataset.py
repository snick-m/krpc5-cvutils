import fiftyone as fo
from fiftyone.types import VOCDetectionDataset

# Path to the dataset
dataset_dir = "./coco_data/xml_data/train_data"
data_path = f"{dataset_dir}/data"
labels_path = f"{dataset_dir}/labels"

# Load the dataset
dataset = fo.Dataset.from_dir(
    dataset_type=VOCDetectionDataset,
    dataset_dir=dataset_dir,
    dataset_name="xml_dataset"
)

# Launch FiftyOne app
session = fo.launch_app(dataset)
session.wait()

