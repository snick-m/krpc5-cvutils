#!/bin/bash
fiftyone convert --input-dir ./coco_data/train_data \
  --input-type fiftyone.types.COCODetectionDataset \
  --input-kwargs data_path=images \
  --output-dir ./coco_data/xml_data/train_data \
  --output-type fiftyone.types.VOCDetectionDataset

fiftyone convert --input-dir ./coco_data/test_data \
  --input-type fiftyone.types.COCODetectionDataset \
  --input-kwargs data_path=images \
  --output-dir ./coco_data/xml_data/test_data \
  --output-type fiftyone.types.VOCDetectionDataset

fiftyone convert --input-dir ./coco_data/val_data \
  --input-type fiftyone.types.COCODetectionDataset \
  --input-kwargs data_path=images \
  --output-dir ./coco_data/xml_data/val_data \
  --output-type fiftyone.types.VOCDetectionDataset
