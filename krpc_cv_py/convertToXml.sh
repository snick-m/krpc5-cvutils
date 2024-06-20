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

cd coco_data/xml_data

cd train_data/data && mogrify -format jpg *.png && rm -rf *.png && cd ../../
cd val_data/data && mogrify -format jpg *.png && rm -rf *.png && cd ../../
cd test_data/data && mogrify -format jpg *.png && rm -rf *.png && cd ../../

sed -i 's/.png/.jpg/' ./**/**/*.xml
sed -i 's/<\/object>/    <difficult>Unspecified<\/difficult>\
        <truncated>Unspecified<\/truncated>\
        <pose>Unspecified<\/pose>\
    <\/object>/' ./**/**/*.xml
