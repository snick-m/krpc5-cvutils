"""
This script is used to convert the data from the krpc_cv_py/krpc_cv_py/data_outputs/* folders to the COCO format. 90% goes to training, 10% goes to validation.

COCO Directory structure:
<output_dir>/
  train_data/
    images/
        <img0>.<png>
        <img1>.<png>
        ...
    labels.json
  val_data/
    images/
        <img0>.<png>
        <img1>.<png>
        ...
    labels.json

labels.json:
{
  "categories":[
    {"id":1, "name":<cat1_name>},
    ...
  ],
  "images":[
    {"id":0, "file_name":"<img0>.<jpg/jpeg>"},
    ...
  ],
  "annotations":[
    {"id":0, "image_id":0, "category_id":1, "bbox":[x-top left, y-top left, width, height]},
    ...
  ]
}

"""

import os
import json

input_path = "data_outputs"
output_path = "coco_data"
train_percent = 0.8
val_percent = 0.1


def main():
    generated_items = os.listdir(input_path)

    train_labels = {"categories": [], "images": [], "annotations": []}
    val_labels = {"categories": [], "images": [], "annotations": []}
    test_labels = {"categories": [], "images": [], "annotations": []}

    image_idx = 1
    annotation_idx = 1

    for item_idx in range(len(generated_items)):
        item = generated_items[item_idx]

        train_labels["categories"].append({"id": item_idx + 1, "name": item})
        val_labels["categories"].append({"id": item_idx + 1, "name": item})
        test_labels["categories"].append({"id": item_idx + 1, "name": item})

        for level_idx in range(1, 3):
            level = f"level_{level_idx}"
            files = os.listdir(f"{input_path}/{item}/{level}")

            images = [
                f"{input_path}/{item}/{level}/{image}"
                for image in files
                if image.endswith(".png")
            ]
            labels = "labels.json"
            print(f"Processing {item}/{level}/{labels}")
            with open(f"{input_path}/{item}/{level}/{labels}", "r") as f:
                labels = json.load(f)

            label_data = labels["bounding_boxes"]
            num_images = len(images)
            num_train = int(num_images * train_percent)
            num_val = int (num_images * val_percent)
            num_test = num_images - num_train - num_val

            train_images = images[:num_train]
            val_images = images[num_train:num_train + num_val]
            test_images = images[num_train + num_val:]

            for _, image in enumerate(train_images):
                image_label = label_data[image.split("/")[-1]]
                img = image.split("/")[-1]
                num_items = image_label["num_items"]
                boxes = image_label["bounding_boxes"]

                train_labels["images"].append(
                    {"id": image_idx, "file_name": img, "width": 225 * 2, "height": 150 * 2}
                )

                for anno_n in range(num_items):
                    box = boxes[anno_n]
                    box = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
                    train_labels["annotations"].append(
                        {
                            "id": annotation_idx,
                            "image_id": image_idx,
                            "category_id": item_idx + 1,
                            "bbox": box,
                        }
                    )
                    annotation_idx += 1

                image_idx += 1

            for _, image in enumerate(val_images):
                image_label = label_data[image.split("/")[-1]]
                img = image.split("/")[-1]
                num_items = image_label["num_items"]
                boxes = image_label["bounding_boxes"]

                val_labels["images"].append(
                    {"id": image_idx, "file_name": img, "width": 225 * 2, "height": 150 * 2}
                )

                for anno_n in range(num_items):
                    box = boxes[anno_n]
                    box = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
                    val_labels["annotations"].append(
                        {
                            "id": annotation_idx,
                            "image_id": image_idx,
                            "category_id": item_idx + 1,
                            "bbox": box,
                        }
                    )
                    annotation_idx += 1

                image_idx += 1

            for _, image in enumerate(test_images):
                image_label = label_data[image.split("/")[-1]]
                img = image.split("/")[-1]
                num_items = image_label["num_items"]
                boxes = image_label["bounding_boxes"]

                test_labels["images"].append(
                    {"id": image_idx, "file_name": img, "width": 225 * 2, "height": 150 * 2}
                )

                for anno_n in range(num_items):
                    box = boxes[anno_n]
                    box = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
                    test_labels["annotations"].append(
                        {
                            "id": annotation_idx,
                            "image_id": image_idx,
                            "category_id": item_idx + 1,
                            "bbox": box,
                        }
                    )
                    annotation_idx += 1

                image_idx += 1

            if not os.path.exists(f"{output_path}/train_data"):
                os.makedirs(f"{output_path}/train_data/images", exist_ok=True)
            if not os.path.exists(f"{output_path}/val_data"):
                os.makedirs(f"{output_path}/val_data/images", exist_ok=True)
            if not os.path.exists(f"{output_path}/test_data"):
                os.makedirs(f"{output_path}/test_data/images", exist_ok=True)

            for image in train_images:
                os.system(f"cp {image} {output_path}/train_data/images/")
            for image in val_images:
                os.system(f"cp {image} {output_path}/val_data/images/")
            for image in test_images:
                os.system(f"cp {image} {output_path}/test_data/images/")

    with open(f"{output_path}/train_data/labels.json", "w") as f:
        json.dump(train_labels, f, indent=4)
    with open(f"{output_path}/val_data/labels.json", "w") as f:
        json.dump(val_labels, f, indent=4)
    with open(f"{output_path}/test_data/labels.json", "w") as f:
        json.dump(test_labels, f, indent=4)


if __name__ == "__main__":
    main()
