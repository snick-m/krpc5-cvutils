"""
Author: Mushfiqur Rahman <realmrahman.19@gmail.com>

This script imports selected item image and generates training data of different levels for KIBO RPC 5th.
Levels are defined by min items, maximum items, scaling, rotation, and overlapping booleans.
At time of export, for each level, a labels.json file is also generated for each image with the following data:
    item_name: str
    num_items: int
    bounding_boxes: int[][4] # [x1, y1, x2, y2]
"""
import os
import json
import cv2
import numpy as np
import imutils

class Level:
    def __init__(
        self,
        min_items: int = 1,
        max_items: int = 1,
        scaling: bool = False,
        rotation: bool = False,
        overlapping: bool = False,
    ):
        self.min_items = min_items
        self.max_items = max_items
        self.scaling = scaling # Scale each item randomly between 0.5 and 1.2
        self.rotation = rotation # Rotate each item randomly between 0 and 360 degrees at stops of 45 degrees
        self.overlapping = overlapping # Have items partially overlap with each other, if not True make sure they don't overlap


levels = [
    Level(1, 1, False, False, False),
    Level(1, 3, True, False, False),
    Level(1, 5, True, True, False),
    Level(1, 5, True, True, True),
]

item_name = "screwdriver"
item_width = 60
image_dimensions = (150, 225)
image_per_level = 10

input_path = "item_inputs"
output_path = "data_outputs"

def getNonOverlappingBoundingBoxes(existing_bounding_boxes, scale):
    width = int(item_width * scale)
    x1 = np.random.randint(0, image_dimensions[1] - width)
    y1 = np.random.randint(0, image_dimensions[0] - width)
    x2 = x1 + width
    y2 = y1 + width

    new_bounding_box = [x1, y1, x2, y2]

    for bounding_box in existing_bounding_boxes:
        x1 = max(bounding_box[0], new_bounding_box[0])
        y1 = max(bounding_box[1], new_bounding_box[1])
        x2 = min(bounding_box[2], new_bounding_box[2])
        y2 = min(bounding_box[3], new_bounding_box[3])

        if x1 < x2 and y1 < y2:
            return getNonOverlappingBoundingBoxes(existing_bounding_boxes, scale)

    return new_bounding_box

def main():
    item = cv2.imread(f"{input_path}/{item_name}.png", cv2.IMREAD_UNCHANGED)
    
    item_alpha = item[:,:,3]
    item = item[:,:,:3]
    
    item = cv2.resize(item, (item_width, int(item_width * item.shape[1] / item.shape[0])))
    item_alpha = cv2.resize(item_alpha, (item_width, int(item_width * item_alpha.shape[1] / item_alpha.shape[0])))
    
    item_bg = np.ones(item.shape, np.uint8) * 255
    item_bg = cv2.bitwise_or(item_bg, item_bg, mask=cv2.bitwise_not(item_alpha))

    item = cv2.bitwise_or(item_bg, item)

    for i in range(len(levels)):
        print("Creating Level", i + 1, "Images")
        level = levels[i]
        images_bounding_boxes = {}

        for image_idx in range(image_per_level):
            num_items = np.random.randint(level.min_items, level.max_items + 1)
            bounding_boxes = []

            image = np.ones(image_dimensions + (3,), np.uint8) * 255

            for _ in range(num_items):
                item_copy = item.copy()
                item_alpha_copy = item_alpha.copy()
                item_copy = cv2.bitwise_and(item_copy, item_copy, mask=item_alpha_copy)
                scale = 1

                if level.scaling:
                    scale = np.random.uniform(0.7, 1.1)
                    item_copy = cv2.resize(item_copy, (int(item_width * scale), int(item_width * scale)))
                    item_alpha_copy = cv2.resize(item_alpha_copy, (int(item_width * scale), int(item_width * scale)))
                    
                x1 = np.random.randint(0, image_dimensions[1] - int(item_width * scale))
                y1 = np.random.randint(0, image_dimensions[0] - int(item_width * scale))
                x2 = x1 + int(item_width * scale)
                y2 = y1 + int(item_width * scale)

                if not level.overlapping:
                    try:
                        x1, y1, x2, y2 = getNonOverlappingBoundingBoxes(bounding_boxes, scale)
                    except RecursionError:
                        break

                if level.rotation: # Rotate between 0 and 360 degrees at stops of 45 degrees
                    angle = np.random.randint(0, 8) * 45

                    size = item_copy.shape[1], item_copy.shape[0]
                    item_copy = cv2.resize(imutils.rotate_bound(item_copy, angle), size)
                    item_alpha_copy = cv2.resize(imutils.rotate_bound(item_alpha_copy, angle), size)

                # Place item on image with alpha by converting alpha channel to a 3 channel image
                alpha = item_alpha_copy.astype(float) / 255
                alpha = cv2.merge((alpha, alpha, alpha))
                image_sub = cv2.multiply(1.0 - alpha, image[y1:y2, x1:x2].astype(float)).astype(np.uint8)
                image[y1:y2, x1:x2] = cv2.add(image_sub, item_copy)

                bounding_boxes.append([x1, y1, x2, y2])

            # cv2.imshow(f"Level {i + 1}", image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            image_name = f"{item_name}_level_{i + 1}_{image_idx + 1}.png"
            image_path = f"{output_path}/{item_name}/level_{i + 1}/{image_name}"
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            cv2.imwrite(image_path, image)
            
            print('\t', image_path)

            images_bounding_boxes[image_name] = {
                "num_items": len(bounding_boxes),
                "bounding_boxes": bounding_boxes
            }

        print("\n")

        with open(f"{output_path}/{item_name}/level_{i + 1}/labels.json", "w") as f:
            f.write(json.dumps({
                "item_name": item_name,
                "images_path": f"level_{i + 1}/",
                "bounding_boxes": images_bounding_boxes
            }, indent=4))

if __name__ == "__main__":
    main()