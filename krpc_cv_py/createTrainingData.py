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
import re
from typing import List
import cv2
import numpy as np
import imutils
import argparse

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


# Define levels here
# Level(min_items, max_items, scaling, rotation, overlapping)
levels = [
    # Level(1, 1, False, False, False),
    # Level(1, 3, True, False, False),
    Level(1, 4, True, True, False),
    Level(3, 5, True, True, True),
]

# item_name = "screwdriver"
item_width = 60 * 2
item_height = 0
image_dimensions = (150 * 2, 225 * 2)
# image_per_level = 5

input_path = "item_inputs"
output_path = "data_outputs"

def getBoundingBoxWithOverlap(bounding_boxes: List[List[int]], width: int, height: int, allowed_overlap: float):
    """
    Generate a random bounding box that overlaps with existing bounding boxes but only to certain degree.
    
    :param bounding_boxes: List of existing bounding boxes
    :type bounding_boxes: List[List[int]]
    :param width: Width of the new bounding box
    :type width: int
    :param height: Height of the new bounding box
    :type height: int
    :param allowed_overlap: Allowed overlap percentage with existing bounding boxes
    :type allowed_overlap: float

    :return: A new bounding box that overlaps with existing bounding boxes
    :rtype: [int, int, int, int]
    """
    x1 = np.random.randint(0, image_dimensions[1] - width)
    y1 = np.random.randint(0, image_dimensions[0] - height)
    x2 = x1 + width
    y2 = y1 + height

    new_bounding_box = [x1, y1, x2, y2]

    for bounding_box in bounding_boxes:
        x1_overlap = max(bounding_box[0], new_bounding_box[0])
        y1_overlap = max(bounding_box[1], new_bounding_box[1])
        x2_overlap = min(bounding_box[2], new_bounding_box[2])
        y2_overlap = min(bounding_box[3], new_bounding_box[3])

        overlap_width = max(0, x2_overlap - x1_overlap)
        overlap_height = max(0, y2_overlap - y1_overlap)

        overlap_area = overlap_width * overlap_height
        total_area = width * height

        if overlap_area / total_area > allowed_overlap:
            return getBoundingBoxWithOverlap(bounding_boxes, width, height, allowed_overlap)

    return new_bounding_box

def getNonOverlappingBoundingBoxes(existing_bounding_boxes: List[List[int]], width: int, height: int):
    """
    Generate a random bounding box that does not overlap with existing bounding boxes.
    If it does, recursively call the function until a non-overlapping bounding box is found.
    
    :param existing_bounding_boxes: List of existing bounding boxes
    :type existing_bounding_boxes: List[List[int]]
    :param width: Width of the new bounding box
    :type width: int
    :param height: Height of the new bounding box
    :type height: int
    :return: A new bounding box that does not overlap with existing bounding boxes
    :rtype: [int, int, int, int]
    """
    x1 = np.random.randint(0, image_dimensions[1] - width)
    y1 = np.random.randint(0, image_dimensions[0] - height)
    x2 = x1 + width
    y2 = y1 + height

    new_bounding_box = [x1, y1, x2, y2]

    for bounding_box in existing_bounding_boxes:
        x1 = max(bounding_box[0], new_bounding_box[0])
        y1 = max(bounding_box[1], new_bounding_box[1])
        x2 = min(bounding_box[2], new_bounding_box[2])
        y2 = min(bounding_box[3], new_bounding_box[3])

        if x1 < x2 and y1 < y2:
            return getNonOverlappingBoundingBoxes(existing_bounding_boxes, width, height)

    return new_bounding_box

def rotate_image(img, angle):
    size_reverse = np.array(img.shape[1::-1]) # swap x with y
    M = cv2.getRotationMatrix2D(tuple(size_reverse / 2.), angle, 1.)
    MM = np.absolute(M[:,:2])
    size_new = MM @ size_reverse
    M[:,-1] += (size_new - size_reverse) / 2.
    rotated_image = cv2.warpAffine(img, M, tuple(size_new.astype(int)), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    return rotated_image

def trim_image(image):    
    # Find the non-transparent pixels (where alpha > 0)
    if image.shape[2] == 4:  # Ensure we have an alpha channel
        alpha_channel = image[:, :, 3]
        # Find coordinates of non-transparent pixels
        non_transparent_coords = np.column_stack(np.where(alpha_channel > 0))
        
        if len(non_transparent_coords) > 0:
            # Get the bounding box of non-transparent pixels
            min_row, min_col = np.min(non_transparent_coords, axis=0)
            max_row, max_col = np.max(non_transparent_coords, axis=0)
            
            # Crop the image to the bounding box
            cropped_image = image[min_row:max_row+1, min_col:max_col+1]
            return cropped_image
    
    return image # If no non-transparent pixels, return the original image

def main(item_name: str, image_per_level: int, allowed_overlap: float = 0.5):
    item_name = item_name.split(".")[0]
    item = cv2.imread(f"{input_path}/{item_name}.png", cv2.IMREAD_UNCHANGED)
    
    global item_height, item_width

    item_width = 65 * 2
    item_height = 0

    if item.shape[0] > item.shape[1]:
        item_height = item_width
        item_width = int(item_height * item.shape[1] / item.shape[0])
    else:
        item_height = int(item_width * item.shape[0] / item.shape[1])
    
    item = cv2.resize(item, (item_width, item_height), interpolation=cv2.INTER_AREA)
    for i in range(len(levels)):
        print("Creating Level", i + 1, "Images")
        level = levels[i]
        images_bounding_boxes = {}

        for image_idx in range(image_per_level):
            num_items = np.random.randint(level.min_items, level.max_items + 1)
            bounding_boxes = []

            image = np.ones(image_dimensions + (4,), np.uint8) * 255

            for _ in range(num_items):
                item_copy = item.copy()

                scale = 1

                if level.scaling:
                    scale = np.random.uniform(0.65, 1.0)
                    item_copy = cv2.resize(item_copy, (int(item_width * scale), int(item_height * scale)), interpolation=cv2.INTER_AREA)

                if level.rotation: # Rotate between 0 and 360 degrees at stops of 45 degrees
                    angle = np.random.randint(0, 8) * 45
                    item_copy = rotate_image(item_copy, angle)
                
                item_copy = trim_image(item_copy) # Trim the image to remove transparent pixels
                    
                x1 = np.random.randint(0, image_dimensions[1] - item_copy.shape[1])
                y1 = np.random.randint(0, image_dimensions[0] - item_copy.shape[0])
                x2 = x1 + item_copy.shape[1]
                y2 = y1 + item_copy.shape[0]

                if not level.overlapping:
                    try:
                        x1, y1, x2, y2 = getNonOverlappingBoundingBoxes(bounding_boxes, item_copy.shape[1], item_copy.shape[0])
                    except RecursionError:
                        break
                else:
                    try:
                        x1, y1, x2, y2 = getBoundingBoxWithOverlap(bounding_boxes, item_copy.shape[1], item_copy.shape[0], allowed_overlap)
                    except RecursionError:
                        break

                # Place item on image with alpha by converting alpha channel to a 3 channel image
                alpha = cv2.merge((item_copy[:,:,3] / 255,) * 4).astype(np.uint8) # Invert alpha to black out the item
                item_copy = cv2.multiply(alpha, item_copy) # Black out pixels outside the item on item image
                
                image_sub = cv2.multiply(1.0 - alpha, image[y1:y2, x1:x2].astype(float)).astype(np.uint8) # Black out pixels inside the item on main image
                image[y1:y2, x1:x2] = cv2.add(image_sub, item_copy) # (Black Outside + Item) + (Black Inside + Main Image)

                bounding_boxes.append([x1, y1, x2, y2])

            image_name = f"{item_name}_level_{i + 1}_{image_idx + 1}.png"
            image_path = f"{output_path}/{item_name}/level_{i + 1}/{image_name}"
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            # Apply a random brightness of 0.7 to 0.95
            image = cv2.convertScaleAbs(image, alpha=1, beta=np.random.uniform(-0.7, -0.2) * 127)
            # Apply Gaussian blur with a kernel size of 2
            image = cv2.GaussianBlur(image, (3, 3), 0)
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
    parser = argparse.ArgumentParser(description="Generate training data for KIBO RPC 5th")
    parser.add_argument("-n", "--item_name", type=str, default="_all", help="Name of the item to generate data for")
    parser.add_argument("-c", "--per_level", type=int, default=5, help="Number of images per level")
    parser.add_argument("-o", "--allowed_overlap", type=float, default=0.5, help="Allowed overlap percentage with existing bounding boxes")

    item_name = parser.parse_args().item_name
    image_per_level = parser.parse_args().per_level
    allowed_overlap = parser.parse_args().allowed_overlap

    if item_name == "_all":
        items = os.listdir(input_path)
        for item in items:
            print("Generating data for", item)
            main(item, image_per_level, allowed_overlap)
    else:
        main(item_name, image_per_level, allowed_overlap)