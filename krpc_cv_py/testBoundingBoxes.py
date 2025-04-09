import cv2
import numpy as np
import os
import json

input_path = "data_outputs"


def main():
    generated_items = os.listdir(input_path)

    for item in generated_items:
        for i in range(1, 5):
            level = f"level_{i}"
            try:
                files = os.listdir(f"{input_path}/{item}/{level}")
            except FileNotFoundError:
                continue
            files.remove("labels.json")
            images = zip(files, [
                cv2.imread(f"{input_path}/{item}/{level}/{image}")
                for image in files
                if image.endswith(".png")
            ])
            labels = "labels.json"
            with open(f"{input_path}/{item}/{level}/{labels}", "r") as f:
                labels = json.load(f)
            
            label_data = labels["bounding_boxes"]
            print(labels["item_name"])
            print(labels["images_path"])
            for image in images:
                image_label = label_data[image[0]]
                img = image[1]
                num_items = image_label["num_items"]
                boxes = image_label["bounding_boxes"]

                for i in range(num_items):
                    box = boxes[i]
                    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                
                cv2.imshow("image", img)
                key = cv2.waitKey(0)
                if key & 0xFF == ord('n'): # Next image
                    cv2.destroyAllWindows()
                elif key & 0xFF == ord('s'): # Skip to next level
                    cv2.destroyAllWindows()
                    break
                elif key & 0xFF == ord('q'): # Quit program
                    cv2.destroyAllWindows()
                    exit()


if __name__ == "__main__":
    main()
