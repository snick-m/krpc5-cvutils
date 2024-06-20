import cv2
import json

# 05-20 01:27:28.629 10199 10199 I DETECT  : image_1c_cropped.png
# 05-20 01:27:28.818 10199 10199 I DETECT  : Detected: beaker; Confidence 0.8233063; Location: [0.027652213, 0.19217357, 0.047179833, -0.5084606]
# 05-20 01:27:28.818 10199 10199 I DETECT  : Detected: beaker; Confidence 0.93917155; Location: [0.04418211, -0.06582195, -0.10650474, 0.051615328]
# 05-20 01:27:28.818 10199 10199 I DETECT  : Detected: beaker; Confidence 0.85366344; Location: [0.056109767, -0.044014167, 0.26563698, -0.30249855]
# 05-20 01:27:28.818 10199 10199 I DETECT  : Detected: beaker; Confidence 0.8304632; Location: [0.04063418, -0.01745231, 0.04912975, -0.4639119]
# 05-20 01:27:28.818 10199 10199 I DETECT  : Detected: beaker; Confidence 0.7840786; Location: [-0.1676221, -0.067124315, -0.09524428, 0.048288792]
# 05-20 01:27:28.818 10199 10199 I DETECT  : Detected: kapton_tape; Confidence 0.8577009; Location: [-0.041779503, 0.07816897, 0.01975371, 0.24304958]
# 05-20 01:27:28.818 10199 10199 I DETECT  : Detected: kapton_tape; Confidence 0.84406203; Location: [-0.034876052, -0.19823083, 0.012903228, 0.22489026]

detectionInfo = {
    "image": "image_1c_cropped.png",
    "detections": [
        {
            "label": "beaker",
            "confidence": 0.8233063,
            "location": [0.027652213, 0.19217357, 0.047179833, -0.5084606]
        },
        {
            "label": "beaker",
            "confidence": 0.93917155,
            "location": [0.04418211, -0.06582195, -0.10650474, 0.051615328]
        },
        {
            "label": "beaker",
            "confidence": 0.85366344,
            "location": [0.056109767, -0.044014167, 0.26563698, -0.30249855]
        },
        {
            "label": "beaker",
            "confidence": 0.8304632,
            "location": [0.04063418, -0.01745231, 0.04912975, -0.4639119]
        },
        {
            "label": "beaker",
            "confidence": 0.7840786,
            "location": [-0.1676221, -0.067124315, -0.09524428, 0.048288792]
        },
        {
            "label": "kapton_tape",
            "confidence": 0.8577009,
            "location": [-0.041779503, 0.07816897, 0.01975371, 0.24304958]
        },
        {
            "label": "kapton_tape",
            "confidence": 0.84406203,
            "location": [-0.034876052, -0.19823083, 0.012903228, 0.22489026]
        }
    ]
}

def main():
    image = cv2.imread("images/image_1c_cropped.png")
    image_resized = cv2.resize(image, (256, 256))

    for detection in detectionInfo["detections"]:
        label = detection["label"]
        confidence = detection["confidence"]
        location = detection["location"]

        print([min(256, max(0, int(128 + coord * 256))) for coord in location])

        x1 = min(256, max(0, int(128 + location[1] * 256)))
        y1 = min(256, max(0, int(128 + location[0] * 256)))
        x2 = min(256, max(0, int(128 + location[2] * 256)))
        y2 = min(256, max(0, int(128 + location[3] * 256)))

        # x1 = int((location[1] + 0.5) * 256)
        # y1 = int((location[0] + 0.5) * 256)
        # x2 = int((location[3] + 0.5) * 256)
        # y2 = int((location[2] + 0.5) * 256)

        cv2.rectangle(image_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_resized, f"{label} {confidence:.2f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    image = cv2.resize(image_resized, (270, 150))

    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()