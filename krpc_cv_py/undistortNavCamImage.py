import cv2
import numpy as np

def main():
    img = cv2.imread("images/image_0.png", cv2.IMREAD_COLOR)

    # Undistort
    # From PGManual.pdf
    camera_matrix = np.array([[523.105750, 0.000000, 635.434258], [0.000000, 534.765913, 500.335102], [0.000000, 0.000000, 1.000000]])
    distortion_coefficients = np.array([-0.164787, 0.020375, -0.001572, -0.000369, 0.000000])

    img = cv2.undistort(img, camera_matrix, distortion_coefficients)

    # Detect aruco
    aruco_params = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, aruco_params)
    img = cv2.aruco.drawDetectedMarkers(img, corners, ids)

    # Homograph
    print(img.shape)
    # Corners TL, TR, BR, BL: [207.5:12.5, 257.5:12.5, 257.5:50, 207.5:50]
    pts1 = np.float32(corners[0])
    pts2 = np.float32([[208,12], [258,12], [258,50], [208,50]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img = cv2.warpPerspective(img, matrix, (270, 150))
    
    cv2.imshow("image", img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()