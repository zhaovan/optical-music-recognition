import numpy as np
import cv2


def make_bounding_boxes(image):
    raw_image = image

    # cv2.imshow('Original Image', raw_image)
    # cv2.waitKey(0)

    bilateral_filtered_image = cv2.bilateralFilter(raw_image, 5, 175, 175)
    # cv2.imshow('Bilateral', bilateral_filtered_image)
    # cv2.waitKey(0)

    edge_detected_image = cv2.Canny(bilateral_filtered_image, 75, 200)
    # cv2.imshow('Edge', edge_detected_image)
    # cv2.waitKey(0)

    _, contours, _ = cv2.findContours(
        edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    drawing = raw_image.copy()

    color = (25, 25, 25)

    bounding_boxes = np.zeros((len(contours), 4))

    for i in range(len(contours)):
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        boxed_img = raw_image[y: y+h, x: x+w]
        if np.size(boxed_img) > 0:
            bounding_boxes[i] = [x, y, w, h]

    # print(bounding_boxes.shape)

    # print(bounding_boxes)

    bounding_boxes = list(bounding_boxes)

    max_bound_boxes, _ = cv2.groupRectangles(bounding_boxes, 1, 0.9)
    # return bounding_boxes

    return np.array(max_bound_boxes).astype(int)
