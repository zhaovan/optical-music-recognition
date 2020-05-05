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

    # color = (25, 25, 25)

    bounding_boxes = np.zeros((len(contours), 4))

    for i in range(len(contours)):
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        boxed_img = raw_image[y: y+h, x: x+w]
        if np.size(boxed_img) > 0:
            bounding_boxes[i] = [x, y, w, h]
        #     cv2.imshow('Box', boxed_img)
        #     cv2.waitKey(0)

        # cv2.rectangle(drawing, (int(x), int(y)),
        #               (int(x + w), int(y + h)), color, 2)

    # cv2.imshow("Boxes", drawing)
    # cv2.waitKey(0)

    return bounding_boxes

def consolidate(bb):
    index = bb.shape(0)
    print(index)
    while (index > 0): 
        dimensionsA = bb.shape[index]
        dimensionsB = bb.shape[index - 1]
        range_x_one = int(bb.shape[0] - .1 * bb.shape[w])
        range_x_two = int(bb.shape[0] - 1.1 * bb.shape[w])

        if (dimensionsB[0] )
        bb[index]
