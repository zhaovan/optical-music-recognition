from skimage.morphology import flood
import numpy as np
import random as rng

from preprocessing.staff_removal import staff_removal

import cv2


# Takes in an image and the average height of the note
def create_bounding_boxes(image, height=30):
    image_len = len(image)
    image_width = len(image[0])

    returned_img = image.copy()

    bound_arr = []

    for i in range(image_len):
        for j in range(image_width):

            blob_len = 0
            if image[i, j] == 1:
                mask = flood(image, (i, j))
                blob_len = np.maximum(np.sum(mask, axis=0))
                blob_width = np.maximum(np.sum(mask, axis=0))

            if blob_len > height:
                bounding_rec = cv2.boundingRect(mask)
                bound_arr.append(bounding_rec)

    print(bound_arr)

    bound_arr = np.array(bound_arr)
    for i in range(len(bound_arr)):
        color = (256, 256, 256)
        cv2.rectangle(returned_img, (int(bound_arr[i][0]), int(bound_arr[i][1])),
                      (int(bound_arr[i][0]+bound_arr[i][2]), int(bound_arr[i][1]+bound_arr[i][3])), color, 2)

    cv2.imshow("Boxes", returned_img)
    cv2.waitKey(0)

    return returned_img


def main():
    # img = cv2.imread('../results/processed.png', cv2.IMREAD_COLOR)

    img = staff_removal('../results/processed.png')

    create_bounding_boxes(img)


if __name__ == "__main__":
    main()
