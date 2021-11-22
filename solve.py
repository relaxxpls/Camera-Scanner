import cv2
import matplotlib.pyplot as plt
import numpy as np


def match_keypoints(featuresA, featuresB):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # ? compute the raw matches and initialize the list of actual matches
    rawMatches = bf.knnMatch(featuresA, featuresB, 2)
    matches = []
    ratio = 0.75

    # ? loop over the raw matches and ensure the distance is within
    # ? a certain ratio of each other (i.e. Lowe's ratio test)
    for m, n in rawMatches:
        if m.distance < n.distance * ratio:
            matches.append(m)

    return matches


def remove_black_border(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    x, y, w, h = cv2.boundingRect(contours[0])

    return img[y : y + h, x : x + w]


def merge_img(queryImg, trainImg):
    """
    Merge two images into one
    """

    trainImg_gray = cv2.cvtColor(trainImg, cv2.COLOR_RGB2GRAY)
    queryImg_gray = cv2.cvtColor(queryImg, cv2.COLOR_RGB2GRAY)

    descriptor = cv2.ORB_create()
    kpsA, featuresA = descriptor.detectAndCompute(trainImg_gray, None)
    kpsB, featuresB = descriptor.detectAndCompute(queryImg_gray, None)

    matches = match_keypoints(featuresA, featuresB)

    # ? construct the two sets of points
    ptsA = np.float32([kpsA[m.queryIdx].pt for m in matches])
    ptsB = np.float32([kpsB[m.trainIdx].pt for m in matches])

    # ? estimate the homography between the sets of points
    H, _ = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold=4)

    height = trainImg.shape[0] + queryImg.shape[0]
    width = trainImg.shape[1] + queryImg.shape[1]

    result = np.zeros((height, width, 3), dtype=np.uint8)
    result = cv2.warpPerspective(trainImg, H, (width, height))
    result[0 : queryImg.shape[0], 0 : queryImg.shape[1]] = queryImg

    result = remove_black_border(result)

    return result


if __name__ == "__main__":
    img1 = cv2.imread("dataset/3/1.jpeg")
    img2 = cv2.imread("dataset/3/2.jpeg")

    merged_img = merge_img(img1, img2)

    plt.figure(figsize=(25, 9))
    plt.subplot(2, 2, 1)
    plt.imshow(img1)
    plt.title("Image 1", fontsize=16)
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(img2)
    plt.title("Image 2", fontsize=16)
    plt.axis("off")

    plt.subplot(2, 1, 2)
    plt.imshow(merged_img)
    plt.title("Combined image", fontsize=16)
    plt.axis("off")

    plt.show()
