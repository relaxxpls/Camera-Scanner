import cv2
import matplotlib.pyplot as plt
import numpy as np


def merge_img(queryImg, trainImg):
    """
    Merge two images into one
    """

    trainImg_gray = cv2.cvtColor(trainImg, cv2.COLOR_RGB2GRAY)
    queryImg_gray = cv2.cvtColor(queryImg, cv2.COLOR_RGB2GRAY)

    descriptor = cv2.ORB_create()
    kpsA, featuresA = descriptor.detectAndCompute(trainImg_gray, None)
    kpsB, featuresB = descriptor.detectAndCompute(queryImg_gray, None)

    # ###############
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # ? compute the raw matches and initialize the list of actual matches
    rawMatches = bf.knnMatch(featuresA, featuresB, 2)
    print("Raw matches (knn):", len(rawMatches))

    matches = []
    ratio = 0.75

    # ? loop over the raw matches and ensure the distance is within
    # ? a certain ratio of each other (i.e. Lowe's ratio test)
    for m, n in rawMatches:
        if m.distance < n.distance * ratio:
            matches.append(m)

    # ###############
    # ? construct the two sets of points
    ptsA = np.float32([kpsA[m.queryIdx].pt for m in matches])
    ptsB = np.float32([kpsB[m.trainIdx].pt for m in matches])

    # ? estimate the homography between the sets of points
    H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold=4)
    print(H)

    height = trainImg.shape[0] + queryImg.shape[0]
    width = trainImg.shape[1] + queryImg.shape[1]

    result = np.zeros((height, width, 3), dtype=np.uint8)
    result = cv2.warpPerspective(trainImg, H, (width, height))
    result[0 : queryImg.shape[0], 0 : queryImg.shape[1]] = queryImg

    return result


if __name__ == "__main__":
    img1 = cv2.imread("dataset/3/2.jpeg")
    img2 = cv2.imread("dataset/3/1.jpeg")

    merged_img = merge_img(img1, img2)

    fig, ax = plt.subplots(nrows=1, ncols=2, constrained_layout=False, figsize=(25, 9))
    ax[0].imshow(img1)
    ax[0].set_title("Image 1", fontsize=16)
    ax[0].axis("off")

    ax[1].imshow(img2)
    ax[1].set_title("Image 2", fontsize=16)
    ax[1].axis("off")

    plt.imshow(merged_img)

    plt.figure(figsize=(20, 10))
    plt.axis("off")
