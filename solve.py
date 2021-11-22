from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance


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


def get_boundary_points(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    epsilon = 0.005 * cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], epsilon, True)

    # # draw
    # cv2.drawContours(img, contours, -1, (255, 255, 255), 3)
    # cv2.imshow("Contours", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # print(f"contours: {len(contours)}")
    # print(f"largest contour has {len(contours[0])} points")

    # print(f"eps: {epsilon}")
    # cv2.drawContours(img, [approx], 0, (255, 255, 255), 3)
    # cv2.imshow("Contours", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return approx


def order_points(pts):
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    dist = distance.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(dist)[::-1], :]

    return np.array([tl, tr, br, bl], dtype="float32")


def four_point_transform(image, pts):
    pts = np.array(pts, dtype="float32")
    src = order_points(pts)
    tl, tr, br, bl = src

    widthA, widthB = np.linalg.norm(br - bl), np.linalg.norm(tr - tl)
    heightA, heightB = np.linalg.norm(tr - br), np.linalg.norm(tl - bl)
    maxWidth, maxHeight = int(max(widthA, widthB)), int(max(heightA, heightB))

    dst = [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]]
    dst = np.array(dst, dtype="float32")
    dst = order_points(dst)

    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        image, matrix, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR
    )

    return warped


def merge_img(queryImg, trainImg):
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


# def display_result(img1, img2, img12):
#     plt.figure(figsize=(25, 9))
#     plt.subplot(2, 2, 1)
#     plt.imshow(img1)
#     plt.title("Image 1", fontsize=16)
#     plt.axis("off")

#     plt.subplot(2, 2, 2)
#     plt.imshow(img2)
#     plt.title("Image 2", fontsize=16)
#     plt.axis("off")

#     plt.subplot(2, 1, 2)
#     plt.imshow(img12)
#     plt.title("Merged image", fontsize=16)
#     plt.axis("off")

#     plt.subplot(2, 1, 2)
#     plt.imshow(img12)
#     plt.title("Merged + warped image", fontsize=16)
#     plt.axis("off")

#     plt.show()


def display_result(img1, img2, img3, img4):
    plt.figure(figsize=(25, 9))

    plt.subplot(2, 2, 1)
    plt.imshow(img1)
    plt.title("Image 1", fontsize=16)
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(img2)
    plt.title("Image 2", fontsize=16)
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(img3)
    plt.title("Merged image", fontsize=16)
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(img4)
    plt.title("Merged + warped image", fontsize=16)
    plt.axis("off")

    plt.show()


def mainIndividual(images):
    for i in range(1, len(images)):
        image_merged = merge_img(images[i - 1], images[i])

        points = get_boundary_points(image_merged)
        print(f"Simplified contour has {len(points)} points")

        if len(points) < 4 or len(points) > 6:
            print(f"{i}th image has {len(points)} points")
            break
        elif len(points) == 4:
            warped = four_point_transform(image_merged, points)
        elif len(points) == 5:
            warped = four_point_transform(
                image_merged, [points[i][0] for i in [0, 1, 3, 4]]
            )
        elif len(points) == 6:
            warped = four_point_transform(
                image_merged, [points[i][0] for i in [0, 1, 3, 5]]
            )

        display_result(images[i - 1], images[i], image_merged, warped)


if __name__ == "__main__":
    IMG_DIR = "dataset/3"
    filepaths = sorted(Path(IMG_DIR).glob("*.jpeg"))

    images = [cv2.imread(str(path)) for path in filepaths]
    mainIndividual(images)

    # img1 = cv2.imread("dataset/3/0.jpeg")
    # img2 = cv2.imread("dataset/3/1.jpeg")
    # img12 = merge_img(img1, img2)
    # display_result(img1, img2, img12)

    # img = image_merged.copy()
    # print(points)
    # for [[x, y]] in points:
    #     cv2.putText(
    #         img,
    #         f"{x}, {y}",
    #         (x, y),
    #         cv2.FONT_HERSHEY_SIMPLEX,
    #         1,
    #         (255, 0, 0),
    #         2,
    #     )

    # cv2.imshow("image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
