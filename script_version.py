import sys
import cv2
import os.path
import glob as gb
import numpy as np
import matplotlib.pylab as plt

"""
C:/Users/Administrator/AppData/Local/Programs/Python/Python39/python.exe C:/Users/Administrator/Desktop/license_plate/script_version.py ./images_test/mat/mat1.png ./images_test/car
"""


def sift_matches(img1, img2, match_ratio, shrink, display_matches=False):

    if img1.ndim > 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if img2.ndim > 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    if shrink < 1:
        img1 = cv2.resize(
            img1, (0, 0), fx=shrink, fy=shrink, interpolation=cv2.INTER_CUBIC
        )
        img2 = cv2.resize(
            img2, (0, 0), fx=shrink, fy=shrink, interpolation=cv2.INTER_CUBIC
        )

    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    all_matches = bf.knnMatch(des1, des2, k=2)

    matches = np.asarray(
        [[m] for m, d in all_matches if m.distance < match_ratio * d.distance]
    )

    if display_matches:
        # draw good matches
        matchedKnn_img = cv2.drawMatchesKnn(
            img1, kp1, img2, kp2, matches, img1, matchColor=(0, 255, 0), flags=2
        )
        # show matching image
        plt.imshow(matchedKnn_img[:, :, ::-1]), plt.title(
            "SIFT's good matches"
        ), plt.axis("off")
        plt.show()

    return matches


mat_path = str(sys.argv[1])
if not os.path.exists(mat_path):
    raise Exception("Sorry, the license plate image don't exist")

mat_img = cv2.imread(mat_path)
print("license plate image read")

cars_path = str(sys.argv[2])
if not os.path.isdir(cars_path):
    raise Exception("Sorry, the cars images directory don't exist")

car_images_path = sorted(gb.glob(cars_path + "/*.jpg"), key=len)
cars = [cv2.imread(img) for img in car_images_path]
print(f"numbre of cars ... {len(car_images_path)}")

print("search for best matche...")
lst = np.array([])
for car in cars:
    matches = sift_matches(
        mat_img, car, match_ratio=0.75, shrink=0.7, display_matches=True
    )
    lst = np.append(lst, len(matches))

max_idx = lst.argsort()[-1]
find_car = cars[max_idx]
if find_car.shape[0] > 700 and find_car.shape[1] > 700:
    find_car = cv2.resize(
        find_car, (0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC
    )
print("show best result")
# plt.figure(figsize=(20, 20))
plt.subplot(1, 2, 1)
plt.imshow(mat_img[:, :, ::-1]), plt.title("current licance plate"), plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(find_car[:, :, ::-1]), plt.title("best matched car"), plt.axis("off")
plt.show()
