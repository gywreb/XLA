import cv2 as cv
import numpy as np

import DetectLetters as dletter
import PreProcess as prep

dletter.load_and_train_knn()

contours1 = []
contours2 = []
listChar = []

img = cv.imread("data2/99.png")

img_gray, img_prep = prep.preprocess(img)

listChar = dletter.find_possible_letter(img_prep)
list_match = dletter.find_list_of_lists_matching_letters(listChar)
# create a black img
height, width, channels = img.shape
img_contours1 = np.zeros((height, width), np.uint8)
img_contours2 = np.zeros((height, width), np.uint8)

del contours1[:]  # clear the contours list
del contours2[:]

for list_char in list_match:
    for possibleChar in list_char:
        contours1.append(possibleChar.contour)

    for possibleChar in listChar:
        contours2.append(possibleChar.contour)
    # draw contours
    cv.drawContours(img_contours1, contours1, -1, (255, 255, 255))
    cv.drawContours(img_contours2, contours2, -1, (255, 255, 255))

for list_char in list_match:
    list_match_without_overlapping = dletter.remove_inner_overlapping_letters(list_char)
    res, img_color = dletter.regconize_letters_in_plate(img, img_prep, list_match_without_overlapping)
    print(res)

    cv.imshow("img_letter", img_color)
    cv.waitKey(0)
    cv.destroyAllWindows()

cv.imshow("img_contours", img_contours1)
cv.imshow("img_contours2", img_contours2)
cv.imshow("img", img_prep)

cv.waitKey(0)
cv.destroyAllWindows()
