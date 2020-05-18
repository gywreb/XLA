import cv2 as cv

# ############################
# GLOBAL VAR
GAUSSIAN_KERNEL_SIZE = (1, 1)
MAXVAL = 255
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_C = 9


##############################

##########################################################
# function
def get_grayscale(img_original):
    img_hsv = cv.cvtColor(img_original, cv.COLOR_BGR2HSV)

    img_hue, img_sat, img_value = cv.split(img_hsv)

    # get the GrayScale img as imgValue
    return img_value


def maximize_contrast(img_grayscale):
    # kernel
    structuring_element = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

    img_tophat = cv.morphologyEx(img_grayscale, cv.MORPH_TOPHAT, structuring_element)
    img_blackhat = cv.morphologyEx(img_grayscale, cv.MORPH_BLACKHAT, structuring_element)

    img_grayscale_plus_tophat = cv.add(img_grayscale, img_tophat)
    img_grayscale_plus_tophat_sub_blachat = cv.subtract(img_grayscale_plus_tophat, img_blackhat)

    return img_grayscale_plus_tophat_sub_blachat


def preprocess(img_original):
    # grayscale --> maxcontrast --> blur --> threshold
    img_grayscale = get_grayscale(img_original)

    img_maxcontrast_grayscale = maximize_contrast(img_grayscale)

    # sigmaX, sigmaY = 0
    img_blur = cv.GaussianBlur(img_maxcontrast_grayscale, GAUSSIAN_KERNEL_SIZE, 0)
    img_thresh = cv.adaptiveThreshold(img_blur, MAXVAL, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,
                                      ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_C)

    return img_grayscale, img_thresh

###################################################################
# # test function
# img = cv.imread("data2/1.png")
#
# imgGrayScale = get_grayscale(img)
# img_maxcontrast = maximize_contrast(imgGrayScale)
# img_gs, img_preprocess = preprocess(img)
#
# cv.imshow("img", imgGrayScale)
# cv.imshow("img2", img_maxcontrast)
# cv.imshow("img3", img_preprocess)
#
# cv.waitKey(0)
# cv.destroyAllWindows()
