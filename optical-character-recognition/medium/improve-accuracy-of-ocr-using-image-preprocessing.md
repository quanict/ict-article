# [Improve Accuracy of OCR using Image Preprocessing][source]

OCR stands for Optical Character Recognition, the conversion of a document photo or scene photo into machine-encoded text. There are many tools available to implement OCR in your system such as [Tesseract OCR][01] and [Cloud Vision][02]. They use AI and Machine Learning as well as trained custom model. Text Recognition depends on a variety of factors to produce a good quality output. OCR output highly depends on the quality of input image. This is why every OCR engine provides guidelines regarding the quality of input image and its size. These guidelines help OCR engine to produce accurate results.

Here Image Preprocessing comes into play to improve the quality of input image so that the OCR engine gives you an accurate output. Use the following image processing operation to improve the quality of your input image.

![i01]

## Scaling of image :

Image Rescaling is important for image analysis. Mostly OCR engine give an accurate output of the image which has 300 DPI. DPI describes the resolution of the image or in other words, it denotes printed dots per inch.

```py
def set_image_dpi(file_path):
    im = Image.open(file_path)
    length_x, width_y = im.size
    factor = min(1, float(1024.0 / length_x))
    size = int(factor * length_x), int(factor * width_y)
    im_resized = im.resize(size, Image.ANTIALIAS)
    temp_file = tempfile.NamedTemporaryFile(delete=False,   suffix=
'.png')
    temp_filename = temp_file.name
    im_resized.save(temp_filename, dpi=(300, 300))
    
return temp_filename
```

![i02]

## Skew Correction :

A Skewed image is defined as a document image which is not straight. Skewed images directly impact the line segmentation of OCR engine which reduces its accuracy. We need to process the following steps to correct text skew.

1. Detect the text block with skew in the image.

```py
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 10, 50)
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
screenCnt = None
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        screenCnt = approx
        break

cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
```

![i03]

2. Now calculate the angle of rotation.

3. Rotating the image to correct the skew.

```py
pts = np.array(screenCnt.reshape(4, 2) * ratio)
warped = four_point_transform(orig, pts)
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped
```

![i04]


## Binarization :

Mostly, an OCR engine does binarization internally because they work on Black & White images. The simplest way is to calculate a threshold value and convert all pixels to white with a value above threshold value and rest of pixels convert into the black. I am using OpenCV with Simple Thresholding, Adaptive Thresholding, and Otsu’s Binarization.

## Noise Removal or Denoise :

Noise is a random variation of color or brightness between pixels of an image. Noise decrease the readability of text from an image. There are two major types of noise - salt and pepper noise and Gaussian noise.

```py
def remove_noise_and_smooth(file_name):
    img = cv2.imread(file_name, 0)
    filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 41)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    img = image_smoothening(img)
    or_image = cv2.bitwise_or(img, closing)
    return or_image

```

## Useful Links :

There are plenty of tools and articles for image processing on the internet. I am putting up some useful links which can help you to implement these process to improve your accuracy.

1. Image processing [OpenCV][03]
2. [Skew correction][04] using python
3. Mobile [document scanner][05]

-------------------------
[05]: https://www.pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/
[04]: https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
[03]: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_table_of_contents_imgproc/py_table_of_contents_imgproc.html
[02]: https://cloud.google.com/vision/
[01]: https://github.com/tesseract-ocr/tesseract
[source]: https://medium.com/cashify-engineering/improve-accuracy-of-ocr-using-image-preprocessing-8df29ec3a033

[i04]: img/1_kG5YMxRZit_b2HZdtnzaCg.png
[i03]: img/1_i0Xv2BnK6SVEQsh0Ex7iJg.png
[i02]: img/1_77OmClHlclhqQ9vGoPcd4w.png "Image Rescaling"
[i01]: img/1_pSUZFdoaiWrJaxQU31ATHg.jpeg