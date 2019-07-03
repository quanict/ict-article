# [Text skew correction with OpenCV and Python][source]

![i01]

Today’s tutorial is a Python implementation of my favorite blog post by [Félix Abecassis][01] on the process of text skew correction (i.e., “deskewing text”) using OpenCV and image processing functions.

Given an image containing a rotated block of text at an unknown angle, we need to correct the text skew by:

1. Detecting the block of text in the image.
2. Computing the angle of the rotated text.
3. Rotating the image to correct for the skew.

We typically apply text skew correction algorithms in the field of automatic document analysis, but the process itself can be applied to other domains as well.

## Text skew correction with OpenCV and Python

The remainder of this blog post will demonstrate how to deskew text using basic image processing operations with Python and OpenCV.

We’ll start by creating a simple dataset that we can use to evaluate our text skew corrector.

We’ll then write Python and OpenCV code to automatically detect and correct the text skew angle in our images.

### Creating a simple dataset

Similar to Félix’s example, I have prepared a small dataset of four images that have been rotated by a given number of degrees:

![i02]

The text block itself is from Chapter 11 of my book, [Practical Python and OpenCV][02], where I’m discussing contours and how to utilize them for image processing and computer vision.

The filenames of the four files follow:

```
$ ls images/
neg_28.png	neg_4.png	pos_24.png	pos_41.png
```

The first part of the filename specifies whether our image has been rotated counter-clockwise (negative) or clockwise (positive).

The second component of the filename is the actual number of degrees the image has been rotated by.

The goal our text skew correction algorithm will be to correctly determine the direction and angle of the rotation, then correct for it.

To see how our text skew correction algorithm is implemented with OpenCV and Python, be sure to read the next section.

### Deskewing text with OpenCV and Python

To get started, open up a new file and name it `correct_skew.py` .

From there, insert the following code:

```python
# import the necessary packages
import numpy as np
import argparse
import cv2
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image file")
args = vars(ap.parse_args())
 
# load the image from disk
image = cv2.imread(args["image"])
```

**Lines 2-4** import our required Python packages. We’ll be using OpenCV via our `cv2`  bindings, so if you don’t already have OpenCV installed on your system, please refer to my list of [OpenCV install tutorials][03] to help you get your system setup and configured.

We then parse our command line arguments on **Lines 7-10**. We only need a single argument here, `--image` , which is the path to our input image.

The image is then loaded from disk on **Line 13**.

Our next step is to isolate the text in the image:

```python
# convert the image to grayscale and flip the foreground
# and background to ensure foreground is now "white" and
# the background is "black"
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)
 
# threshold the image, setting all foreground pixels to
# 255 and all background pixels to 0
thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
```

Our input images contain text that is dark on a light background; however, to apply our text skew correction process, we first need to invert the image (i.e., the text is now light on a dark background — we need the inverse).

When applying computer vision and image processing operations, it’s common for the foreground to be represented as light while the background (the part of the image we are not interested in) is dark.

A thresholding operation (**Lines 23** and **24**) is then applied to binarize the image:

![i03]

Given this thresholded image, we can now compute the minimum rotated bounding box that contains the text regions:

```python
# grab the (x, y) coordinates of all pixel values that
# are greater than zero, then use these coordinates to
# compute a rotated bounding box that contains all
# coordinates
coords = np.column_stack(np.where(thresh > 0))
angle = cv2.minAreaRect(coords)[-1]
 
# the `cv2.minAreaRect` function returns values in the
# range [-90, 0); as the rectangle rotates clockwise the
# returned angle trends to 0 -- in this special case we
# need to add 90 degrees to the angle
if angle < -45:
	angle = -(90 + angle)
 
# otherwise, just take the inverse of the angle to make
# it positive
else:
	angle = -angle
```

**Line 30** finds all (x, y)-coordinates in the `thresh`  image that are part of the foreground.

We pass these coordinates into `cv2.minAreaRect`  which then computes the minimum rotated rectangle that contains the entire text region.

The `cv2.minAreaRect`  function returns angle values in the range [-90, 0). As the rectangle is rotated clockwise the angle value increases towards zero. When zero is reached, the angle is set back to -90 degrees again and the process continues.

> **Note**: For more information on `cv2.minAreaRect` , please see [this excellent explanation][04] by Adam Goodwin.

**Lines 37** and **38** handle if the angle is less than -45 degrees, in which case we need to add 90 degrees to the angle and take the inverse.

Otherwise, **Lines 42** and **43** simply take the inverse of the angle.

Now that we have determined the text skew angle, we need to apply an affine transformation to correct for the skew:

```python
# rotate the image to deskew it
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(image, M, (w, h),
	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
```

**Lines 46 and 47** determine the center (x, y)-coordinate of the image. We pass the center  coordinates and rotation angle into the `cv2.getRotationMatrix2D`  (**Line 48**). This rotation matrix `M`  is then used to perform the actual transformation on **Lines 49 and 50**.

Finally, we display the results to our screen:

```python
# draw the correction angle on the image so we can validate it
cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
# show the output image
print("[INFO] angle: {:.3f}".format(angle))
cv2.imshow("Input", image)
cv2.imshow("Rotated", rotated)
cv2.waitKey(0)
```

**Line 53** draws the `angle`  on our image so we can verify that the output image matches the rotation angle (you would obviously want to remove this line in a document processing pipeline).

**Lines 57-60** handle displaying the output image.

### Skew correction results

To grab the code + example images used inside this blog post, be sure to use the **“Downloads”** section at the bottom of this post.

From there, execute the following command to correct the skew for our `neg_4.png`  image:

```
$ python correct_skew.py --image images/neg_4.png 
[INFO] angle: -4.086
```

![i04]

Here we can see that that input image has a counter-clockwise skew of 4 degrees. Applying our skew correction with OpenCV detects this 4 degree skew and corrects for it.

Here is another example, this time with a counter-clockwise skew of 28 degrees:

```
$ python correct_skew.py --image images/neg_28.png 
[INFO] angle: -28.009
```

![i05]

Again, our skew correction algorithm is able to correct the input image.

This time, let’s try a clockwise skew:

```
$ python correct_skew.py --image images/pos_24.png 
[INFO] angle: 23.974
```

![i06]

And finally a more extreme clockwise skew of 41 degrees:

```
$ python correct_skew.py --image images/pos_41.png 
[INFO] angle: 41.037
```
![i07]

Regardless of skew angle, our algorithm is able to correct for skew in images using OpenCV and Python.

## other resource

```
https://github.com/jhansireddy/AndroidScannerDemo
```
---

[04]: http://stackoverflow.com/questions/15956124/minarearect-angles-unsure-about-the-angle-returned
[03]: https://www.pyimagesearch.com/opencv-tutorials-resources-guides/
[02]: https://www.pyimagesearch.com/practical-python-opencv/
[01]: http://felix.abecassis.me/2011/10/opencv-rotation-deskewing/
[source]: https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/

[i07]: img/text_skew_pos41_results.png "Figure 6: Deskewing text with OpenCV."
[i06]: img/text_skew_pos24_results.png "Figure 5: Correcting for skew in text regions with computer vision."
[i05]: img/text_skew_neg28_results.png "Figure 4: Deskewing images using OpenCV and Python."
[i04]: img/text_skew_neg4_results.png "Figure 3: Applying skew correction using OpenCV and Python."
[i03]: img/text_skew_thresh.jpg "Figure 2: Applying a thresholding operation to binarize our image. Our text is now white on a black background."
[i02]: img/text_skew_inputs.png "Figure 1: Our four example images that we’ll be applying text skew correction to with OpenCV and Python."
[i01]: img/text_skew_pos41_results.png