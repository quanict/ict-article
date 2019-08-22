# YOLO Object Detection with OpenCV and Python

![img-01]

If you have been keeping up with the advancements in the area of object detection, you might have got used to hearing this word 'YOLO'. It has kind of become a buzzword.

## What is YOLO exactly?

YOLO (You Only Look Once) is a method / way to do object detection. It is the algorithm /strategy behind how the code is going to detect objects in the image.

The official implementation of this idea is available through [DarkNet][01] (neural net implementation from the ground up in 'C' from the author). It is available on `github` for people to use.

Earlier detection frameworks, looked at different parts of the image multiple times at different scales and repurposed image classification technique to detect objects. This approach is slow and inefficient.

YOLO takes entirely different approach. It looks at the entire image only once and goes through the network once and detects objects. Hence the name. It is very fast. That’s the reason it has got so popular. 

There are other popular object detection frameworks like **Faster R-CNN** and **SSD** that are also widely used. 

In this post, we are going to look at how to use a pre-trained YOLO model with OpenCV and start detecting objects right away

.. . .

## OpenCV dnn module

DNN (Deep Neural Network) module was initially part of `opencv_contrib` repo. It has been moved to the master branch of opencv repo last year, giving users the ability to run inference on pre-trained deep learning models within OpenCV itself. 

(One thing to note here is, dnn module is not meant be used for training. It’s just for running inference on images/videos.)

Initially only Caffe and Torch models were supported. Over the period support for different frameworks/libraries like TensorFlow is being added. 

Support for YOLO/DarkNet has been added recently. We are going to use the OpenCV dnn module with a pre-trained YOLO model for detecting common objects.

## Let’s get started ..

Enough of talking. Let’s start writing code. (in Python obviously)

```py
# import required packages
import cv2
import argparse
import numpy as np

# handle command line arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help = 'path to input image')
ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
args = ap.parse_args()
```

## Installing dependencies

Following things are needed to execute the code we will be writing.

- Python 3
- Numpy
- OpenCV Python bindings

## Python 3

If you are on Ubuntu, it’s most likely that Python 3 is already installed. Run python3 in terminal to check whether its installed. If its not installed use

```
sudo apt-get install python3
```

For macOS please refer my earlier post on [deep learning setup for macOS][02].

I highly recommend using **Python virtual environment**. Have a look at my earlier post if you need a starting point.

## Numpy

```
pip install numpy
```

This should install `numpy`. Make sure pip is linked to Python 3.x ( pip -V will show this info)

If needed use `pip3`. Use `sudo apt-get install python3-pip` to get `pip3` if not already installed.

## OpenCV-Python

You need to compile OpenCV from source from the master branch on github to get the Python bindings. (recommended)Adrian Rosebrock has written a good blog post on PyImageSearch on this. (Download the source from master branch instead of from archive) If you are feeling overwhelmed by the instructions to get OpenCV Python bindings from source, you can get the unofficial Python package using pip install opencv-python This is not maintained officially by OpenCV.org. It’s a community maintained one. Thanks to the efforts of Olli-Pekka Heinisuo.


----

[02]: http://www.arunponnusamy.com/deep-learning-setup-macos.html
[01]: https://pjreddie.com/darknet/
[img-01]: img/yolo-object-detection.jpg (Image Source: DarkNet github repo)
[source]: https://www.arunponnusamy.com/yolo-object-detection-opencv-python.html