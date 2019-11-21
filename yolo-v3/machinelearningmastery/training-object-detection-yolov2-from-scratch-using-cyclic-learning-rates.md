# Training Object Detection (YOLOv2) from scratch using Cyclic Learning Rates

Object detection is the task of identifying all objects in an image along with their class label and bounding boxes. It is a challenging computer vision task which has lately been taken over by deep learning algorithms like Faster-RCNN, SSD, Yolo. This post focuses on the latest Yolo v2 algorithm which is said to be fastest (approx 90 FPS on low res images when run on Titan X) and accurate than SSD, Faster-RCNN on few datasets. I will be discussing how Yolo v2 works and the steps to train. If you would like to dig deeper into the landscape of object detection algorithms you can refer [here][01] and [here][02].

This post assumes that you have a basic understanding of [Convolutional Layers][03], Max pooling, [Batchnorm][04]. If not, I would suggest you to get a brief idea about the topics in the links attached.

## Yolo v2: You Only Look Once

In an image shown below, we need to identify the bounding boxes for the one instance of Person, Tv Monitor and Bicycle.


-------------------

[04]: https://www.coursera.org/learn/deep-neural-network/lecture/81oTm/why-does-batch-norm-work
[03]: https://www.youtube.com/embed/Oqm9vsf_hvU?start=265&end=396
[02]: https://github.com/Nikasa1889/HistoryObjectRecognition/blob/master/HistoryOfObjectRecognition.png
[01]: http://cv-tricks.com/object-detection/faster-r-cnn-yolo-ssd/
[source]: https://towardsdatascience.com/training-object-detection-yolov2-from-scratch-using-cyclic-learning-rates-b3364f7e4755