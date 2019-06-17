# [Detecting custom objects in images/video using YOLO with Darkflow][source]

> This is another story of an ongoing series about object detection using **YOLO (You Only Look Once)**, the first one is an introduction about the algorithm and a brief exploration of (some of) the different implementations: [link][2]

## Introduction

The different [YOLO][1] implementations ([Darknet][Darknet], [Darkflow][Darkflow], etc) are amazing tools that can be used to start detecting common objects in images or videos “out of the box”, to do that detection it is only necessary to download and install the system and already trained weights. For instance, in the official [Darknet website][1] we can find the steps to obtain and use the weights trained for [COCO dataset][3] or [VOC PASCAL][4].

![img-1]

There will be cases, however, that the objects we want to detect are simply not part of these popular datasets. In such cases we will need to create our training set and execute our own training.

This tutorial will follow step by step the procedure to create the dataset and run the training using [Darkflow][Darkflow] (a [Darknet][1] translation to run over TensorFlow).

![img-2]

## Step 1: Obtain the images

For this tutorial, we will train Darkflow to **detect text in illustrated maps**.

![img-3]

As you can imagine this kind of images create a problem because they are usually very big in dimensions and contain many instances of the class we will detect (text). That’s why we will use tiled version of the images. To create these tiles we can use the following tool:

https://pinetools.com/split-image

In my case I created a grid of squares of 608 px. by 608 px.

![img-5]
![img-4]
![img-6]

Naturally this step is optional and may not be necessary if your images are consistent in size and they don’t contain a lot of objects.

## Step 2: Annotate the objects



[source]: https://medium.com/coinmonks/detecting-custom-objects-in-images-video-using-yolo-with-darkflow-1ff119fa002f
[1]: https://pjreddie.com/darknet/yolo/
[2]: https://medium.com/@monocasero/object-detection-with-yolo-implementations-and-how-to-use-them-5da928356035
[3]: http://cocodataset.org/#home
[4]: http://host.robots.ox.ac.uk/pascal/VOC/
[Darknet]: https://pjreddie.com/darknet/
[Darkflow]: https://github.com/thtrieu/darkflow

[img-1]: img/1_0sW72yu16QEPEqkGW3iRFw.png (You can download the weights and start detecting horses 🐎)
[img-2]: img/1_25YT3gx25x5z5qOkVHIEgg.png (Real example of a trained YOLO network to detect text in maps
)
[img-3]: img/1_DxXWi3PrFo03U0tqi7MeBQ.jpeg (Property of Kyoto City Tourism Office
)

[img-4]: img/1_e5XNsrIrtasReC-jWiInSg.jpeg (Some instances of the resultant tiles
)

[img-5]:img/1_4kDM2vUbylgIAklMcQHkvQ.jpeg
[img-6]:img/1_jbkmUo8LKK9wNYUKl7yaXQ.jpeg