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

Since this particular problem (find text in maps) only requires the detection of a single class, we will use my fork of [BBox-Label-Tool] to annotate the images. It is also easier to install and simpler to use than other alternatives.

(If your problem includes the detection of multiple classes per image, I suggest you use something more sophisticated, like https://github.com/tzutalin/labelImg)
To install [BBox-Label-Tool], we will run:

```
pip install pillow
git clone https://github.com/enriqueav/BBox-Label-Tool.git
cd BBox-Label-Tool
```

The directory structure is organized as

```
BBox-Label-Tool
|
|--main.py # source code for the tool
|--Images/ # directory containing the images to be labeled
|--Labels/ # directory for the labeling results
|--Examples/ # directory for the example bboxes
|--AnnotationsXML/ # directory for the labeling results to be used by Darkflow
```

Inside `Images` , `Labels` , `Examples` and `AnnotationsXML` the tool expects to find numbered directories that will contain subsets of the images and their corresponding annotations. We will create the number `002` to store the images in our dataset

```
mkdir Images/002 Labels/002 Examples/002 AnnotationsXML/002
```

And then we need to copy all the images (or tiles) from the previous step into the directory `Images/002`

```
cp /path/to/your/images/*.jpg Images/002
```

And then we can launch the tool and star annotating!

```
python main.py
```

In the initial GUI under “Image Dir:” we will input “2” (to load subset “002”) and click load:

![img-7]

This will load all the images in the directory `Images/002`, then we can start drawing the bounding boxes containing text.

![img-8]

Once we are finished with an image we click “Next >>” to go to the next one. It is also possible to navigate back, or to move to a specific image number with the navigation toolbar at the bottom.

Now, we need to do this process for each one of the images. This will naturally be the longest and most boring step of the process, but there is nothing we can do about it ¯\_(ツ)_/¯.

Once we are done with the annotations, all the useful .xml files will be stored at `AnnotationsXML/002` this is what we will feed to the Darkflow training!

## Step 3: Installing Darkflow

To download and install the system, the easiest way is to run the following commands (you may need to install tensorflow and numpy beforehand):

```
git clone https://github.com/thtrieu/darkflow.git
cd darkflow
python3 setup.py build_ext --inplace
```

You can find more information in the github page of the project: [github.com][Darkflow]

## Step 4: Modifying configuration files (configuring the network)

There is two possible network configurations that can be used to train, yolo or tiny-yolo. As the name suggest tiny-yolo is a smaller network, that obviously will be faster to process but will suffer from lower accuracy. Under `cfg/` there are configuration files for both of these versions:

```
$ ls -1 cfg/ | grep yolo.cfg
tiny-yolo.cfg
yolo.cfg
```

[source]: https://medium.com/coinmonks/detecting-custom-objects-in-images-video-using-yolo-with-darkflow-1ff119fa002f
[1]: https://pjreddie.com/darknet/yolo/
[2]: https://medium.com/@monocasero/object-detection-with-yolo-implementations-and-how-to-use-them-5da928356035
[3]: http://cocodataset.org/#home
[4]: http://host.robots.ox.ac.uk/pascal/VOC/
[Darknet]: https://pjreddie.com/darknet/
[Darkflow]: https://github.com/thtrieu/darkflow
[BBox-Label-Tool]: https://github.com/enriqueav/BBox-Label-Tool


[img-1]: img/1_0sW72yu16QEPEqkGW3iRFw.png "You can download the weights and start detecting horses 🐎"
[img-2]: img/1_25YT3gx25x5z5qOkVHIEgg.png "Real example of a trained YOLO network to detect text in maps"
[img-3]: img/1_DxXWi3PrFo03U0tqi7MeBQ.jpeg "Property of Kyoto City Tourism Office"
[img-4]: img/1_e5XNsrIrtasReC-jWiInSg.jpeg "Some instances of the resultant tiles"
[img-5]:img/1_4kDM2vUbylgIAklMcQHkvQ.jpeg
[img-6]:img/1_jbkmUo8LKK9wNYUKl7yaXQ.jpeg
[img-7]: img/1_A50qUpI90g3euj_8Kgj32Q.png
[img-8]: img/1_dwytIYHTOqvSmu3ek54-tQ.png
[img-9]: img/1_RLODdaCUjb2palQDRa1m7Q.png