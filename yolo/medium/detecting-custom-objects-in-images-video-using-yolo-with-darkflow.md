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

For this example we will use the full yolo configuration, for that we need to create a copy of this file `yolo.cfg`, that we will need to modify for our problem

```
cp cfg/yolo.cfg cfg/yolo-new.cfg
# modify cfg/yolo-new.cfg
vi cfg/yolo-new.cfg
```

We need to modify two lines:

1. In the last **[convolutional]** section, we need to change the number of `filters` , the formula is `filters=(number of classes + 5)*5` , since we have only one class, we set `filters=30` .
2. Under the section **[region]** there is a line to specify the number of classes (around line 244), change it to `classes=1` or the number of classes you have.

> **NOTE**: There is a series of parameters, specially at the beginning of the file, that are taken directly from Darknet, but do not make a difference in Darkflow. For instance `batch=32` will be ignored by Darkflow, we need to specify the batch size in the command line (with `--batch <batch_size>`), otherwise it will take a default of 16. Other instance of this is the learning rate you need to specify it with `--lr <learning_rate>`.

There is also another file that will be necessary, it is a text file containing the name of the classes, one for each line, since we only have one class, we can create it directly from command line

```
echo "map_text" >> one_label.txt
```

## Step 5: Starting the training

We have come a long way, haven’t we? The good news is that we are ready to run the training.

Just as a reminder, in Step 2 we created the training set, consisting in several image files and their corresponding xml file containing the annotations. They will be stored in these locations (you need to substitute `<path_to_bbox-label-tool>` with the actual path you installed the script.

```
<path_to_bbox-label-tool>/Images/002
<path_to_bbox-label-tool>/AnnotationsXML/002
```

Now, coming back to Darkflow, to start the training we need to run

```
python3 flow --model cfg/yolo-new.cfg \
    --labels one_label.txt  \
    --train --trainer adam \
    --dataset "<path_to_bbox-label-tool>/Images/002" \
    --annotation "<path_to_bbox-label-tool>/AnnotationsXML/002"
```

If you have GPU to train (and you should!), you may add to this command

```
--gpu 1.0
```

Darkflow should then start booting up and loading the images, eventually you should start seeing lines like these, printing the loss of each training step:

```
...
step 1 - loss 227.32052612304688 - moving ave loss 227.3205261230469
step 2 - loss 226.1829376220703 - moving ave loss 227.2067672729492
step 3 - loss 225.60186767578125 - moving ave loss 227.046277313232
step 4 - loss 227.2750701904297 - moving ave loss 227.0691566009522
step 5 - loss 227.2261199951172 - moving ave loss 227.0848529403687
...
```


As you probably know by now, deep learning usually takes a lot of time to train. The time will obviously depend entirely in your hardware, the size of your training set, etc. It may take everything from one hour to several days to give useful results.

By default Darkflow will save a checkpoint every 250 steps, so you can stop the training at any time to take a break and/or validate the current weights. **If you want to restart from the last checkpoint, you just need to add --load -1 to the same command you used to start the training.**

> I recommend you take a look at these terminal tips and tricks to easily monitor the current state of your training, including instantly plotting the loss value at [link][5]

![img-9]

## Step 6: Validating the results

At any point you can stop the training and test the detection in a (hopefully never seen by training) set of images, given you have these images in `<path_to_imgs>` :

```
python3 flow --model cfg/yolo-new.cfg \
    --imgdir <path_to_imgs> \
    --load -1 \
    --labels one_label.txt \
    --gpu 1.0
```

By default it will create a directory called out inside `<path_to_imags>` with the annotated images. For instance these are some of the results after training on my dataset for about a day. Is not perfect, but is pretty reasonable given the size of the training set (not quite big) and the difficulty of the problem.

- Some examples of the trained YOLO.

![img-10]
![img-11]
![img-12]
![img-13]
![img-14]
![img-15]
![img-16]

## Where can I find the model and the weights?

### Update 2018–09–11

The architecture of the model is defined in the .cfg file that we modified during **Step 4**, so we have to be careful to retain it.

Darkflow will store the weights in the same directory as the checkpoint information. By default it will use `<your_git_directory>/darkflow/ckpt` . Four files will be created every checkpoint, and a text file called `checkpoint` will be updated.

![img-17]

According [to this][6], the **.meta** file is where the weights are stored. And [here][7] it says the **.meta**, **.index** and **.data** are files related to TensorFlow.






[source]: https://medium.com/coinmonks/detecting-custom-objects-in-images-video-using-yolo-with-darkflow-1ff119fa002f
[1]: https://pjreddie.com/darknet/yolo/
[2]: https://medium.com/@monocasero/object-detection-with-yolo-implementations-and-how-to-use-them-5da928356035
[3]: http://cocodataset.org/#home
[4]: http://host.robots.ox.ac.uk/pascal/VOC/
[5]: https://medium.com/@monocasero/useful-terminal-tips-and-tricks-for-the-machine-learning-practitioner-6e96b61b2bc2
[6]: https://github.com/thtrieu/darkflow/issues/256
[7]: https://github.com/thtrieu/darkflow/issues/309
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
[img-10]: img/1_9rp5NJFKU9bvXgdDTTA68w.jpeg
[img-11]: img/1_8mcJFSsDLftnPTukQXbukw.jpeg
[img-12]: img/1_j-7qE9hU7j1kymOhLvyyzQ.jpeg
[img-13]: img/1_vk_095cxmJiV8soKGX719A.jpeg
[img-14]: img/1_cebXjtD13bMx4Im1gVnt6g.jpeg
[img-15]: img/1_V1O5xnsN5oSuOeS5-KICoA.jpeg
[img-16]: img/1_0j7kojY1-FMILVBhjhsjaA.jpeg
[img-17]: img/1_9rzTxlvxzuQzoyRSgwwSBg.png