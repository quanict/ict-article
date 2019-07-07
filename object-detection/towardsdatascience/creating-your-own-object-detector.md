# [Creating your own object detector][source]

Creating you own object detector using the Tensorflow object detection API

![img-01]

Object detection is the craft of detecting instances of a certain class, like animals, humans and many more in an image or video.

The Tensorflow Object Detection API makes it easy to detect objects by using pretrained object detection models, as explained in [my last article][01].

In this article, we will go through the process of training your own object detector for whichever objects you like. I chose to create an object detector which can distinguish between four different microcontrollers.

This article is based on a video I made.

[https://youtu.be/HjiBbChYRDw]

## Introduction

In this article, we will go over all the steps needed to create our object detector from gathering the data all the way to testing our newly created object detector.

If you don’t have the Tensorflow Object Detection API installed yet you can watch [my tutorial][02] on it.

The steps needed are:

1. Gathering data
2. Labeling data
3. Generating TFRecords for training
4. Configuring training
5. Training model
6. Exporting inference graph
7. Testing object detector

## Gathering data

Before we can get started creating the object detector we need data, which we can use for training.

To train a robust classifier, we need a lot of pictures which should differ a lot from each other. So they should have different backgrounds, random object, and varying lighting conditions.

You can either take the pictures yourself or you can download them from the internet. For my microcontroller detector, I took about 25 pictures of each individual microcontroller and 25 pictures containing multiple microcontrollers.

![img-03]
![img-02]
![img-04]
![img-05]
![img-06]
![img-07]
![img-08]
![img-09]
Figure 2: Example of collected images

These images are pretty big because they have a high resolution so we want to transform them to a lower scale so the training process is faster.

I wrote a little script that makes it easy to transform the resolution of images.

```py
from PIL import Image
import os
import argparse
def rescale_images(directory, size):
    for img in os.listdir(directory):
        im = Image.open(directory+img)
        im_resized = im.resize(size, Image.ANTIALIAS)
        im_resized.save(directory+img)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rescale images")
    parser.add_argument('-d', '--directory', type=str, required=True, help='Directory containing the images')
    parser.add_argument('-s', '--size', type=int, nargs=2, required=True, metavar=('width', 'height'), help='Image size')
    args = parser.parse_args()
    rescale_images(args.directory, args.size)
```

To use the script we need to save it in the parent directory of the images as something like `transform_image_resolution.py` and then go into the command line and type:

```
python transform_image_resolution.py -d images/ -s 800 600
```

## Labeling data

Now that we have our images we need to move about 80 percent of the images into the `object_detection/images/train` directory and the other 20 percent in the `object_detection/images/test` directory.

In order to label our data, we need some kind of image labeling software. LabelImg is a great tool for labeling images. It’s also freely available on Github and prebuilts can be downloaded easily.

- [LabelImg Github][03]
- [LabelImg download][04]

After downloading and opening LabelImg you can open the training and testing directory using the “Open Dir” button.

![img-10]

To create the bounding box the “Create RectBox” button can be used. After creating the bounding box and annotating the image you need to click save. This process needs to be repeated for all images in the training and testing directory.

## Generating TFRecords for training

With the images labeled, we need to create TFRecords that can be served as input data for training of the object detector. In order to create the TFRecords we will use two scripts from [Dat Tran’s raccoon detector][05]. Namely the xml_to_csv.py and generate_tfrecord.py files.

After downloading both scripts we can first of change the main method in the xml_to_csv file so we can transform the created xml files to csv correctly.

```py
# Old:
def main():
    image_path = os.path.join(os.getcwd(), 'annotations')
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv('raccoon_labels.csv', index=None)
    print('Successfully converted xml to csv.')
# New:
def main():
    for folder in ['train', 'test']:
        image_path = os.path.join(os.getcwd(), ('images/' + folder))
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv(('images/'+folder+'_labels.csv'), index=None)
        print('Successfully converted xml to csv.')
```

Now we can transform our xml files to csvs by opening the command line and typing:

```
python xml_to_csv.py
```

These creates two files in the images directory. One called test_labels.csv and another one called train_labels.csv.

Before we can transform the newly created files to TFRecords we need to change a few lines in the generate_tfrecords.py file.

From:

```py
# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'basketball':
        return 1
    elif row_label == 'shirt':
        return 2
    elif row_label == 'shoe':
        return 3
    else:
        return None
```

To:

```py
def class_text_to_int(row_label):
    if row_label == 'Raspberry_Pi_3':
        return 1
    elif row_label == 'Arduino_Nano':
        return 2
    elif row_label == 'ESP8266':
        return 3
    elif row_label == 'Heltec_ESP32_Lora':
        return 4
    else:
        return None
```

If you are using a different dataset you need to replace the class-names with your own.

Now the TFRecords can be generated by typing:

```
python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record
python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record
```

These two commands generate a train.record and a test.record file which can be used to train our object detector.

## Configuring training

The last thing we need to do before training is to create a label map and a training configuration file.

### Creating a label map

The label map maps an id to a name. We will put it in a folder called training, which is located in the object_detection directory. The labelmap for my detector can be seen below.

```js
item {
    id: 1
    name: 'Raspberry_Pi_3'
}
item {
    id: 2
    name: 'Arduino_Nano'
}
item {
    id: 3
    name: 'ESP8266'
}
item {
    id: 4
    name: 'Heltec_ESP32_Lora'
}
```
The id number of each item should match the id of specified in the generate_tfrecord.py file.

### Creating a training configuration

Now we need to create a training configuration file. Because as my model of choice I will use faster_rcnn_inception, which just like a lot of other models can be downloaded from [this page][06] I will start with a sample config ( faster_rcnn_inception_v2_pets.config ), which can be found in the sample folder.

First of I will copy the file into the training folder and then I will open it using a text editor in order to change a few lines in the config.

Line 9: change the number of classes to number of objects you want to detect (4 in my case)

Line 106: change fine_tune_checkpoint to the path of the model.ckpt file:

```
fine_tune_checkpoint: "C:/Users/Gilbert/Downloads/Other/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"
```

Line 123: change input_path to the path of the train.records file:

```
input_path: "C:/Users/Gilbert/Downloads/Other/models/research/object_detection/train.record"
```

Line 135: change input_path to the path of the test.records file:

```
input_path: "C:/Users/Gilbert/Downloads/Other/models/research/object_detection/test.record"
```

Line 125–137: change label_map_path to the path of the label map:

```
label_map_path: "C:/Users/Gilbert/Downloads/Other/models/research/object_detection/training/labelmap.pbtxt"
```

Line 130: change num_example to the number of images in your test folder.

--------------------------------------------

## Training model

To train the model we will use the train.py file, which is located in the object_detection/legacy folder. We will copy it into the object_detection folder and then we will open a command line and type:

Update: Use the model_main file in the object_detection folder instead

```
python model_main.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
```

If everything was setup correctly the training should begin shortly.

![img-11]

About every 5 minutes the current loss gets logged to Tensorboard. We can open Tensorboard by opening a second command line, navigating to the object_detection folder and typing:

```
tensorboard --logdir=training
```

This will open a webpage at localhost:6006.

![img-12]

You should train the model until it reaches a satisfying loss. The training process can then be terminated by pressing Ctrl+C.

## Exporting inference graph

Now that we have a trained model we need to generate an inference graph, which can be used to run the model. For doing so we need to first of find out the highest saved step number. For this, we need to navigate to the training directory and look for the model.ckpt file with the biggest index.

Then we can create the inference graph by typing the following command in the command line.

```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
```

XXXX represents the highest number.

## Testing object detector

In order to test our newly created object detector, we can use the code from [my last Tensorflow object detection tutorial][07]. We only need to replace the fourth code cell.

From:

```py
# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that are used to add a correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
```

To:

```py
MODEL_NAME = 'inference_graph'
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = 'training/labelmap.pbtxt'
```

Now we can run all the cells and we will see a new window with a camera stream opening.

![img-13]

## Conclusion

The Tensorflow Object Detection API allows you to create your own object detector using transfer learning.

If you liked this article consider subscribing on my [Youtube Channel][08] and following me on social media.

The code covered in this article is available as a [Github Repository][09].

If you have any questions, recommendations or critiques, I can be reached via [Twitter][10] or the comment section.

--------------------------------------------
[10]: https://twitter.com/Tanner__Gilbert
[09]: https://github.com/TannerGilbert/Tutorials/blob/master/Tensorflow%20Object%20Detection/object_detection_with_own_model.ipynb
[08]: https://www.youtube.com/channel/UCBOKpYBjPe2kD8FSvGRhJwA
[img-13]: img/0_rXKokxTVAP7q8-2A.png "Figure 6: Detecting the microcontrollers"
[07]: https://gilberttanner.com/2018/12/30/tensorflow-object-detection-tutorial-2-live-object-detection/
[img-12]: img/0_A48mVQylFStPiNIB.png "Figure 5: Monitoring loss using Tensorboard"
[img-11]: img/0_HHyiCvnExKUO9NdP.png "Figure 4: Training the object detector"
[06]: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
[05]: https://github.com/datitran/raccoon_dataset
[img-10]: img/0_HuSiRTCtOGnLk_rK.png "Figure 3: Labeling the data using the LabelImg tool"
[04]: https://tzutalin.github.io/labelImg/
[03]: https://github.com/tzutalin/labelImg
[img-09]: img/1_EaQwggINUqw6N_sBTUDQRw.jpeg
[img-08]: img/1_DmtON20g-oKU-B6oSyI_Ng.jpeg
[img-07]: img/1_fsThkASZkhqFUHH3m5TbQQ.jpeg
[img-06]: img/1_zSBNAtIVxSbI7f2z4p3b4w.jpeg
[img-05]: img/1_TC9WseCcOXFg0vSN7_brUw.jpeg
[img-04]: img/1_6BAtBgNjSw5zASPddlXgTw.jpeg
[img-03]: img/1_AqKv7Huz_35szke5iXrqrA.jpeg
[img-02]: img/1_BaaZXr2t2-Fch1bmz2JrRQ.jpeg
[02]: https://gilberttanner.com/2018/12/22/tensorflow-object-detection-tutorial-1-installation/
[01]: https://towardsdatascience.com/live-object-detection-26cd50cceffd
[img-01]: img/1_M-tFFhFd_82CJVv2gJ-Hyg.png "Figure 1: Detecting microcontrollers"
[source]: https://towardsdatascience.com/creating-your-own-object-detector-ad69dda69c85?gi=79642b0c4cb7