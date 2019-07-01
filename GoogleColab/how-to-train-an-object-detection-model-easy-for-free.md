# [How to train an object detection model easy for free][source]

![img-01]

In this tutorial, you will learn how to train a custom object detection model easily with TensorFlow object detection API and Google Colab's free GPU.

Annotated images and source code to complete this tutorial are included.

TL:DR; Open [the Colab notebook][01] and start exploring.
Otherwise, let's start with creating the annotated datasets.

## Step 1: Annotate some images

During this step, you will find/take pictures and annotate objects' bounding boxes. It is only necessary if you want to use your images instead of ones comes with [my repository][02].

If your objects are simple ones like nuts and fruits in my example, 20 images can be enough with each image containing multiple objects.

In my case, I use my iPhone to take those photos, each come with 4032 x 3024 resolution, it will overwhelm the model if we use that as direct input to the model. Instead, resize those photos to uniformed size `(800, 600)` can make training and inference faster.

You can use the [resize_images.py][03] script in the repository to resize your images.

First, save your photos, ideally with `jpg` extension to `./data/raw` directory. Then run,

```
python resize_images.py --raw-dir ./data/raw --save-dir ./data/images --ext jpg --target-size "(800, 600)"
```

Resized images will locate in ./data/images/

Next, we split those files into two directories, `./data/images/train` and `./data/images/test`. The model will only use images in the "**train**" directory for training and images in "**test**" directory serve as additional data to evaluate the performance of the model.

![img-02]

Annotate resized images with [labelImg][04], this annotation tool supports both Windows and Linux, it will generate `xml` files inside `./data/images/train` and `./data/images/test` directories.

Tips: use shortcuts (`w`: draw box, `d`: next file, `a`: previous file, etc.) to accelerate the annotation.

## Step 2: prepare `tfrecord` files (source included in [Colab notebook][01])


[source]: https://www.dlology.com/blog/how-to-train-an-object-detection-model-easy-for-free/
[01]: https://colab.research.google.com/github/Tony607/object_detection_demo/blob/master/tensorflow_object_detection_training_colab.ipynb 
[02]: https://github.com/Tony607/object_detection_demo
[03]: https://github.com/Tony607/object_detection_demo/blob/master/resize_images.py
[04]: https://tzutalin.github.io/labelImg/


[img-01]: img/custom_detection.png
[img-02]: img/labelimg.png