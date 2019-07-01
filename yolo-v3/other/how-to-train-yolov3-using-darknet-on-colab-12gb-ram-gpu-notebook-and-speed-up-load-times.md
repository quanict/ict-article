# [How to train YOLOv3 using Darknet on Colab 12GB-RAM GPU notebook and speed up load times][source]

![img-01]

## Turn Google Colab notebook into the tool for your real research projects!

- Would you like to work on some object detection system and you don't have GPU on your computer?
- Did you tried Darknet on your computer and found that your **GPU is not enought** for the YOLO model you wanted to train?
- Would you like to use a full machine learning environment with **12GB-RAM GPU** for free?
- Did you **checked Google Colab** before but you found it **not so handy** to work with for a real project?

In this post I will explain how to take advantage of the 12GB GPU power of the free Google Colaboratory notebooks in a useful way.

## Contents

- Train a **Yolo v3** model using **Darknet** using the Colab **12GB-RAM GPU**.
- Turn Colab notebooks into an effective tool to work on real projects. Dealing with the handicap of a runtime that will **blow up every 12 hours** into the space!
    - Working directly from the files on your computer.
    - Configure your notebook to install everything you need and start training in about a minute (Tested using 550MB dataset).
    - Receive your trained weights directly on your computer during the training. While the notebook is training you can check how it is going using your trained weights in your computer.

> **GET TO THE POINT MODE:**
> - I'm going to intro what are the handicaps on Colab, but `if you want to get to the point`, you know what a Colab notebook is, how it works and some basics about YOLO and darknet, you can access directly to the [Colab notebook for training YOLO using Darknet with tips & tricks to turn Colab notebook into a useful tool][1]
> - If you are not sure, I encourage you to read a little bit more!

## What's Google Colaboratory

Google Colab is a free cloud service for machine learning education and research. It provides a runtime fully configured for deep learning and **free-of-charge access to a robust GPU**

For further information on what's exactly Google Colab you can take a look at this video: [Get started with Google Colaboratory][2]

Colab is the perfect place to do your research, tutorials or projects. Obviously, *not everything can be wonderful*. Colab has some limitations that can make some steps a little bit hard or tedious. In this tutorial I compiled some tips and tricks to mitigate these limitations. Notebooks are not very handy to program in. I'll keep as much work as possible to be done on your computer transparently and leave the notebook the training tasks.

But what are these handicaps?

## Training on Colab notebooks limitations

- The first and main problem is that **Colab runtime is volatile**. Your Virtual Machine (VM) will blow up after 12 hours and will disappear in the space!
    - This means that your VM and all **files are lost after 12 hours**.
    - After 12 hours you'll have to **reconfigure your runtime** in order to start training again. This means, download all the tools, compiling libraries, upload your files and so on and so forth. This can take some time each time we need to start every VM.
- You work with a remote VM. You don't have direct access to the VM filesystem. You have to upload your files in order to be used and download the files created during the training.

## How can we solve these limitations?

Luckily, we have some ways to solve these handicaps. Let's see how.

**Remote filesystem** - Google has included Drive API on the notebooks, making very easy to map your Google Drive as a VM drive. Besides, we will synchro one folder of our computer to Google Drive. That's it! now you'll have direct access to your Colab filesystem. You'll be able to work with your YOLO config files locally and test on the notebook instantly.

![img-02]

**Volatile VM. Files are lost every 12 hours** - Google drive to the rescue again. We'll save our files directly to the mapped drive.

**Reconfiguring entire runtime every time** - Basically, speeding up the process. We can configure the entire runtime to train YOLOv3 model using Darknet in less than a minute and just with one manual interaction.

Let's do that!

## What we need to run YOLO in Darknet
---

To train a YOLO model using darknet we need the following (You don't need to download anything right now!).

- A powerful GPU
- Nvidia CUDA and cuDNN [More info][3]
- Open Source Computer Vision Library (OpenCV) [More info][4]
- Darknet [More info][5]

Happily, Colab notebooks have almost all of them the pre-configured for you! We need configure for ourselves the following:

### To run Darknet

1. Configure Runtime type to use GPU. (Only the first time)
2. Get access to Google Drive and map as network drive
3. Install cuDNN
3. Clone and compile Darknet

### To train YOLO

1. YOLO model configuration file (yolo-...-.cfg)
2. An Image data set
3. Data configuration file (obj.data)
4. The pre-trained weights file (file.weights)

Without further ado, let's get our hands dirty!

## Creating the notebook
---

Notebooks can be shared. I can create a notebook and share with you, but if you open it, you will have your own VM runtime to play with. You can make a copy of the notebook and apply your own code.

### You can acess to the notebook in two different ways

1. You can access to [Colab notebook for training YOLO using Darknet with tips & tricks to turn Colab notebook into a useful tool][6] to follow all the explanations.

2. Clone this [github repo][7] and upload to your Google Drive. From there, you'll be able to access and work on it.

Now, you can go to the notebook and start working there. Anyway, if you find interesting you can continue reading a very basic guide about deep learning, object detection and some terms used in YOLO training.

## Some information about object detection, YOLO, Darknet and some basics about deep learning

![img-03]

Let's see some basic stuff to understand what does mean to train a model and some concepts involved. It's a very shallow and very basic explanation!

### The goal. Computer vision

Humans are amazingly great at detecting and recognizing objects. It seems a very simple task for us, but it's really not. What's harder than this ability is to mimic on computers.

If you show a dog picture to a toddler it will be able to recognize another dog as soon as he/she see it. On the contrary, a computer needs to see thousands of dog pictures in order to be able to recognize it. Even more, the computer can see thousands of dog images in one situation, environment and not be able to recognize dogs in other situations or environments (This is what's called image distributions). Dogs on the woods are not the same that dogs on the beach for a computer.

This is why we need to train our models.

### What means to train a model
One of the approaches to solve this problem is what's called **artificial neural networks**. An ANN is based on a collection of connected units or nodes called artificial neurons [Wikipedia][8]. That is, we have an **input**, an image in our case, we process it through a neural network and we get an **output**, a collection of **bounding boxes** indicating objects n the image in our case.

![img-04]

image source [wikipedia][img-05].

This collection of neurons and nodes is something like a pipeline where different inputs have to finish on different outputs. A cat image as input has to open some stopcocks and a dog has to open some other stopcocks. Oversimplifying, each stopcock will have what's called a **weight** that will decide if for some input it has to be open or closed making the water arriving to the desired result.

> To train a model means to find all the **weights** for every unit of the neural network in order to achieve our desired result for a concrete input.

To train a model we follow these steps:

- Input an image we know what contains.
- Process this through our neural network and get a result.
- Compare the result to what we know it really contains.
- Adjust the weights in order to minimize the difference between what we obtained and we know we had.
- This process is repeated until we get the best weights we can.

Here are some terms involved in this process. To know what these terms are, is necessary for start training your own object detection model.

- **Weights** - The parameters we need to train in our neural network in order to, given an input, having a desired output.
- **Train** - Apply a defined number of images to our model to get and correct the best weights for our purpose.
- **Data Set** - A pairs of images and labels.
- **Labels** - The annotations we'll prepare for each image to indicate to the model what it has to be found in each image.
- **Train Data Set** - The data set of images we'll use to train our NN.
- **Test Data Set** - The data set we'll use to validate our NN.
- **Classes** - The number of objects we want to detect in our model.

Now, let's back to YOLO.

## What's YOLO and Darknet?

[YOLO][9], acronym of You Only Look Once is a state-of-the-art, real-time object detection system created by [Joseph Redmon][10]. On a Pascal Titan X it processes images at 30 FPS and has a mAP of 57.9% on COCO test-dev.

[Darknet][5], also created by Joseph Redmon is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

What you have to prepare before you start training the model, is the dataset. Here, you have a cheat sheet of all the config files and dataset configuration you need in order to train your YOLO model

![img-06]

You can download the image [here][img-07] to expand.

Thank you for reading! I encourage you to get in touch with me for further help, questions or suggestions. I'll appreciate these!

## SOURCES

Other sources

- YOLO original web site [Joseph Redmon Page][9]
- AlexeyAB darknet [repo github][11]
- The Ivan Goncharov [notebook][12] inspired me to try Google Colab and end up creating this notebook.

[source]: http://blog.ibanyez.info/blogs/coding/20190410-run-a-google-colab-notebook-to-train-yolov3-using-darknet-in/
[1]: https://colab.research.google.com/drive/1lTGZsfMaGUpBG4inDIQwIJVW476ibXk_
[2]: https://www.youtube.com/watch?v=inN8seMm7UI
[3]: https://developer.nvidia.com/cuda-zone
[4]: https://opencv.org/
[5]: https://pjreddie.com/darknet/
[6]: https://colab.research.google.com/drive/1lTGZsfMaGUpBG4inDIQwIJVW476ibXk_
[7]: https://github.com/kriyeng/yolo-on-colab-notebook
[8]: https://en.wikipedia.org/wiki/Artificial_neural_network
[9]: https://pjreddie.com/darknet/yolo/
[10]: https://twitter.com/pjreddie
[11]: https://github.com/AlexeyAB/darknet/
[12]: https://github.com/ivangrov/YOLOv3-GoogleColab/blob/master/YOLOv3_GoogleColab.ipynb


[img-01]: img/B20190410T000000071.jpg
[img-02]: img/B20190408T000000060.jpg
[img-03]: img/B20190410T000000073.jpg
[img-04]: img/B20190410T000000074.png
[img-05]: https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/MultiLayerNeuralNetworkBigger_english.png/800px-MultiLayerNeuralNetworkBigger_english.png
[img-06]: img/B20190410T000000072.png
[img-07]: http://blog.ibanyez.info/download/B20190410T000000072.png