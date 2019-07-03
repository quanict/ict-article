# [Latest Deep Learning OCR with Keras and Supervisely in 15 minutes][source]

![i01]

Hello world. This tutorial is a gentle introduction to building modern text recognition system using deep learning in 15 minutes. It will teach you the main ideas of how to use [Keras][01] and [Supervisely][02] for this problem. This guide is for anyone who is interested in using Deep Learning for text recognition in images but has no idea where to start.

We are going to consider simple real-world example: number plate recognition. This is a good start point and you can easily customize it for your task. Simple tutorial on how to detect number plates you can find [here][03].

When we dove into this field we faced a lack of materials in the internet. Through long research and reading many papers we have developed an understanding of main principles behind creating effective recognition systems. And we have shared our understanding with community in two small video lectures ([part1][04] and [part2][05]) and explain how it works in plain language. We feel that this content is extremely valuable, because it is impossible to find nice and simple explanation of how to build modern recognition systems. We highly recommended to watch them before you start, because they will give you a lot of intuition behind this topic.

To pass this tutorial without problems, you will need Ubuntu, GPU and Docker.

All sources are available at [github][06]. Source code is located at a [single jupyther notebook][07] with comments and useful visualizations.

![i02]

## Where to get training data?

For this tutorial we have generated artificial dataset of more than 10K images that are very similar to real number plates. The images look like this.

![i03]
![i04]
![i05]
![i06]
![i07]
![i08]
![i09]
![i10]

You can easily get this dataset from [Supervisely][02]. Let us say a few words about it. We at [DeepSystems][08] do a lot of computer vision developments like [self-driving car][09], receipt recognition system, [road defect detection][10] and so on. We as data scientists spend a lot of time to working with training data: creating custom image annotations, merging our data with public datasets, making data augmentations and so on. [Supervisely][02] simplifies the way you work with training data and automate many routine tasks. We believe you’ll find it useful in your everyday work.

The first step is to [register][11] in [Supervisely][02]. The next step is to go to “Import” -> “Datasets library” tab and click to “anpr_ocr” project.

![i11]

After that type name “anpr_ocr” and click “Next” button.

![i12]

Then click “Upload” button. That’s all. Now the project “anpr_ocr” is added to your account.

![i13]

It consists of two datasets: “train” and “test”.

![i14]

If you want to preview images, just click to dataset and you will instantly get into annotation tool. For each image we have a text description that will be used as ground truth to train our system. To view it just click to small icon opposite the selected image (market in red).

![i15]

Now we have to download it in a specific format. To do it just click to “DTL” page and insert this [config][12] to text area. It will look like this.

In the screenshot above you can see the scheme illustrating the export steps. We will not dig into technical details (you can read the [documentation][13] if needed) but try to explain this process below. In our “anpr_ocr” project we have two datasets. “Test” dataset is exported as is(all images will be tagged as “test”). “Train” dataset is splitted to two sets: “train” and “val”. Random 95 percent of images will be tagged as “train”, and the rest 5 percent as “val”.

Now you can click “Start” button and wait two minutes while the system prepare archive to download. Click “DTL” -> “Task status” -> “Three vertical dots” -> “Download” button to get training data (marked in red).

![i16]

## Let’s start our experiment

We prepared all you need in our [git repository][06]. Clone it with the following commands

```
git clone https://github.com/DeepSystems/supervisely-tutorials.git 
cd supervisely-tutorials/anpr_ocr
```

Directory structure will be the following

```
.
├── data
├── docker
│   ├── build.sh
│   ├── Dockerfile
│   └── run.sh
└── src
    ├── architecture.png
    ├── export_config.json
    └── image_ocr.ipynb
```

Put downloaded zip archive into “data” directory and run the command below

```
unzip <archive name>.zip -d .
```

In my case the command was

```
unzip anpr_ocr.zip -d .
```

Now lets build and run docker container with prepared working environment (tensorflow and keras). Just go to “docker” directory and run the following commands

```
./build.sh
./run.sh
```

After that you will be inside the container. Run next command to start Jupyther notebook

```
jupyter notebook
```

In terminal you will see something like this

![i17]

You have to copy selected link and paste it into web browser. Notice, that your link will be slightly different from mine.

The last step is to run whole “image_ocr.ipynb” notebook. Click “Cell” -> “Run all”.

Notebook consists of few main parts: data loading and visualisation, model training, model evaluation on test set. On average for this dataset training process takes around 30 minutes.

If everything will be ok, you’ll see the following output

![i18]

As you can see, the predicted string will be the same as ground truth. Thus we have constructed the modern OCR system in one pretty [clear jupyther notebook][07]. In the next chapter of this tutorial we will cover and explain all main principles of how it works.

## How it works

As for us, the understanding of neural network architecture is the key. Please, don’t be lazy and take 15 minutes to watch our small and simple [video lecture][14] about high level overview of NN architecture, that was mentioned at the beginning. It will give you general understanding. If you have already done — bravo! :-)

Here i will try to give you short explanation. High level overview is the following

![i19]

Firstly, image is feeded to CNN to extract image features. The next step is to apply [Recurrent Neural Network][15] to these features followed by the special decoding algorithm. This decoding algorithm takes lstm outputs from each time step and produces the final labeling.

Detailed architecture will be the following. FC — fully connected layer, SM — softmax layer.

![i20]

Image has the following shape: height equals to 64, width equals to 128 and num channels equal to three.

As you have seen before we feed this image tensor to CNN feature extractor and it produces tensor with shape 4*8*4. We put image “apple” to the feature tensor so you can understand how to interpret it. Height equals to 4, width equals to 8 (These are spatial dimentions) and num channels equals to 4. Thus we transform input image with 3 channels to 4 channel tensor. In practice number of channels should be much larger, but we constructed small demo network only because everything fit on the slide.

Next we do reshape operation. After that we obtain the sequence of 8 vectors of 16 elements. After that we feed these 8 vectors to the LSTM network and get its output — also the vectors of 16 elements. Then we apply fully connected layer followed by softmax layer and get the vector of 6 elements. This vector contains probability distribution of observing alphabet symbols at each LSTM step.

In practice, the number of CNN output vectors can reach 32, 64 or more. The choice will depend on the specific task. Also in production it is better to use multilayered bidirectional LSTM. But this simple example explains only most important concepts.

But How does decoding algorithm work? On the above diagram we have eight vectors of probabilities at each LSTM time step. Let’s take most probable symbol at each time step. As a result we obtain the string of eight characters — one most probable letter at each time step. Then we have to glue all consecutive repeating characters into one. In our example two “e” letters are glued to single one. Special blank character allows us to split symbols that are repeated in the original labeling. We added blank symbol to the alphabet to teach our neural network to predict blank between such case symbols. Then we remove all blank symbols. Look at the illustration below

![i21]

When we train our network we replace decoding algorithm with CTC Loss layer. It is explained in our second video lecture. Now it is available only in russian, sorry about it. But the good news are: we have english slides and we will publish english version soon.

A bit complex NN architecture is used in our implementation. The architecture is the following, but the main principles are still the same.

![i22]

After the model training we apply it on images from test set and get really high accuracy. We also visualize probability distributions from each RNN step as a matrix. Here is the example.

![i23]

The rows of this matrix are correspond to all alphabet symbols plus “blank”. Columns correspond to RNN steps.

## Conclusion

We are happy to share our experience with community. We believe that video lectures, this tutorial, ready-to-use artificial data and source code will help you get basic intuition and that everyone can build modern OCR system from scratch.

Feel free to ask any questions! Thank you!

> To install nvidia-docker to you host, go to [github][16]


[source]: https://hackernoon.com/latest-deep-learning-ocr-with-keras-and-supervisely-in-15-minutes-34aecd630ed8
[01]: https://keras.io/
[02]: https://supervise.ly/
[03]: https://medium.com/towards-data-science/number-plate-detection-with-supervisely-and-tensorflow-part-1-e84c74d4382c
[04]: https://medium.com/towards-data-science/lecture-on-how-to-build-a-recognition-system-part-1-best-practices-46208e1ae591
[05]: https://goo.gl/KwWR48
[06]: https://github.com/DeepSystems/supervisely-tutorials
[07]: https://github.com/DeepSystems/supervisely-tutorials/blob/master/anpr_ocr/src/image_ocr.ipynb
[08]: https://deepsystems.ai/
[09]: https://deepsystems.ai/solutions/autonomous-driving
[10]: https://deepsystems.ai/solutions/road-defects-detection
[11]: https://app.supervise.ly/signup
[12]: https://github.com/DeepSystems/supervisely-tutorials/blob/master/anpr_ocr/src/export_config.json
[13]: https://docs.supervise.ly/
[14]: https://youtu.be/uVbOckyUemo
[15]: https://medium.com/towards-data-science/lecture-evolution-from-vanilla-rnn-to-gru-lstms-58688f1da83a
[16]: https://github.com/NVIDIA/nvidia-docker
[17]: https://blogs.dropbox.com/tech/2017/04/creating-a-modern-ocr-pipeline-using-computer-vision-and-deep-learning/

[i01]: img/1_IZw3Q79NyaLMGnyF2jSvjw.png
[i02]: img/1_BgDbuTIXEWY-IHCLiQpw2w.png
[i03]: img/1_AISU0Dv2N7ogfXA8dpR2Dg.png
[i04]: img/1_fIhytfdk_M3lGDp5F7kvWA.png
[i05]: img/1_hWs4o_kUQvi45lM7b-nvMQ.png
[i06]: img/1_Q6oWS1TQCEOcShjhHht8jg.png
[i07]: img/1_zNK7HkV2qp2uHKRwcnGhvw.png
[i08]: img/1_l4f9-lAijYYWSMNmXF-OYQ.png
[i09]: img/1_X1XVLranZjC2wyFSprOOCw.png
[i10]: img/1_UD3egYd7cQ0IDHGq3mtQSA.png
[i11]: img/1_1VPtJFPXr5BGvXWYmxODjg.png
[i12]: img/1_iuFoP3MN-qFqtewNFZmhMw.png
[i13]: img/1_dO_iUkxHSoSNEaGyvB0P9g.png
[i14]: img/1_mBw72dKzTjP5dBPpCtdrrg.png
[i15]: img/1_PNdhLOt6aFN29kk0_fCyMw.png
[i16]: img/1_HK4d-muok6Y-Qhx8G5Kp3Q.jpeg
[i17]: img/1_WEsv-tANld1xVapNZNH6FQ.png
[i18]: img/1_-JEPmeYUhscVcURoBns_Hw.png
[i19]: img/1_sdb9_e5LVSJnxivblcFxEg.png
[i20]: img/1_ppxHSM2dKhtH6lOTlKInfg.png
[i21]: img/1_JOmMMj4lxy4quHd4qjkd_g.png
[i22]: img/1_MyHdDOccT2gqrCcNxKeQYA.png
[i23]: img/1_drhzXqNlpRY5VuYt7rhaDw.png