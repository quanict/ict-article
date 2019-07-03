# [Build a Handwritten Text Recognition System using TensorFlow][source]

![img-01]

Offline Handwritten Text Recognition (HTR) systems transcribe text contained in scanned images into digital text, an example is shown in Fig. 1. We will build a Neural Network (NN) which is trained on word-images from the IAM dataset. As the input layer (and therefore also all the other layers) can be kept small for word-images, NN-training is feasible on the CPU (of course, a GPU would be better). This implementation is the bare minimum that is needed for HTR using TF.

![img-02]

## Get code and data

1. You need Python 3, TensorFlow 1.3, numpy and OpenCV installed
2. Get the implementation from https://github.com/githubharald/SimpleHTR
3. Further instructions (how to get the IAM dataset, command line parameters, …) can be found in the README

## Model Overview

We use a NN for our task. It consists of convolutional NN (CNN) layers, recurrent NN (RNN) layers and a final Connectionist Temporal Classification (CTC) layer. Fig. 2 shows an overview of our HTR system.

![img-03]

We can also view the NN in a more formal way as a function (see Eq. 1) which maps an image (or matrix) M of size W×H to a character sequence (c1, c2, …) with a length between 0 and L. As you can see, the text is recognized on character-level, therefore words or texts not contained in the training data can be recognized too (as long as the individual characters get correctly classified).

![img-04]

### Operations

**CNN**: the input image is fed into the CNN layers. These layers are trained to extract relevant features from the image. Each layer consists of three operation. First, the convolution operation, which applies a filter kernel of size 5×5 in the first two layers and 3×3 in the last three layers to the input. Then, the non-linear RELU function is applied. Finally, a pooling layer summarizes image regions and outputs a downsized version of the input. While the image height is downsized by 2 in each layer, feature maps (channels) are added, so that the output feature map (or sequence) has a size of 32×256.

**RNN**: the feature sequence contains 256 features per time-step, the RNN propagates relevant information through this sequence. The popular Long Short-Term Memory (LSTM) implementation of RNNs is used, as it is able to propagate information through longer distances and provides more robust training-characteristics than vanilla RNN. The RNN output sequence is mapped to a matrix of size 32×80. The IAM dataset consists of 79 different characters, further one additional character is needed for the CTC operation (CTC blank label), therefore there are 80 entries for each of the 32 time-steps.

**CTC**: while training the NN, the CTC is given the RNN output matrix and the ground truth text and it computes the loss value. While inferring, the CTC is only given the matrix and it decodes it into the final text. Both the ground truth text and the recognized text can be at most 32 characters long.




[source]: https://towardsdatascience.com/build-a-handwritten-text-recognition-system-using-tensorflow-2326a3487cd5


[img-01]: img/1_ozO04QLClSzCaPgFDi6RYw.jpeg
[img-02]: img/1_6cEKOYqHG27tYwhQVvJqPQ.png "Fig. 1: Image of word (taken from IAM) and its transcription into digital text."
[img-03]: img/1_P4UW-wqOMSpi82KIcq11Pw.png "Fig. 2: Overview of the NN operations (green) and the data flow through the NN (pink)."
[img-04]: img/1_tjy5KJVpbw7tmce2b3bavg.png "Eq. 1: The NN written as a mathematical function which maps an image M to a character sequence (c1, c2, …)."