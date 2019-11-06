# Traffic Sign Classification with Keras and Deep Learning

![img-01]

In this tutorial, you will learn how to train your own traffic sign classifier/recognizer capable of obtaining over 95% accuracy using Keras and Deep Learning.

Last weekend I drove down to Maryland to visit my parents. As I pulled into their driveway I noticed something strange — there was a car I didn’t recognize sitting in my dad’s parking spot.

I parked my car, grabbed my bags out of the trunk, and before I could even get through the front door, my dad came out, excited and enlivened, exclaiming that he had just gotten back from the car dealership and traded in his old car for a brand new 2020 Honda Accord.

Most everyone enjoys getting a new car, but for my dad, who puts a lot of miles on his car each year for work, getting a new car is an especially big deal.

My dad wanted the family to go for a drive and check out the car, so my dad, my mother, and I climbed into the vehicle, the “new car scent” hitting you like bad cologne that you’re ashamed to admit that you like.

> As we drove down the road my mother noticed that the speed limit was automatically showing up on the car’s dashboard — how was that happening?

The answer?

> Traffic sign recognition.

In the 2020 Honda Accord models, a front camera sensor is mounted to the interior of the windshield behind the rearview mirror.

That camera polls frames, looks for signs along the road, and then classifies them.

The recognized traffic sign is then shown on the LCD dashboard as a reminder to the driver.

It’s admittedly a pretty neat feature and the rest of the drive quickly turned from a vehicle test drive into a lecture on how computer vision and deep learning algorithms are used to recognize traffic signs (I’m not sure my parents wanted that lecture but they got it anyway).

When I returned from visiting my parents I decided it would be fun (and educational) to write a tutorial on traffic sign recognition — you can use this code as a starting point for your own traffic sign recognition projects.

## Traffic Sign Classification with Keras and Deep Learning

In the first part of this tutorial, we’ll discuss the concept of traffic sign classification and recognition, including the dataset we’ll be using to train our own custom traffic sign classifier.

From there we’ll review our directory structure for the project.

We’ll then implement `TrafficSignNet`, a Convolutional Neural Network which we’ll train on our dataset.

Given our trained model we’ll evaluate its accuracy on the test data and even learn how to make predictions on new input data as well.

### What is traffic sign classification?

![img-02]
**Figure 1**: Traffic sign recognition consists of object detection: (1) detection/localization and (2) classification. In this blog post we will only focus on classification of traffic signs with Keras and deep learning.

Traffic sign classification is the process of automatically recognizing traffic signs along the road, including speed limit signs, yield signs, merge signs, etc. Being able to automatically recognize traffic signs enables us to build “smarter cars”.

Self-driving cars need traffic sign recognition in order to properly parse and understand the roadway. Similarly, “driver alert” systems inside cars need to understand the roadway around them to help aid and protect drivers.

Traffic sign recognition is just one of the problems that computer vision and deep learning can solve.

### Our traffic sign dataset

![img-03]
**Figure 2**: The German Traffic Sign Recognition Benchmark (GTSRB) dataset will be used for traffic sign classification with Keras and deep learning. ([image source][01])

The dataset we’ll be using to train our own custom traffic sign classifier is the [German Traffic Sign Recognition Benchmark (GTSRB)][02].

The GTSRB dataset consists of **43 traffic sign classes** and **nearly 50,000 images**.

A sample of the dataset can be seen in **Figure 2** above — notice how the traffic signs have been pre-cropped for us, implying that the dataset annotators/creators have manually labeled the signs in the images and extracted the traffic sign Region of Interest (ROI) for us, thereby simplifying the project.

In the real-world, traffic sign recognition is a two-stage proces

1. Localization: Detect and localize where in an input image/frame a traffic sign is.
2. Recognition: Take the localized ROI and actually recognize and classify the traffic sign

Deep learning object detectors can perform localization and recognition in a single forward-pass of the network — if you’re interested in learning more about object detection and traffic sign localization using Faster R-CNNs, Single Shot Detectors (SSDs), and RetinaNet, be sure to refer to my book, [Deep Learning for Computer Vision with Python][03], where I cover the topic in detail.

### Challenges with the GTSRB dataset


-----------------
[sourece]: https://www.pyimagesearch.com/2019/11/04/traffic-sign-classification-with-keras-and-deep-learning/

[img-01]: ../images/traffic_sign_recognition_header.jpg
[img-02]: ../images/traffic_sign_classification_phases.jpg
[img-03]: ../images/traffic_sign_classification_dataset.jpg
[01]: https://steemit.com/programming/@kasperfred/looking-at-german-traffic-signs
[02]: https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
[03]: https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/