# Running YOLO V2 for Real Time Object Detection on Videos/Images Via DarkFlow

![img-01]

With advent of powerful hardware and advances in deep learning algorithms, real time detection of objects in live video is no more a far-sighted task. This quick blog-post shows how to set up real time object detection task on images or videos via Darkflow. We will start by introducing terminologies related to this task followed by installation of packages and illustration of the objective of this blog-post.

## Object Detection Task
Object detection task requires to go beyond classification (i.e. simply classifying the object that appear in an image or a video sequence), and to locate these objects (by creating a bounding box around the object in an image or video sequence).

## You Only Look Once : YOLO
You only look once (YOLO) is a state-of-the-art, real-time object detection system. A single convolutional network simultaneously predicts multiple bounding boxes and class probabilities for those boxes. YOLO trains on full images and directly optimizes detection performance. Here are some of the texts quoted in the initial research paper by the authors of YOLO.

> First, YOLO is extremely fast. Since we frame detection as a 
regression problem we don’t need a complex pipeline. We simply run 
our neural network on a new image at test time to predict detections

> Second, YOLO reasons globally about the image when making predictions
Unlike sliding window and region proposal-based techniques, YOLO sees
the entire image during training and test time so it implicitly encodes
contextual information about classes as well as their appearance. 

> Third, YOLO learns generalizable representations of objects. When 
trained on natural images and tested on art work, YOLO outperforms 
top detection methods like DPM and R-CNN by a wide margin. Since 
YOLO is highly generalizable it is less likely to break down when 
applied to new domains or unexpected inputs.

There are numerous articles, blog-post, video tutorials on YOLO where you can read more about it. You may like to go through Coursera deep learning [course][01] by Andrew Ng to study about YOLO. Also, You may like to read literature from YOLO authors [here](02), [here](03) and [here](04).

### DarkNet

Darknet is an open source custom neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation. You can find the source on [GitHub](05). Originally, YOLO algorithm is implemented in DarkNet framework by Joseph Redmon (author of YOLO). Further, Darknet models had been converted to tensorflow, keras etc to use them in their deep learning tool of choice.

### DarkFlow

It’s a deep learning library which translates darknet to tensorflow and hence the name. You can find the source on [GitHub](06). Once you install darkflow, you can use it in any other python application which is not easily possible using Darknet. One can also train new models on their choice of dataset or object classes. In this blog-post I will be showing how to install Darkflow and set up an real time object detection task on recorded videos.

I am using Nvidia GeForce GTX 1080 GPU with 8 GB of GPU memory.  Having said all this, Let’s start with the installation part.

## Installation : DarkFlow

You may have lot of software applications, libraries and tools already installed. Sometimes, it is really a pain to install an application with lot of dependencies which can disturb the other applications. So, I did the DarkFlow installation in a separate conda environment. The idea is to make python level isolation and to do clean room development without hidden dependencies on the base system. So here it goes.

1. Clone the darkflow github repository:
```
git clone https://github.com/thtrieu/darkflow
```
2. Create conda environment for darkflow installation
```
conda create -n darkflow-env python=3.6
```
3. Activate the new environment
```
source activate darkflow-env
```

4. Install tensorflow or tensorflow-gpu, cython and numpy (If you have GPU, always remember to install tensorflow with gpu support)
```
conda install tensorflow-gpu cython numpy
```

5. Install OpenCV with conda-forge repository. conda-forge is a github organization containing repositories of conda recipies
```
conda config --add channels conda-forge
conda install opencv
```
6. Build the Cython extensions in place.
```
python3 setup.py build_ext --inplace
```
The above steps will setup an environment to run darkflow and perform object detection task on images or videos.

## Running YOLO V2 (command line)

The pre-trained model name is YOLOv2 608×608 which is trained on coco dataset containing 80 objects. So, firstly you need to download the `yolov2.weights` file from [here](07). You can create a bin directory for keeping the weights file. Also, create a yolov2.cfg text file of corresponding model in cfg directory inside darkflow cloned repository. Check [here](08).

1. Running YOLO V2 on a directory containing images
```
./flow –model cfg/yolov2.cfg –load bin/yolov2.weights –imgdir sample_img –gpu 0.9
```
2. Running YOLO V2 on a recorded video
```
./flow –model cfg/yolov2.cfg –load bin/yolov2.weights –demo sample.MP4 –gpu 0.95
````
3. Running YOLO V2 on specific GPU device
```
CUDA_VISIBLE_DEVICES=0 ./flow –model cfg/yolov2.cfg –load bin/yolov2.weights –demo sample_video.mp4 –gpu 0.9 –gpuName /gpu:0
```
4. Running YOLO V2 and saving the output video with bounding box detection
```
CUDA_VISIBLE_DEVICES=0 ./flow –model cfg/yolov2.cfg –load bin/yolov2.weights –demo sample.mp4 –gpu 0.9 –gpuName /gpu:0 –saveVideo
```
This will save the resulting video with bounding boxes by name “video.avi”

5. Running YOLO V2 and displaying the video with bounding boxes at run time

If you want to display the already saved video with bounding boxes on runtime you may have to make little change in code so that cv2 shows the video in camera mode. To do this, you can open help.py in net folder in darkflow. The “file” attribute in def camera() decides whether to show video or not depending upon the camera mode. So set this “file” variable to 0 and cv2 will show the video with bounding box.

```python
elapsed = int()
start = timer()
self.say('Press [ESC] to quit demo')
# Loop through frames
file = 0 
while camera.isOpened():
    elapsed += 1
    _, frame = camera.read()
    if frame is None:
        print ('\nEnd of Video')
        break
    preprocessed = self.framework.preprocess(frame)
    buffer_inp.append(frame)
    buffer_pre.append(preprocessed)
        
    # Only process and imshow when queue is full
    if elapsed % self.FLAGS.queue == 0:
        feed_dict = {self.inp: buffer_pre}
        net_out = self.sess.run(self.out, feed_dict)
        for img, single_out in zip(buffer_inp, net_out):
            postprocessed = self.framework.postprocess(
                single_out, img, False)
            if SaveVideo:
                videoWriter.write(postprocessed)
            if file == 0: #camera window
                cv2.imshow('', postprocessed)
        # Clear Buffers
        buffer_inp = list()
        buffer_pre = list()
```

After modifying the code slightly as shown above, one can run the same command as given in 4 and you will see real time detection with bounding boxes around objects in the recorded video. Bingo !!!!.


## Results

Once you ran YOLO V2 on a “sample_img” directory as shown above (command 1), you will find following detections in “out” directory.

![img-02]
![img-03]
![img-04]
![img-05]
![img-06]
![img-07]
![img-08]
![img-09]

Finally, It’s time to watch real time object detection on recorded videos. I achieved 30 frame rate per second while running YOLO V2 on recorded video with GTX 1080 GPU. The dimensions of the video frame was 600*400.

As there is no support of video in free plan of WordPress, I am posting the YouTube links of videos with bounding box detected.

- [Example 1](09)
- [Example 2](10)
- [Example 3](11)

## End Remarks

Hope it was an easy go through for the readers. I would encourage you to reproduce the results. Also, the intend of the blog-post was to illustrate and set up a real time video detection system. Though the example shown in the blog-post are command line illustrations, one can also extend it by

.1 Using python programming to further build some applications on top of it (you can import darkflow library to perform detection task in python).
2. Using opencv in order to fetch live streams from camera and perform object detection task in real time.
3. Retraining the YOLO based model on their choice of objects.

Most importantly, One would need to understand YOLO algorithm because that would provide more insights about the parameters and working of algorithm. Doing that would help in building practical real world application more accurately. At the end, I would like to add that the computation and processing speed depends on how powerful or high end GPU is being used.

If you liked the post, follow this blog to get updates about upcoming articles. Also, share it so that it can reach out to the readers who can actually gain from this. Please feel free to discuss anything regarding the post. I would love to hear feedback from you.

Happy deep learning 🙂



----

[11]: https://www.youtube.com/watch?v=IaiqgG7rOnY&index=2&list=PL8ltdovcjXqL0nwEc1wuWfrTAwM4l_Pn0
[10]: https://www.youtube.com/watch?v=4Men9HzlBoo&list=PL8ltdovcjXqL0nwEc1wuWfrTAwM4l_Pn0&index=1
[09]: https://www.youtube.com/watch?v=OrSn06qPxHg&index=3&list=PL8ltdovcjXqL0nwEc1wuWfrTAwM4l_Pn0
[img-09]: img/offyc.jpg
[img-08]: img/offyc_orig.jpg
[img-07]: img/sample_person_out.jpg
[img-06]: img/sample_person.jpg
[img-05]: img/sample_horses_out.jpg
[img-04]: img/sample_horses.jpg
[img-03]: img/sample_dog_out.jpg
[img-02]: img/sample_dog.jpg
[08]: https://github.com/pjreddie/darknet/blob/master/cfg/yolov2.cfg
[07]: https://pjreddie.com/media/files/yolov2.weights
[06]: https://github.com/thtrieu/darkflow
[05]: https://github.com/pjreddie/darknet
[04]: https://pjreddie.com/media/files/papers/YOLOv3.pdf
[03]: https://pjreddie.com/media/files/papers/YOLO9000.pdf
[02]: https://pjreddie.com/media/files/papers/yolo_1.pdf
[01]: https://www.coursera.org/specializations/deep-learning
[img-01]: img/sitting_area2.jpg