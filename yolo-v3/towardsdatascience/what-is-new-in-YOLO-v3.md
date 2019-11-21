# What’s new in YOLO v3

You only look once, or YOLO, is one of the faster object detection algorithms out there. Though it is no longer the most accurate object detection algorithm, it is a very good choice when you need real-time detection, without loss of too much accuracy.

A few weeks back, the third version of YOLO came out, and this post aims at explaining the changes introduced in YOLO v3. This is not going to be a post explaining what YOLO is from the ground up. I assume you know how YOLO v2 works. If that is not the case, I recommend you to check out the following papers by Joseph Redmon et all, to get a hang of how YOLO works.

1. [YOLO v1][01]
2. [YOLO v2][02]
3. [A nice blog post on YOLO][03]

## YOLO v3: Better, not Faster, Stronger

The official title of YOLO v2 paper seemed if YOLO was a milk-based health drink for kids rather than a object detection algorithm. It was named “YOLO9000: Better, Faster, Stronger”.

For it’s time YOLO 9000 was the fastest, and also one of the most accurate algorithm. However, a couple of years down the line and it’s no longer the most accurate with algorithms like RetinaNet, and SSD outperforming it in terms of accuracy. It still, however, was one of the fastest.

But that speed has been traded off for boosts in accuracy in YOLO v3. While the earlier variant ran on 45 FPS on a Titan X, the current version clocks about 30 FPS. This has to do with the increase in **complexity of underlying architecture called Darknet.**

## Darknet-53



------------------------------------------------------------------------------
[01]: https://pjreddie.com/media/files/papers/yolo_1.pdf
[02]: https://pjreddie.com/media/files/papers/YOLO9000.pdf
[03]: http://christopher5106.github.io/object/detectors/2017/08/10/bounding-box-object-detectors-understanding-yolo.html
[source]: https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b