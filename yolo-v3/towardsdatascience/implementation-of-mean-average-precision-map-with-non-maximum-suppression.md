# Implementation of Mean Average Precision (mAP) with Non-Maximum Suppression (NMS)

You may think that the toughest part is over after writing your CNN object detection model. What about the metrics to measure how well your object detector is doing? The metric to measure objection detection is mAP. To implement the mAP calculation, the work starts from the predictions from the CNN object detection model.

### Non-Maximum Suppression

A CNN object detection model such as Yolov3 or Faster RCNN produces more bounding box (bbox) predictions than is actually needed. The first step is to clean up the predictions by Non-Maximum Suppression.

![img-01]
> ground truth bbox (Blue), predicted bbox (light pink), averaged predicted bbox (red)

The above figure shows an image where the blue rectangles are the ground truth bounding boxes. The light pink rectangles are the predicted bounding boxes that have Objectness, a confidence score that there is an object in the bounding box, of more than 0.5. The red bounding boxes are the final predicted bounding boxes that were averaged from the light pink bounding boxes. The averaging of light pink bounding boxes to red bounding boxes is called Non-Maximum Suppression.

https://github.com/eriklindernoren/PyTorch-YOLOv3 offers a detailed implementation of Non-Maximum Suppression and mAP calculation after predictions from Yolov3, as explained throughout this post. Each Yolov3’s prediction consists of top right bounding box coordinates (x1,y1), bottom left bounding box coordinates (x2,y2), an Objectness confidence (Objectness), and Classification confidences for each class (C1,..,C60 if there are 60 classes that the bounding box content could be classified as). For each image, assume 10654 initial bounding boxes were predicted, only 6 predictions that have Objectness confidence above 0.5 are kept. Of the 6 bounding boxes, bounding boxes that have high overlap with each other (high IOU) and are predicting the same class are averaged together. The detailed explanation of the steps are explained below:

![img-02]

### True Positive Detection

Once final predictions are determined, the predicted bounding boxes could be measured against the ground truth grounding boxes to give rise to mAP to see how well the object detector is doing. To do that, the number of true positives needs to be identified. If a predicted bounding overlapped a ground truth bounding box by an IOU threshold (0.5), it is considered a successful detection and the predicted bounding box is a true positive. If a predicted bounding box overlapped a ground truth by less than the threshold, it is considered an unsuccessful detection and the predicted bounding box is a false positive. The precision and recall can be calculated from the true and false positives as shown:

![img-03]

The detailed implementation is shown below. For each image in a batch, for each predicted bounding box in the image, if the predicted class of the bounding box is not one of the target class in the image, record the bounding box as false positive, else, check the predicted bounding box with all target boxes in the image and get the highest overlap with a target box. If the highest overlap is bigger than a IOU threshold, the target box is considered successfully detected and the predicted bounding box is recorded a true positive. Else the bounding box is recorded as false positive. Stash the successfully detected target box away and continue the loop to check for other predicted bounding box. Returns each prediction’s Objectness , its predicted class, and if it is a true positive. The detailed explanation of the steps are shown below:

![img-04]

### mAP Calculation

The outputs from the above step are used to calculate the mAP. Sort the predictions by descending order of Objectness. Starting from the prediction with the highest Objectness, measure the recall (count of true positive/count of all target boxes globally) and precision (count of true positives/ count of predictions up till this point) after each incremental prediction and plot the Recall versus Precision curve. The area under the curve is the mAP. The area calculated is in rectangles, hence the triangulated portion of the graph is ignored. mAP could be calculated for each class of prediction and then averaged over all class. Thanks https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173 for his detailed explanation on mAP. The plotting of Recall versus Precision curve is shown below:

![img-05]
> dataframe for Recall-Precision graph (left), Recall-Precision graph (right)

The detailed implementation is shown below. Sort the True Positives record, Objectness record and Class record by descending order of Objectness. For each class, following the sort order, find the cumulative true positives and false positives from the True Positive record after each incremental prediction. Find the corresponding Recall and Precision values by dividing the cumulative true positives by number of ground truths and predictions (true positives + false positives) respectively. Calculate area under curve for each class. The detailed explanation of the steps are shown below:

![img-06]

The full implementation of NMS and mAP can be referred from test.py and utils/utils.py in https://github.com/eriklindernoren/PyTorch-YOLOv3.

--------------------------

[img-06]: img/1_5QYoSL15Uat3dd4G3OkK_w.png
[img-05]: img/1_MSQp8fGs-QEHUVM5pz7hdQ.png
[img-04]: img/1__HVzM33lxbZckTwFtPJ1Gw.png
[img-03]: img/1_tuFvcejuyrhpRvoIkClAHg.png
[img-02]: img/1_W5sYLWss03jx74PQEEy4_g.png
[img-01]: img/1_75yL5grKt4WF7ETORLzAnw.png
[source]: https://towardsdatascience.com/implementation-of-mean-average-precision-map-with-non-maximum-suppression-f9311eb92522

