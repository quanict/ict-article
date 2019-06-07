# Tool collect dataset

## 1. [labelme]

Labelme is a graphical image annotation tool inspired by [http://labelme.csail.mit.edu](http://labelme.csail.mit.edu).

It is written in Python and uses Qt for its graphical interface.
![alt text][labelme-desc]

## 2. [BBox-Label-Tool]
A simple tool for labeling object bounding boxes in images, implemented with Python Tkinter.
![alt text][BBox-Label-Tool-img]

## 3. [LabelImgTool] [labelImgPlus] [error-use]

    LabelImg is a graphical image annotation tool.
    It is written in Python and uses Qt for its graphical interface.
    The annotation file will be saved as an XML file. The annotation format is PASCAL VOC format, and the format is the same as ImageNet

![alt text][LabelImgTool-img]
![alt text][LabelImgTool-img-2]

## 4. [pyImSegm]

Image segmentation - general superpixel segmentation & center detection & region growing https://borda.github.io/pyImSegm

![alt text][pyImSegm-img]

## 5. [CVAT] (Computer Vision Annotation Tool)

CVAT is completely re-designed and re-implemented version of [Video Annotation Tool from Irvine](http://carlvondrick.com/vatic/), California tool. It is free, online, interactive video and image annotation tool for computer vision. It is being used by our team to annotate million of objects with different properties. Many UI and UX decisions are based on feedbacks from professional data annotation team.

![alt text][CVAT-img]

## 6. [Anno-Mage: A Semi Automatic Image Annotation Tool][semi-auto-image-annotation-tool]

Semi Automatic Image Annotation Toolbox with RetinaNet as the suggesting algorithm. The toolbox suggests 80 class objects from the MS COCO dataset using a pretrained RetinaNet model.
![alt text][Semi-img]


## 7.[Yolo_mark]
GUI for marking bounded boxes of objects in images for training neural network Yolo v3 and v2 https://github.com/AlexeyAB/darknet

Windows & Linux GUI for marking bounded boxes of objects in images for training Yolo v3 and v2




## other tool not install success

2. https://github.com/DewMaple/image_label
3. https://github.com/ryouchinsa/Rectlabel-support


[CVAT]: https://github.com/opencv/cvat
[semi-auto-image-annotation-tool]: https://github.com/virajmavani/semi-auto-image-annotation-tool
[pyImSegm]: https://github.com/Borda/pyImSegm
[LabelImgTool]: https://github.com/lzx1413/LabelImgTool
[BBox-Label-Tool]: (https://github.com/puzzledqs/BBox-Label-Tool)
[labelme]: https://github.com/wkentaro/labelme
[Yolo_mark]: https://github.com/AlexeyAB/Yolo_mark

[CVAT-img]: img/cvat.jpg "CVAT"
[Semi-img]: img/demo.gif
[pyImSegm-img]: img/schema_slic-fts-clf-gc.jpg "pyImSegm"
[LabelImgTool-img]: img/setting_panel.jpg "LabelImgTool"
[LabelImgTool-img-2]: img/parse_label.jpg "LabelImgTool"
[BBox-Label-Tool-img]: img/BBox-Label-Tool-img.png "BBox-Label-Tool"

[labelme-desc]: img/labelme-annotation.jpg "LabelMe description"
