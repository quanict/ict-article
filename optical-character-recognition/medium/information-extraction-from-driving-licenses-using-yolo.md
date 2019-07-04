# [Information Extraction from Driving Licenses using YOLO][source]

![img-01]

I was working on a project to **extract text information** from driving licenses of different states from all over India. To do this I broke down the problem into four different stages:

- Creating a dataset of driving licenses from the DL’s available on the internet.
- Identifying regions of interest from the acquired images and making bounding boxes around the areas of text in the DL’s.
- Train YOLO to identify such regions.
- Crop the regions identified by YOLO.
-  OCR on the identified region of interests.

[YOLO][01]- You Only Look Once is a state of the art object detection system capable of **object detection** + **object classification** + **multiple object detection** all at the same time and that also in real time.

## Creating a dataset-

After downloading enough number of driving licenses we need to draw bounding boxes around our areas of interests(text ) using LabelImg. To use labelimg clone the git from [here][02]. Enter the “labelImg-master” in your terminal and run- python labelimg.py. It will open a nice and easy to use application for bounding boxes creation.

You need to make sure that the name of annotation files being created is the same as the image files, and the annotation files are in a different folder as to the image files and in the same sequence arranged as the image files.

## Training YOLO-

Follow the given steps:

- Clone

```
https://github.com/thtrieu/darkflow
```

- Create conda virtual env for the project

```
conda create -n NAME python=3.6
```

- Access new environment

```
source activate NAME
```

- Install dependencies (no prob if they are already installed, conda will just skip them)
```
conda install tensorflow cython numpy
```

- Add the repo with the magic opencv version
```
conda config --add channels conda-forge
```

- Install magic opencv
```
conda install opencv
```

- With a **clean** cloned repo run the setup inside the darkflow folder
```
python3 setup.py build_ext --inplace
```

- [Download][03] both CFG and and WEIGHTS files of the model you’ll use and place them in their respective folders (create the `bin/` folder for weights). If they don’t match you’ll get errors.

Everything should work now!!!

You could download the weights from here also-[weights][04].

Visit this-[darkflow][05] once atleast to know more about how to work with yolo.

After following the above steps enter the darkflow folder and run:

```
flow —h
```

If it runs successfully then everything is installed perfectly and we are good to go.

Before starting the training do give a reading to the README.md inside the darkflow folder and do as instructed in it.

After completing all the requirements we are now ready for training-

From the terminal give the command-

```
python flow --model cfg/tiny-yolo-voc1.cfg  --train --dataset “path_to_the_images_folder" --annotation    “path_to_the_annotations_folder"  
```

It will start the training-

![img-02]

After training our model we can use the following code to read the text and see the ROI’s.

And we are done.

Github- https://github.com/vaibhavska/extracting_text_information_using_YOLO

Please share any queries or if you are facing any obstructions anywhere and i would be happy to help.

-----------------------------------------------

[img-02]: img/1_6xFBiXKRcBPemneiDIVrig.png
[05]: https://github.com/thtrieu/darkflow
[04]: https://drive.google.com/drive/folders/0B1tW_VtY7onidEwyQ2FtQVplWEU
[03]: https://pjreddie.com/darknet/yolo/
[02]: https://github.com/tzutalin/labelImg
[01]: https://pjreddie.com/darknet/yolo/
[img-01]: img/1_bSLNlG7crv-p-m4LVYYk3Q.png
[source]: https://medium.com/@vaibhavshukla182/information-extraction-from-driving-licenses-using-yolo-f6178e81967d