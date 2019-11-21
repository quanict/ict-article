# Tutorial: Build your own custom real-time object classifier

In this tutorial, we will learn how to build a custom real-time object classifier to detect any object of your choice! We will be using BeautifulSoup and Selenium to scrape training images from Shutterstock, Amazon’s Mechanical Turk (or BBox Label Tool) to label images with bounding boxes, and YOLOv3 to train our custom detection model.

### Pre-requisites:

1. Linux
2. CUDA supported GPU (Optional)

### Table of Contents:

- Step 1: Scraping
- Step 2: Labeling
- Step 3: Training

## Step 1: Scraping

--------------
[source]: https://towardsdatascience.com/tutorial-build-an-object-detection-system-using-yolo-9a930513643a