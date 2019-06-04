# [Tutorial: Build your own custom real-time object classifier](https://towardsdatascience.com/tutorial-build-an-object-detection-system-using-yolo-9a930513643a)

![img-1](img/1_qnw8wbLlQlHS0YXGH1bzpA.jpeg)

In this tutorial, we will learn how to build a custom real-time object classifier to detect any object of your choice! We will be using [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) and [Selenium](https://www.seleniumhq.org/) to scrape training images from [Shutterstock](https://www.shutterstock.com/), [Amazon’s Mechanical Turk](https://www.mturk.com/) (or [BBox Label Tool](https://github.com/puzzledqs/BBox-Label-Tool)) to label images with bounding boxes, and [YOLOv3](https://pjreddie.com/darknet/yolo/) to train our custom detection model.

### Pre-requisites:

1. Linux
2. CUDA supported GPU (Optional)

### Table of Contents:

- Step 1: Scraping
- Step 2: Labeling
- Step 3: Training

## Step 1: Scraping

In Step 1, we will be using `shutterscrape.py`, a small Python program I wrote, to help us batch download training images from Shutterstock.

### 1. Installing [ChromeDriver](https://sites.google.com/a/chromium.org/chromedriver/downloads)

Open Terminal, download the ChromeDriver zip file, unzip, and run chromedriver:

```command
wget https://chromedriver.storage.googleapis.com/2.43/chromedriver_linux64.zip
unzip chromedriver_linux64.zip
./chromedriver
```

We will use ChromeDriver to simulate human clicks for navigating across Shutterstock’s website.

### 2. Installing dependencies

Open Terminal and install dependencies:

```command
pip install beautifulsoup4
pip install selenium
pip install lxml
```

### 3. Downloading script
Save [shutterscrape.py](https://github.com/chuanenlin/shutterscrape/blob/master/shutterscrape.py) in your working directory.

```python
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from urllib import urlretrieve
import os
import Tkinter, Tkconstants, tkFileDialog
import time

def videoscrape():
    try:
        driver = webdriver.Chrome()
        driver.maximize_window()
        for i in range(1, searchPage + 1):
            url = "https://www.shutterstock.com/video/search/" + searchTerm + "?page=" + str(i)
            driver.get(url)
            print("Page " + str(i))
            for j in range (0, 50):
                while True:
                    container = driver.find_elements_by_xpath("//div[@data-automation='VideoGrid_video_videoClipPreview_" + str(j) + "']")
                    if len(container) != 0:
                        break
                    if len(driver.find_elements_by_xpath("//div[@data-automation='VideoGrid_video_videoClipPreview_" + str(j + 1) + "']")) == 0 and i == searchPage:
                        driver.close()
                        return
                    time.sleep(10)
                    driver.get(url)
                container[0].click()
                while True:
                    wait = WebDriverWait(driver, 60).until(ec.visibility_of_element_located((By.XPATH, "//video[@data-automation='VideoPlayer_video_video']")))
                    video_url = driver.current_url
                    data = driver.execute_script("return document.documentElement.outerHTML")
                    scraper = BeautifulSoup(data, "lxml")
                    video_container = scraper.find_all("video", {"data-automation":"VideoPlayer_video_video"})
                    if len(video_container) != 0:
                        break
                    time.sleep(10)
                    driver.get(video_url)
                video_array = video_container[0].find_all("source")
                video_src = video_array[1].get("src")
                name = video_src.rsplit("/", 1)[-1]
                try:
                    urlretrieve(video_src, os.path.join(scrape_directory, os.path.basename(video_src)))
                    print("Scraped " + name)
                except Exception as e:
                    print(e)
                driver.get(url)
    except Exception as e:
        print(e)

def imagescrape():
    try:
        driver = webdriver.Chrome()
        driver.maximize_window()
        for i in range(1, searchPage + 1):
            url = "https://www.shutterstock.com/search?searchterm=" + searchTerm + "&sort=popular&image_type=all&search_source=base_landing_page&language=en&page=" + str(i)
            driver.get(url)
            data = driver.execute_script("return document.documentElement.outerHTML")
            print("Page " + str(i))
            scraper = BeautifulSoup(data, "lxml")
            img_container = scraper.find_all("div", {"class":"z_c_b"})
            for j in range(0, len(img_container)-1):
                img_array = img_container[j].find_all("img")
                img_src = img_array[0].get("src")
                name = img_src.rsplit("/", 1)[-1]
                try:
                    urlretrieve(img_src, os.path.join(scrape_directory, os.path.basename(img_src)))
                    print("Scraped " + name)
                except Exception as e:
                    print(e)
        driver.close()
    except Exception as e:
        print(e)

print("ShutterScrape v1.1")

#scrape_directory = "C:/Users/[username]/[path]"

while True:
    while True:
        print("Please select a directory to save your scraped files.")
        scrape_directory = tkFileDialog.askdirectory()
        if scrape_directory == None or scrape_directory == "":
            print("You must select a directory to save your scraped files.")
            continue
        break
    while True:
        searchMode = raw_input("Search mode ('v' for video or 'i' for image): ")
        if searchMode != "v" and searchMode != "i":
            print("You must select 'v' for video or 'i' for image.")
            continue
        break
    while True:
        searchCount = input("Number of search terms: ")
        if searchCount < 1:
            print("You must have at least one search term.")
            continue
        elif searchCount == 1:
            searchTerm = raw_input("Search term: ")
        else:
            searchTerm = raw_input("Search term 1: ")
            for i in range (1, searchCount):
                searchTermPart = raw_input("Search term " + str(i + 1) + ": ")
                if searchMode == "v":
                    searchTerm += "-" + searchTermPart
                if searchMode == "i":
                    searchTerm += "+" + searchTermPart
        break
    while True:
        searchPage = input("Number of pages to scrape: ")
        if searchPage < 1:
            print("You must have scrape at least one page.")
            continue
        break
    if searchMode == "v":
        videoscrape()
    if searchMode == "i":
        imagescrape()
    print("Scraping complete.")
    restartScrape = raw_input("Keep scraping? ('y' for yes or 'n' for no) ")
    if restartScrape == "n":
        print("Scraping ended.")
        break
```

### 4. Running shutterscrape.py
1. Open Terminal in the directory of shutterscrape.py and run:

```command
python shutterscrape.py
```

2. Enter `i` for scraping images.
3. Enter the number of search terms. For example, if you want to query Shutterstock for images of “fidget spinners”, enter `2`.
4. Enter your search term(s).
5. Enter the number of pages (pages of image search results on Shutterstock) you want to scrape. Higher number of pages correlate to greater quantity of content with lower keyword precision.
6. Go grab a cup of tea ☕ while waiting… oh wait, it’s already done!

After Step 1, you should have your raw training images ready to be labeled. 👏

## Step 2: Labeling

In Step 2, we will be using `Amazon’s Mechancal Turk`, a marketplace for work that requires human intelligence, to help us label our images. However, this automated process requires you to pay the workers ~$0.02 per image labeled correctly. Hence, you may also label images manually, by hand, using `BBox annotator`. This section assumes that you already have the unlabeled images for training or have completed `Step 1`.

A. Using Mechanical Turk

0. Definitions

**Requester**

A Requester is a company, organization, or person that creates and submits tasks (HITs) to Amazon Mechanical Turk for Workers to perform. As a Requester, you can use a software application to interact with Amazon Mechanical Turk to submit tasks, retrieve results, and perform other automated tasks. You can use the Requester website to check the status of your HITs, and manage your account.

**Human Intelligence Task**

A Human Intelligence Task (HIT) is a task that a Requester submits to Amazon Mechanical Turk for Workers to perform. A HIT represents a single, self-contained task, for example, “Identify the car color in the photo.” Workers can find HITs listed on the Amazon Mechanical Turk website. For more information, go to the Amazon Mechanical Turk website.
Each HIT has a lifetime, specified by the Requester, that determines how long the HIT is available to Workers. A HIT also has an assignment duration, which is the amount of time a Worker has to complete a HIT after accepting it.

**Worker**

A Worker is a person who performs the tasks specified by a Requester in a HIT. Workers use the Amazon Mechanical Turk website to find and accept assignments, enter values into the question form, and submit the results. The Requester specifies how many Workers can work on a task. Amazon Mechanical Turk guarantees that a Worker can work on each task only one time.

**Assignment**

An assignment specifies how many people can submit completed work for your HIT. When a Worker accepts a HIT, Amazon Mechanical Turk creates an assignment to track the work to completion. The assignment belongs exclusively to the Worker and guarantees that the Worker can submit results and be eligible for a reward until the time the HIT or assignment expires.

**Reward**

A reward is the money you, as a Requester, pay Workers for satisfactory work they do on your HITs.

### 1. Setting up your Amazon accounts

- Sign up for an [AWS account](https://aws.amazon.com/)
- Sign up for an [MTurk Requester account](https://requester.mturk.com/)
- [Link your AWS account to your MTurk account](https://requester.mturk.com/developer)
- (Optional) Sign up for a [Sandbox MTurk Requester Account](http://requestersandbox.mturk.com/) to test HIT requests without paying - highly recommended if you are new to MTurk. Remember to link your Sandbox account to your AWS account as well.
- [Set up an IAM user for MTurk](https://docs.aws.amazon.com/AWSMechTurk/latest/AWSMechanicalTurkGettingStartedGuide/SetUp.html#create-iam-user-or-role) and save your access key ID and secret access key for later use.
- Create a bucket on [Amazon S3](https://s3.console.aws.amazon.com/s3/home?region=us-east-1#)
- Upload your photos to the bucket, set your bucket access to public, and save the `bucket name` for later use.

### 2. Installing dependencies
Open Terminal and install dependencies:

```command
pip install boto
pip install pillow
```
### 3. Creating image list
Open Terminal in the directory where you have placed all your photos and create a list of all the filenames of your images:

```command
ls > image.list
```

### 4. Downloading scripts
Save [src.html](https://github.com/chuanenlin/autoturk/blob/master/src.html), [generate.py](https://github.com/chuanenlin/autoturk/blob/master/generate.py), [retrieve.py](https://github.com/chuanenlin/autoturk/blob/master/retrieve.py), [andformat.py](https://github.com/chuanenlin/autoturk/blob/master/format.py) in your working directory.

### 5. Creating HIT template for Bounding Box
You can use MTurk to assign a large variety of HITs. In this section, I will go through how to set up a HIT template for drawing bounding boxes to label images, based on [Kota’s bbox annotator](https://github.com/kyamagu/bbox-annotator).

- Sign in to your [MTurk Requester account](http://requester.mturk.com/).
- Click Create > New Project > Other > Create Project.
- Fill in the required fields and click Design Layout > Source.
I recommend setting Reward per assignment to $0.1, Number of assignments per HIT to 1, Time allotted per assignment to 1, HIT expires in to 7, Auto-approve and pay Workers in to 3, and Require that Workers be Masters to do your HITs to No.
- Paste the code in src.html into the editor and adjust the description to your needs.
- Click Source (again) > Save > Preview and Finish > Finish.
- Click Create > New Batch with an Existing Project > [Your project name] and save the HITType ID and Layout ID for later use.

### 5. Generating HITs

1 .In generate.py
```
- change C:/Users/David/autoturk/image.list in line 9 to the local path of your list of image filenames.
- Change drone-net of https://s3.us-east-2.amazonaws.com/drone-net/ in line 12 to your Amazon S3 bucket name where you've uploaded your images.
- Change [Your_access_key_ID] in line 14 to your access key ID.
- Change [Your_secret_access_key] in line 15 to your secret access key.
- Change drone of LayoutParameter("objects_to_find", "drone") in line 19 to your object name.
- Change [Your_hit_layout] in line 22 to your HIT's Layout ID.
- Change [Your_hit_type] in line 24 to your HIT's HITType ID.
```
2 . (Optional) If you are using Sandbox mode
```
- change mechanicalturk.amazonaws.com in line 16 to http://mechanicalturk.sandbox.amazonaws.com.
- Change https://www.mturk.com/mturk/preview?groupId= in lines 30 and 31 to https://workersandbox.mturk.com/mturk/preview?groupId=.
```
3 .If you are using the normal (non-sandbox) mode, remember [to charge up your account balance](https://requester.mturk.com/account) to pay your hardworking workers!

4 .Open Terminal in the directory of generate.py and run:

```command
python generate.py
```

### 6. Retrieving HITs
---------

