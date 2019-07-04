# [Improve Accuracy of OCR using Image Preprocessing][source]

OCR stands for Optical Character Recognition, the conversion of a document photo or scene photo into machine-encoded text. There are many tools available to implement OCR in your system such as [Tesseract OCR][01] and [Cloud Vision][02]. They use AI and Machine Learning as well as trained custom model. Text Recognition depends on a variety of factors to produce a good quality output. OCR output highly depends on the quality of input image. This is why every OCR engine provides guidelines regarding the quality of input image and its size. These guidelines help OCR engine to produce accurate results.

Here Image Preprocessing comes into play to improve the quality of input image so that the OCR engine gives you an accurate output. Use the following image processing operation to improve the quality of your input image.

![i01]

## Scaling of image :

Image Rescaling is important for image analysis. Mostly OCR engine give an accurate output of the image which has 300 DPI. DPI describes the resolution of the image or in other words, it denotes printed dots per inch.

```py
def set_image_dpi(file_path):
    im = Image.open(file_path)
    length_x, width_y = im.size
    factor = min(1, float(1024.0 / length_x))
    size = int(factor * length_x), int(factor * width_y)
    im_resized = im.resize(size, Image.ANTIALIAS)
    temp_file = tempfile.NamedTemporaryFile(delete=False,   suffix=
'.png')
    temp_filename = temp_file.name
    im_resized.save(temp_filename, dpi=(300, 300))
    
return temp_filename
```


[02]: https://cloud.google.com/vision/
[01]: https://github.com/tesseract-ocr/tesseract
[source]: https://medium.com/cashify-engineering/improve-accuracy-of-ocr-using-image-preprocessing-8df29ec3a033

[i01]: img/1_pSUZFdoaiWrJaxQU31ATHg.jpeg