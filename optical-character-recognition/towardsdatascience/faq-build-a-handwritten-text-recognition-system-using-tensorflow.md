# [FAQ: Build a Handwritten Text Recognition System using TensorFlow][source]

![img-01]

This article is a follow-up of the article presenting a [text recognition model implemented using TensorFlow][01].

There were some questions which I want to discuss here. Let’s have a look at the following three ones:

1. How to recognize text in your samples/dataset?
2. How to recognize text in lines/sentences?
3. How to compute a confidence score for the recognized text?

## 1 How to recognize text in your samples/dataset?

The pre-trained model was trained on the IAM-dataset. One sample from IAM is shown in Fig. 1. The model not only learns how to read text, but it also learns how the samples look like. If you look through the IAM word images, you will notice some patterns:

- Images have high contrast
- Words are tightly cropped
- Bold writing style

![img-02]

If you feed an image with a very different style to the model, you might get a bad result. Let’s take the image shown in Fig. 2.

![img-03]

The model recognizes the text “.” in this image. The reason is, as you might guess, that the model has never seen images like this:

- Low-contrast
- Much space around the word
- Lines very thin

Let’s look at two approaches to improve the recognition result.

## 1.1 Pre-process images

We pre-process the input image such that it looks like a sample from IAM (see Fig. 3). First, let’s crop it. The model still recognizes “.”. Then, let’s increase the contrast. Now, the model gives a much better result: “tello”. This is almost correct. If we thicken the lines by applying a morphological operation, the model is finally able to recognize the correct text: “Hello”.

![img-04]

The cropping can be done with a word-segmentation algorithm like the one proposed by [R. Manmatha and N. Srimal][02]. Increasing the contrast and applying the morphological operation is achieved by the following Python code.

```py
import numpy as np
import cv2

# read
img = cv2.imread('in.png', cv2.IMREAD_GRAYSCALE)

# increase contrast
pxmin = np.min(img)
pxmax = np.max(img)
imgContrast = (img - pxmin) / (pxmax - pxmin) * 255

# increase line width
kernel = np.ones((3, 3), np.uint8)
imgMorph = cv2.erode(imgContrast, kernel, iterations = 1)

# write
cv2.imwrite('out.png', imgMorph)
```

## 1.2 Create IAM-compatible dataset and train model

There are situations for which pre-processing is not suitable:

- New data samples include characters not contained in the IAM dataset (e.g. ö, ä, ü, …) which can’t be recognized by the pre-trained model
- Writing style is completely different from IAM (e.g. machine-printed text)
- Background contains noise which can’t be removed by pre-processing (e.g. cross-section paper)

In these cases training the model on the new data makes sense. You need image-text pairs which you have to convert into an IAM-compatible format. The following code shows how to do this conversion. The getNext() method of the DataProvider class returns one sample (text and image) per call. The createIAMCompatibleDataset() function creates the file words.txt and the directory sub, in which all images are put. You have to adapt the getNext() method if you want to convert your dataset (at the moment it simply creates machine-printed text for the provided words to show an example usage).

```py
import os
import numpy as np
import cv2


class DataProvider():
	"this class creates machine-written text for a word list. TODO: change getNext() to return your samples."

	def __init__(self, wordList):
		self.wordList = wordList
		self.idx = 0

	def hasNext(self):
		"are there still samples to process?"
		return self.idx < len(self.wordList)

	def getNext(self):
		"TODO: return a sample from your data as a tuple containing the text and the image"
		img = np.ones((32, 128), np.uint8)*255
		word = self.wordList[self.idx]
		self.idx += 1
		cv2.putText(img, word, (2,20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0), 1, cv2.LINE_AA)
		return (word, img)


def createIAMCompatibleDataset(dataProvider):
	"this function converts the passed dataset to an IAM compatible dataset"

	# create files and directories
	f = open('words.txt', 'w+')
	if not os.path.exists('sub'):
		os.makedirs('sub')
	if not os.path.exists('sub/sub-sub'):
		os.makedirs('sub/sub-sub')

	# go through data and convert it to IAM format
	ctr = 0
	while dataProvider.hasNext():
		sample = dataProvider.getNext()
		
		# write img
		cv2.imwrite('sub/sub-sub/sub-sub-%d.png'%ctr, sample[1])
		
		# write filename, dummy-values and text
		line = 'sub-sub-%d'%ctr + ' X X X X X X X ' + sample[0] + '\n'
		f.write(line)
		
		ctr += 1
		
		
if __name__ == '__main__':
	words = ['some', 'words', 'for', 'which', 'we', 'create', 'text-images']
	dataProvider = DataProvider(words)
	createIAMCompatibleDataset(dataProvider)
```

After the conversion, copy both the file words.txt and the directory sub into the data directory of the SimpleHTR project. Then you can train the model by executing python main.py --train.

## 2 How to recognize text in lines/sentences?

The model is able to input images of size 128×32 and is able to output at most 32 characters. So, it is possible to recognize one or two words. However, it is not possible to detect longer sentences, because of the small input and output size. Again, there are two ways to handle this.

### 2.1 Pre-process images

If the words of the line are easy to segment (large gaps between words, small gaps between characters of a word), then you can use a word-segmentation method like the one proposed by [R. Manmatha and N. Srimal][02] (see Fig. 4 for an example). You can then feed the segmented words into the model.

![img-05]

### 2.2 Extend model to fit complete text-lines

There are situations in which word-segmentation is difficult:

- Punctuation marks are written next to a word and are therefore difficult to segment
- Cursive writing style is also difficult to segment

You can extend the model such that the input is able to fit larger images and the output is able to fit longer character strings. The changes have to be applied to the module Model.py, especially the constants imgSize and maxTextLen have to be adapted.

Table 1 shows an architecture which I used for text-line recognition. It has a larger input image (800x64) and is able to output larger character strings (up to 100 in length). Additionally, it contains more CNN layers (7) and uses batch normalization in two layers. Finally, the LSTM layers were replaced by a MDLSTM layer to also propagate information along the vertical image axis.

![img-06]

Table 1: Abbreviations: average (avg), bidirectional (bidir), vertical (vert), dimension (dim), batch normalization (BN), convolutional layer (Conv).

## 3 How to compute a confidence score for the recognized text?

The easiest way to get the probability of the recognized text is to use the CTC loss function. The loss function takes the character-probability matrix and the text as input and outputs the loss value L. The loss value L is the negative log-likelihood of seeing the given text, i.e. L=-log(P). If we feed the character-probability matrix and the recognized text to the loss function and afterwards undo the log and the minus, we get the probability P of the recognized text: P=exp(-L).

The following code shows how to compute the probability of the recognized text for a toy example.

```py
"""
Compute score for decoded text in a CTC-trained neural network using TensorFlow:
1. decode text with best path decoding (or some other decoder)
2. feed decoded text into loss function
3. loss is negative logarithm of probability

Example data: two time-steps, 2 labels (0, 1) and the blank label (2). 
Decoding results in [0] (i.e. string containing one entry for label 0).
The probability is the sum over all paths yielding [0], these are: [0, 0], [0, 2], [2, 0]
with probability
0.6*0.3 + 0.6*0.6 + 0.3*0.3 = 0.63.

Expected output:
Best path decoding: [0]
Loss: 0.462035
Probability: 0.63
"""

import numpy as np
import tensorflow as tf


# size of input data
batchSize = 1
numClasses = 3
numTimesteps = 2


def createGraph():
	"create computation graph"
	tinputs = tf.placeholder(tf.float32, [numTimesteps, batchSize, numClasses])
	tseqLen = tf.placeholder(tf.int32, [None]) # list of sequence length in batch

	tgroundtruth = tf.SparseTensor(tf.placeholder(tf.int64, shape=[None, 2]) , tf.placeholder(tf.int32, [None]), tf.placeholder(tf.int64, [2]))
	tloss = tf.nn.ctc_loss(tgroundtruth, tinputs, tseqLen)

	tbest = tf.nn.ctc_greedy_decoder(tinputs, tseqLen, merge_repeated=True)

	return (tinputs, tseqLen, tgroundtruth, tloss, tbest)


def getData():
	"get data as logits (softmax not yet applied)"
	seqLen = [numTimesteps]
	inputs = np.log(np.asarray([ [[0.6, 0.1, 0.3]], [[0.3, 0.1, 0.6]] ], np.float32))
	return (inputs, seqLen) 


def toLabelString(decoderOutput):
	"map sparse tensor from decoder to label string"
	decoded = decoderOutput[0][0]
	idxDict = {b:[] for b in range(batchSize)}
	encodedLabels = [[] for i in range(batchSize)]
	for (idxVal, idx2d) in enumerate(decoded.indices):
		value = decoded.values[idxVal]
		batch = idx2d[0]
		encodedLabels[batch].append(value)

	return encodedLabels[0]


def main():
	# initialize
	(tinputs, tseqLen, tgroundtruth, tloss, tbest) = createGraph()
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	# get data
	(inputs, seqLen) = getData()

	# decode with best path decoding (greedy decoder)
	retBest = sess.run(tbest, {tinputs:inputs, tseqLen:seqLen } )
	print('Best path decoding:', toLabelString(retBest))

	# for decoded result, compute loss
	retLoss = sess.run(tloss, {tinputs:inputs, tseqLen:seqLen, tgroundtruth: (retBest[0][0].indices, retBest[0][0].values, retBest[0][0].dense_shape) })
	
	# print loss and probability of decoded result
	print('Loss:', retLoss[0])
	print('Probability:', np.exp(-retLoss[0]))


if __name__ == '__main__':
	main()
```

## Conclusion

Now you know how to convert your data such that the model is able to recognize the text:

- Make the images look IAM-like
- Split text-lines into separate words

If this does not improve the results, you can still train the model from scratch. Further, you can use the loss function to compute a confidence score for the recognized text.

## References

- [Original article][03]
- [Code of text-recognition model][04]
- [Code for word-segmentation][05]



-----------------------------------------------------------------------------
[05]: https://github.com/githubharald/WordSegmentation
[04]: https://github.com/githubharald/SimpleHTR
[03]: https://towardsdatascience.com/2326a3487cd5
[img-06]: img/1_jE6W45nnus4ocmJAz32jrg.png
[img-05]: img/1_dFfCsLDCk651kyiMVLvETw.png "Fig. 4: Word-segmentation."
[02]: https://github.com/githubharald/WordSegmentation
[img-04]: img/1_JrY2Q-RhOm9mqvRNjKTiJg.png "Fig. 3: Pre-processing steps and the recognized text for each of them."
[img-03]: img/1_fU7l9-NJq2xopdQfZc3BUA.png "Fig. 2: Our new sample for which the model recognizes the text “.”."
[img-02]: img/1_okWPvEDUCTR67MnmWX60XQ.png "Fig. 1: A sample from the IAM dataset."
[01]: https://towardsdatascience.com/2326a3487cd5
[img-01]: img/1_wqOD1RNwjmShxKA1_sggjQ.jpeg
[source]: https://towardsdatascience.com/faq-build-a-handwritten-text-recognition-system-using-tensorflow-27648fb18519