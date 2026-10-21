# 26FDaliBarnaclesChallenge
This repository contains my work on the Computer Vision task of counting barnacles in example images, as used as the Challenge for hiring cycle for new members at the Dali Lab at Dartmouth College.
The files in this repo are organized by
* scripts: contains the runnable scripts that mainly transform the annotations/labels into a workable form, preprocess the image, or zeroshot-apply the cellpose-sam model.
* raw_data: contains all the images provided from the zip file.
* processed_data: contains all outputs from the scripts, organized by which model produced them, or their purpose (inputs), and metrics.json which includes the metrics determined from the zeroshot test runs compared to the annotations.
* final_data: includes the noteworthy overlays my zero-shot approach produced in the end.

My approach was to make a model to automate the counting process based on the image provided. The more accurate the better,
though it is also valuable for the researchers to be able to review the count and the found barnacles, so I want my model
to shade or otherwise mark the found barnacles.

Critical subtasks:
* The Training task: Qualitatively, a model must recognize barnacles. Thus, by pretraining or other means, the basic structure of a barnacle must be detected by the model. This includes telling where a barnacle ends and begins, what is a live barnacle and what isn't. This requires portions of the provided images for training as inputs, and tests on the test images as outputs.
* Our model must be able to shade and count the recognized barnacles as a mask on the image. The recognition must be translated over into a count, and a shaded mask documenting the count
* Pre-Processing: The model should be as accurate as possible, thus the images need to be prepared in resolution. Being provided high resolution images allows for the segmentation of the image itself before reaching the model, likely allowing to optimize training better and increase accuracy rates.
  * segmentation then includes: cutting out the image to be only inside the rectangle that is to be studied
  * segmenting up the rest of the image of what was inside the rectangle, such that reasonable resolution of each barnacle reaches the model, and that reconstructing the whole image from its segments does not double count or miss counts.

I formulated these subtasks already with a prototype in mind. I think dividing up the larger image into subimages will be valuable for potential training, testing, and also as part of the prototype. However, this all presumes a model for this task that is fine-tunable or otherwise trainable exists.
I first looked into modern Computer Vision, according to me having worked briefly in Computer Vision in High School:

https://roboflow.com/blog/best-computer-vision-models

What I took from this mostly was that there were even more major advances in real-time object classification. I mainly took from this that there will be models capable of doing this, and that I can choose significantly slower models that SOTA, since I have no real-time need.
According to the blog and the hints provided in the challenge, I looked into Faster R-CNN:

https://medium.com/@RobuRishabh/understanding-and-implementing-faster-r-cnn-248f7b25ff96
as well as images online.
Noticing is that the object detection and classification is mostly using bounding boxes. Looking into detecting smaller objects, I read online that bounding boxes can cause problems for overlapping objects.

Using Claude for research on this, I came across CellPoseSAM v4.

https://pmc.ncbi.nlm.nih.gov/articles/PMC9718665/
https://www.youtube.com/watch?v=KIdYXgQemcI

The model's ability to generalize well, recognize small, overlapping objects, and being finetunable using the HuggingFace model motivated me to try to make this model work.

To evaluate a zero-shot baseline of the model, or pursue fine-tuning afterwards, I need the annotations in a ML-ready form, which I can derive from the masks.
Once I have these, I can test zero-shot accuracy of Cellpose v4, cpsam_v2, which is the newer base model from June 2026 upgraded for low-contrast areas, and also cpsam since the barnacles do offer a reasonable contrast.
After evaluation, I will need to test whether the next best steps are Human-In-The-Loop approaches, model fine-tuning, or other image processing.

I spent considerable time on this step to figure out what the best way is to figure out the annotations: there were some trade-offs in this step.
* the brush varies strongly in thickness: if I take the brush at face value and count only what is encloses, then singular pixels sometimes are all that remains inside because the brush is so thick. At other times, the brush is so thin that I cannot just half the thickness of the brush without running the risk of connecting the inside of some barnacles with the outside.
  * Setting a minimum size of 10 px for a barnacle captures the vast majority of barnacles but not all, but also avoids some false enclosures. Before I brush up the annotations by hand, this is a conscious simplification.
* because of the brush's width, counting just the interior would make all barnacles appear smaller as a label than in the image. Thus, all labels were resized by half the stroke width.

Having adapted such annotations, I want to move towards zero-shot attempt of cpsam and cpsamv2.
For this, we first write a script on cropping the inside of the square based on the teal-color of the grid.


Having completed this, we write the script to zero shot apply the model and save its outputs.

Results: 
* processed_data/cellpose_zeroshot/['cpsam' or cpsam_v2]/['img1' or 'img2']_overlay
* processed_data/cellpose_zeroshot/metrics.json {group: 'first_Zeroshot'}
Just basing it on zero-shot, we find things to note:
* the model performs reasonably well on image 1: roughly 98% recall, 92% F1 score. The model counts empty/dead barnacles as well, which is to be expected without fine-tuning. Also, from the model's design, it marks the entire barnacle, as was also to be expected, so recall and accuracy have to rather be ranked by how many of the labels in our annotations were counted as distinct labels.
* the model performs much worse on the lower contrast image 2. Recall 21-25%. cpsam_v2 was updated to be better with lower contrast photos, but still fails here.
  * a possible path to pursue might be image pre-processing based on lighting and contrast for such photos

This path shows legitimate promise in being an accurate counter. Before we optimize small rewards like fine-tuning to avoid counting dead barnacles, the higher priority is to accommodate lower contrast images like image 2.

Since fine-tuning will help both models to some degree, I first attempt improving performance on the low contrast photo via pre-processing.
First, passing an average size of a barnacle in pixels in any given image.

Using the diameter parameter of cpsam and cpsam_v2:
https://cellpose.readthedocs.io/en/latest/settings.html

Using the diameter parameter means the model rescales pixel size to a cell (in this case barnacle) being about 30 px: this might increase accuracy, but is said to increase run speed. This does require some input to each picture however, so users would need to provide the average diameter of the barnacle in the image. I considered automating this also based on normalization using the measuring piece provided in the image, but since actual barnacle size also varies significantly between the images, this could be the more robust solution + saves some work for the POC MVP.
I found the average diameter to be around 50 px in image 2 when counting the size of the provided annotations, and 30 px in image 1---Since in application, measurements by eye and a few samples counting pixels would be used, they should also be used here.

Results:
* processed_data/cellpose_zeroshot/['cpsam' or cpsam_v2]/['img1' or 'img2']_['d30' or 'd50']_overlay
* processed_data/cellpose_zeroshot/metrics.json {group: 'Zeroshot_with_Diameter''}

I changed the flag to apply to each image individually. To my surprise, cpsam and cpsam_v2 are doing significantly better on image 2, recall rising to 0.86 and 0.68 respectively, with respective F1 scores .849 and .771. The counts found were 1884/1892 in img 1 and 403/299 in img 2 compared to our ground truth 1694 in img 1 and 396 in img 2.
These results are not yet optimal, but a significant step up compared to the previous attempt on image 2, and that even though the docs had documented size-robustness on cpsam models. Also noteworthy, that cpsam seems to be outperforming cpsam_v2 in our examples. This justifies to continue testing using both models.

Next up, I want to determine a tentative preprocessing scheme for these images:
To note for this is that Cellpose learns via 3 channels; I can provide Cellpose with 3 different versions of the same image to learn from. As such, I can test different versions of what to use on the channels.

I have used CLAHE-L and plain grayscaling, as well as other means like difference of gaussians to test preprocessing.
It seems that most attempts to send composites to the zero shot model make performance overall deteriorate, so the mixture of channels was not helpful.

What I did find was that in CLAHE-L tends towards higher recall, but also higher false positives. RGB currently tends towards better counts overall, but this is because its number of true positives is lower and the number of false positives adds together conveniently on these examples.

Looking at the results from my attempted preprocessing compared to RGB, and seeing what the model marked as a barnacle... I conclude for now that RGB without the tile normalization will be the best candidate for fine-tuning--Sometimes preprocessing can bring good results, it was not the shortcoming here.
