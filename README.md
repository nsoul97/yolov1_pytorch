# YOLO v1: PyTorch Implementation from Scratch
The following repository implements the paper
[You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) in PyTorch. The code follows
the official implementation of the [Darknet](https://github.com/pjreddie/darknet) repository, which has some slight
differences compared to the paper:

- The most important difference is pertinent to the model's architecture. Specifically, the first Fully Connected Layer
is replaced by a Locally Connected Layer. In the paper, the architecture of the YOLO model is the following:
<p align="center" width="100%"> <img src="https://github.com/nsoul97/yolov1_pytorch/blob/main/assets/model_architecture.png?raw=true"/> </p>

- A Batch Norm operation is used in each convolutional layer, after the convolution operation and before the activation
function.
- The learning rate schedule and the max_batches for which the network was trained.

This repository implements the paper from scratch, including:
+ pretraining with the ImageNet dataset,
+ training with the VOC training set (train/val 2007 + train/val 2012), and
+ evaluation with VOC test set (test 2007)

## Requirements
The package requirements are listed in the `requirements.txt`:

- torch
- torchvision
- matplotlib
- pillow
- tqdm 

## Datasets
### PASCAL VOC 2007 + PASCAL VOC 2012 dataset
To download and prepare the VOC dataset, run the following scripts in the given order:
```
./download_voc.sh
./organize_voc.sh
python3 simplify_voc_targets.py
```

### ImageNet 2012 Challenge Dataset
To download the ImageNet dataset, one must first register in [ImageNet's official website](https://image-net.org/). Following that, download the files:
- ILSVRC2012_img_train.tar
- ILSVRC2012_img_val.tar
- ILSVRC2012_devkit_t12.tar.gz

Afterwards, to prepare the data for torchvision's ImageNet Dataset, run the scipt:
```
./organize_imagenet.sh
```

## Results
The pretrained model achieves a Single-Crop Top5 Accuracy of 89% on the ImageNet's validation set compared to the paper's 88%. To evaluate the pretrained model:

```
python3 pretrain.py
```

To evaluate the performance of the trained YOLO model on the VOC test set and to visualize the model's predictions, run:
```
python3 evaluate.py
python3 plot_predictions.py
```
respectively.

The performance of the detection models in the VOC dataset is compared based on the mean average precision metric.
The mean average precision was measured following the interpolation operation that is described in the paper 
[The PASCAL Visual Object Classes Challenge: A Retrospective](http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham15.pdf).
Furthermore, as instructed for evaluating the performance of a detection model in the PASCAL VOC dataset, the difficult
objects in the PASCAL VOC test set are not considered. Furthermore, the bounding boxes of the difficult objects were
also ignored during training to obtain a better Mean Average Precision.

<div align="center">

|  Implementation  |  Mean Average Precision  |
|:----------------:|:------------------------:|
| this repository  |          63.6%           |
|      paper       |          63.4%           |

</div>


<p align="center" width="100%">
  <img width="100%" src="https://github.com/nsoul97/yolov1_pytorch/blob/main/assets/class_aps.png?raw=true"/>
</p>

## Visualizing the Predictions
The following annotated images belong the PASCAL VOC test set and the percentage value corresponds to the probability 
that there is an object in the bounding box. 

<p align="center" width="100%">
  <img src="https://github.com/nsoul97/yolov1_pytorch/blob/main/assets/annnot_img_6.jpg?raw=true"/>
  <img src="https://github.com/nsoul97/yolov1_pytorch/blob/main/assets/annnot_img_21.jpg?raw=true"/>
  <img src="https://github.com/nsoul97/yolov1_pytorch/blob/main/assets/annnot_img_33.jpg?raw=true"/>
  <img src="https://github.com/nsoul97/yolov1_pytorch/blob/main/assets/annnot_img_59.jpg?raw=true"/>
  <img src="https://github.com/nsoul97/yolov1_pytorch/blob/main/assets/annnot_img_70.jpg?raw=true"/>
  <img src="https://github.com/nsoul97/yolov1_pytorch/blob/main/assets/annnot_img_74.jpg?raw=true"/>
  <img src="https://github.com/nsoul97/yolov1_pytorch/blob/main/assets/annnot_img_107.jpg?raw=true"/>
  <img src="https://github.com/nsoul97/yolov1_pytorch/blob/main/assets/annnot_img_108.jpg?raw=true"/>
  <img src="https://github.com/nsoul97/yolov1_pytorch/blob/main/assets/annnot_img_125.jpg?raw=true"/>
  <img src="https://github.com/nsoul97/yolov1_pytorch/blob/main/assets/annnot_img_140.jpg?raw=true"/>
  <img src="https://github.com/nsoul97/yolov1_pytorch/blob/main/assets/annnot_img_198.jpg?raw=true"/>
  <img src="https://github.com/nsoul97/yolov1_pytorch/blob/main/assets/annnot_img_268.jpg?raw=true"/>
  <img src="https://github.com/nsoul97/yolov1_pytorch/blob/main/assets/annnot_img_272.jpg?raw=true"/>
  <img src="https://github.com/nsoul97/yolov1_pytorch/blob/main/assets/annnot_img_303.jpg?raw=true"/>
  <img src="https://github.com/nsoul97/yolov1_pytorch/blob/main/assets/annnot_img_308.jpg?raw=true"/>
  <img src="https://github.com/nsoul97/yolov1_pytorch/blob/main/assets/annnot_img_314.jpg?raw=true"/>
</p>

## References
- Joseph Redmon, Santosh Kumar Divvala, Ross B. Girshick, & Ali Farhadi (2015). You Only Look Once: Unified, Real-Time Object Detection. CoRR, abs/1506.02640.
- Mark Everingham, S. M. Ali Eslami, Luc Van Gool, Christopher K. I. Williams, John M. Winn, & Andrew Zisserman (2014). The Pascal Visual Object Classes Challenge: A Retrospective. International Journal of Computer Vision, 111, 98-136.
