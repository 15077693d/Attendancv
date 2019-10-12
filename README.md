# Attendancv

![](https://github.com/15077693d/Attendancv/blob/master/readme_img/2c.prediction.png?raw=true)

![](https://github.com/15077693d/Attendancv/blob/master/readme_img/3.result.png?raw=true)

[TOC]

# Introduction

- Taking attendane by one image.
- Face detection using MTCNN implementation for TensorFlow by ipazc;
- Face recogition using face embedding and K Nearest Neighbor by ageitgey and scikit-learn;
- This is a object oriented program, you can create your own class for taking attendance;
- Image processing by opencv and matplotlib;
- In the future, I will deploy it by Flask, try different machine learning algorithm for better result;
- Averagers are used for demonstration :)

# 1. Get started
## Required directory and a csv document
In this demo, class name is Avengers.

![](https://github.com/15077693d/Attendancv/blob/master/readme_img/1.createdoc.png?raw=true)

### a. Directory structure

	├── Attendancv
		├── data
			├── < class name >
				├── image
					├── class
					├── individual
				├── < class name >.csv
		├── label_dictionary.py           
		├── model.py
		└── utils.py
### b. CSV file
Puting label names on first row starting by second column, __'time'__ on cell A1.
This csv file will save all attendance record by the program.

![](https://github.com/15077693d/Attendancv/blob/master/readme_img/1.createtable.png?raw=true)

### c. Image directory preparation
Each image on folder ``individual`` corresponding to 1 label. The image name format needs to follow the order of label name on the first row of ``< class name >.csv``.
The number is started from zero.

The format of image name : ``000< label order >_00< label image amount >.jpg``

First individual image of Thanos who is No.13 on ``< class name >.csv``  :  ``0013_000.jpg``
Secord individual image of Thanos who is No.13 on ``< class name >.csv``  :  ``0013_001.jpg``

![](https://github.com/15077693d/Attendancv/blob/master/readme_img/1.createtable_2.png?raw=true)

![](https://github.com/15077693d/Attendancv/blob/master/readme_img/1.createindivdiual.png?raw=true)

Puting the class image on class folder everytime. If you put one image each time, you no need to change image name.
Else, following class image format as ``YYYY-mm-dd HH/mm.jpg``

![](https://github.com/15077693d/Attendancv/blob/master/readme_img/1.createclass.png?raw=true)

# 2. Run the code
Whole procress are three lines, creating class, modelling and prediction.
The code below can be executed on ``label_dictionary.py``.

![](https://github.com/15077693d/Attendancv/blob/master/readme_img/2.runcode.png?raw=true)

## a. Create class

`object` = `Label_Dictionary(< class name >, save_individual_annotated=False)`

The face image name, face vector and loaction of each label are saved to the ``< class name >.json``. it is created after line one was executed.

	├── data
		├── < class name >
			├── image
			├── < class name >.csv
			├── < class name >.json

![](https://github.com/15077693d/Attendancv/blob/master/readme_img/2a.detectindivdualface.png?raw=true)

if `save_individual_annotated = True` in `Label_dictionary( )`, annotated folder will be created and annotated image will save to this folder by program.

![](https://github.com/15077693d/Attendancv/blob/master/readme_img/2a.save_annotated1.png?raw=true)

## b. Modelling

`object.modelling(n_neighbors= 1)`

Executing second line for creating k nearest neighbor(KNN) for face recognition.
You can select `n_neighbors` in fuction `modelling()`;
The KNN model out named as ``< class name> knn < n_neighbors >``.

![](https://github.com/15077693d/Attendancv/blob/master/readme_img/2b.create_dict_model_1.png?raw=true)

## c. Prediction

`object.tick_attendence(img_name = None,save_annotated = True,add_vector = True,n_neighbors = 1)`

If face recognition is wrong on the class image, you have three method to do correction. adding annotation, change label tag and delete annotation.

![](https://github.com/15077693d/Attendancv/blob/master/readme_img/2c.prediction_annotation_1.png?raw=true)

### - Adding Annotation
![](https://github.com/15077693d/Attendancv/blob/master/readme_img/2c.prediction_annotation_2.png?raw=true)
### - Change Face Tag
![](https://github.com/15077693d/Attendancv/blob/master/readme_img/2c.prediction_annotation_3.png?raw=true)
### - Delete Face Tag
![](https://github.com/15077693d/Attendancv/blob/master/readme_img/2c.prediction_annotation_4.png?raw=true)

#3 - Output
The new face vectors, locations from class image will add to the `< class name >.json`, also new KNN model is created.
![](https://github.com/15077693d/Attendancv/blob/master/readme_img/2d.prediction_table_model.png?raw=true)

The attendance record is saved in `< class name >.csv`.

![](https://github.com/15077693d/Attendancv/blob/master/readme_img/3.result.png?raw=true)

