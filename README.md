# opencv-pipeline
A tool to visualize a series of computer vision operations on an image.

## Overview
I wrote this tool to debug and visualze OpenCV pipelines.  A pipeline consists of a series of OpenCV operations (known in this project as 'filters'.) Each filter accepts a single OpenCV image, performs some operation, and returns the updated OpenCV matrix.

The pipline follows a simple sequence, as follows:
1. It is provided an original source image.
2. It passes the image to the first filter in the pipeline.
3. The first filter returns the modified image to the pipeline
4. The pipeline passes this image on to the next filter, and so on.

 
The app has the following features:
* Visualize both the final image and individual internal images. 
* Dynamically enable and disable individual filters.
* Automatically disable filters that throw exceptions.
* Easy-to-use JSON description of the pipeline.

The following illustrates key parts of the UI.

![image](opencv-overview.gif)


## Installing
Make sure you have Python 3.6+ installed.  Run the following command in your local shell:

```
pip install opencv-python kivy
```

## Running
A quick start is the following command:

```
python3 main.py
```

The following parameters are available.

```  
optional arguments:
   -h, --help                          show this help message and exit
   -i IMAGE, --image IMAGE             path to input image
   -p PIPELINE, --pipeline PIPELINE    path to json describing pipeline
```

## To Do
* Add drag-n-drop (perhaps with a platform more suited to this).
* Add Library pane with list of supported filters.
* Add open/save functionality.