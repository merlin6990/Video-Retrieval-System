# Video-Retrieval-System
In this project we aim to implement a video retrieval system that, given a persian prompt, can find the frames that are contextually similar to the input prompt.

## **Table of Contents**
1. [Chosen Dataset](#Chosen-Dataset)
2. [Data Preparation](#Data-Preparation)
## **Chosen Dataset**
The dataset that was chosen for this project is YouCook2 dataset. Some general information about the dataset is given below and directly extracted from the datasets's website which can be found [here](http://youcook2.eecs.umich.edu/).
The total video time for the dataset is 176 hours with an average length of 5.26 mins for each video. Each video captured is within 10 mins and is recorded by camera devices but not slideshows. All the videos and precomputed feature can be downloaded in the Download page.
Each video contains some number of procedure steps to fulfill a recipe. All the procedure segments are temporal localized in the video with starting time and ending time. The distributions of 1) video duration, 2) number of recipe steps per video, 3) recipe segment duration and 4) number of words per sentence are shown below
## **Data Preparation**
The main challenge in this project is to obtain persian-labled pictures. To overcome this issue, we wrote a script that uses an NMT to translate all the english captions of YouCook2 and adds it as a new field to output.json which holds some crucial information such as the youtube link of the video.
**Rendered Output**:
```python
def hello_world():
    print("Hello, World!")




