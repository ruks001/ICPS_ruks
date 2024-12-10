# ICPS_ruks

## 1. Running the repo for first time
Create Virtual Environment
```commandline
python3 -m venv venv
```
Install dependencies
```commandline
pip install -r requirements.txt
```
Activate the virtual Environment
```commandline
source venv/bin/activate
```
Create directory for dataset and weights
```commandline
mkdir -p datasets/detection
mkdir -p datasets/segmentation
mkdir model_weights
```

# Download Dataset
## Dataset for object detection
Download the dataset from following link and extract all the files and images into "datasets/detection"
```commandline
https://drive.google.com/file/d/118iF3ykmaPogW9f2ssJ9NU7vTZrMYM4m/view?usp=sharing
```

## 2. Running the repo after setup (Not first time as above)
Activate virtual environment
```commandline
source venv/bin/activate
```
## 3. Run training for object detection
```commandline
python yolo/train.py
```