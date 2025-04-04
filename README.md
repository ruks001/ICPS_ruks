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

## 2. Running the repo after setup (Not first time as above)
Activate virtual environment
```commandline
source venv/bin/activate
```

# For object Detection
### 1. Dataset for object detection
Download the dataset from following link and extract all the files and images into "datasets/detection"
```commandline
https://drive.google.com/file/d/118iF3ykmaPogW9f2ssJ9NU7vTZrMYM4m/view?usp=sharing
```

### 2. Run training for object detection
```commandline
python yolo/train.py
```

# For lane Segmentation
## 1. Dataset for segmentation
Download the dataset from following link and place the two folders in the "datasets/segmentation" path.
```commandline
https://drive.google.com/drive/folders/1PDSYJm5d-hpqzxgtxdBCzCegX7tB7x6o?usp=sharing
```
## 2. Run trainig for lane segmentation
```commandline
python segmentation/train.py
```

# For trajectory optimization
## 1. Install the docker engine in your pc first.
## 2. Go inside the  ubuntu20.04 folder
## 3. Run following commands to build and run:
```
docker build -t ubuntu20 .
docker run --rm ubuntu20
```