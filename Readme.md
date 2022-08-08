# Crowd-Density-Estimation

A desktop based application which can detect density of crowd in any public place. 
## Algorithm used

A Multicolumn Convolutional Neural Network (MCNN) is used for this application. It has three different columns, each of them has different sizes of kernel. It helps model to detect 
different sizes of head in crowd. 

## Framework

PyTorch
## Installation

* conda create --prefix ./env python==3.8 -y
* conda activate <<path_to_env_directory>>/env
* pip install -r requirements.txt

## Project Demo

### [Demo](https://www.linkedin.com/posts/arnab-mitra-882756227_connections-computervision-ai-activity-6962065440877543424-XG-4?utm_source=linkedin_share&utm_medium=member_desktop_web)

## Documents

### [HLD](https://drive.google.com/file/d/14aVBiCVZkYYQy5pFEB2UTLxxMSNQEp7A/view?usp=sharing)
### [LLD](https://drive.google.com/file/d/1r1aXXx1fLZPpFODBRinFyfh0QfSAzcJ7/view?usp=sharing)
### [ARCHITECTURE DESIGN](https://drive.google.com/file/d/1fnlLpyLsUnSuOrAGYN36lh-whAL4-B0C/view?usp=sharing)
### [WIREFRAME DOCUMENTATION](https://drive.google.com/file/d/1L95a0mbfXeG5RjSqOPqXMfM_X0g5T0K7/view?usp=sharing)
### [DETAILED PROJECT REPORT](https://drive.google.com/file/d/1d5mknzfzzkN0a04IzntiYXNq9BIoWSjJ/view?usp=sharing)

## Steps

### First go to the project directory and run 'python src/app.py' in command line.

#### Upload any image or video
![Take image](home_page.png)

#### Prediction
![Prediction](prediction.png)

## Dataset 

[SanghaiTech Dataset](https://www.kaggle.com/datasets/tthien/shanghaitech)

## Team
### [Arnab Mitra](https://github.com/Arnab1998-cyber)
### [Subhradipta Paul](https://github.com/SDP010)

## Paper link
[MCNN PAPER](https://people.eecs.berkeley.edu/~yima/psfile/Single-Image-Crowd-Counting.pdf)

