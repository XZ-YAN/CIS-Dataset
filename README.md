# CIS-Dataset
## Introduction
The Construction Instance Segmentation (CIS) Dataset contains 50k images with more than 100k annotated instances. 

## Construction Object Categories
* 2 types of workers: workers wearing & not wearing safety helmets.
* 1 type of material: precast component (PC).  
* 7 types of machines: PC delivery truck, dump truck, concrete mixer truck, excavator, roller, dozer & wheel loader.
![Categories](https://github.com/XZ-YAN/CIS-Dataset/blob/main/demo/categories.jpg)  

## Download the CIS Dataset
* [Download link](https://www.aliyundrive.com/s/pDJ2C2xyGK3)
* The dataset is compressed for ease of download. To decode the compressed dataset, move the downloaded "dataset" folder into the root directory of this repo and run:    
  `$ python CIS_decoder.py train 5`  
  `$ python CIS_decoder.py val 1`  
  `$ python CIS_decoder.py test 4`    

## Annotation Visualization
* To visualize the dataset annotations, run:  
  `$ python CIS_annotations_visualizer.py`  
* CIS annotation samples:
![Annotations](https://github.com/XZ-YAN/CIS-Dataset/blob/main/demo/samples.jpg)  

## Dataset Statistics
* To visualize the dataset statistics, run:  
  `$ python CIS_statistics_visualizer.py`  
* CIS statistics:  
![Statistics](https://github.com/XZ-YAN/CIS-Dataset/blob/main/demo/statistics.jpg)  

## License
* The CIS Dataset is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/) to promote the open use of the dataset and future improvements.
* Without permission, the CIS Dataset should only be used for non-commercial scientific research purposes.  

## Citing the CIS Dataset
If you find this repo useful in your research, please consider citing: (To be updated)
