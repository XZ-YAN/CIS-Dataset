# CIS-Dataset
## Introduction
* The Construction Instance Segmentation version 1 (CISv1) dataset contains 50k images with over 83k annotated instances, as introduced in the [paper](https://doi.org/10.1016/j.autcon.2023.105083).  
* The Construction Instance Segmentation version 2 (CISv2) dataset contains 61k images with over 114k annotated instances.  

## Construction Object Categories
* 2 categories of workers: workers wearing & not wearing safety helmets.
* 1 categories of materials: precast components (PCs).  
* 7 categories of machines: PC delivery trucks, dump trucks, concrete mixer trucks, excavators, rollers, dozers & wheel loaders.
![Categories](https://github.com/XZ-YAN/CIS-Dataset/blob/main/demo/categories.jpg)  

## Download the CIS Dataset
* CISv1: [Download link](https://www.alipan.com/s/Ewz8npjTiC2)
* CISv2: [Download link](https://www.alipan.com/s/zPCWUKibWGj)
* Note: If the Cloud Drive is not accessible in your region, please use a VPN.  
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
* The CIS dataset is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/) to promote the open use of the dataset and future improvements.
* Without permission, the CIS dataset should only be used for non-commercial scientific research purposes.  

## Citing the CIS Dataset
If you find this repo useful in your research, please consider citing:  
* Xuzhong Yan, Hong Zhang, Yefei Wu, Chen Lin, Shengwei Liu, Construction Instance Segmentation (CIS) Dataset for Deep Learning-Based Computer Vision, Automation in Construction. 156 (2023) 105083, https://doi.org/10.1016/j.autcon.2023.105083.
