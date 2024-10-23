# cmil

Extracting Conjunctival melanocytic intraepithelial lesions (C-MIL) tissue from Eye tissue and making a Deep Learning Classification Model for Cancer Detection.  


## Folder structure
- data: put all tif images here  
- extracted: tissues extracted from images in data will be saved here  
- extracted/image_name/segmented: roi area segments from tissue images will be saved here  
- workingdir: temporary working directory


## Dependencies
- opencv 4.10.0-dev: https://docs.opencv.org/4.10.0/d7/d9f/tutorial_linux_install.html
- python3: Python 3.8.10
- python3 pip: should already be there with python3
- tiffFile: https://pypi.org/project/tifffile/
- scikit-image: https://scikit-image.org/
