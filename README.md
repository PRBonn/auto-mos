# Auto-MOS: Automatic Labeling to Generate Training Data for Online LiDAR-based Moving Object Segmentation

This repo contains the code for our Auto-MOS, which automatically generates training data for LiDAR-based moving objects segmentation [PDF](http://arxiv.org/pdf/2201.04501).

<img src="pics/framework.png" width="800">

### Table of Contents
1. [Introduction](#Auto-MOS:-Automatic-Labeling-to-Generate-Training-Data-for-Online-LiDAR-based-Moving-Object-Segmentation)
2. [Publication](#Publication)
3. [Logs](#Logs)
4. [Dependencies](#Dependencies)
5. [How to use](#How-to-use)
6. [Application](#Application)
7. [License](#License)

## Publication
If you use our implementation in your academic work, please cite the corresponding paper ([PDF](http://arxiv.org/pdf/2201.04501)):
    
	@article{chen2022ral,
	         author      = {X. Chen and B. Mersch and L. Nunes and R. Marcuzzi and I. Vizzo and J. Behley and C. Stachniss},
	         title       = {{Automatic Labeling to Generate Training Data for Online LiDAR-Based Moving Object Segmentation}},
	         journal     = {IEEE Robotics and Automation Letters (RA-L)},
	         year        = 2022,
	         volume      = 7,
	         number      = 3,
	         pages       = {6107-6114},
	         url         = {http://arxiv.org/pdf/2201.04501},
	         issn        = {2377-3766},
	         doi         = {10.1109/LRA.2022.3166544}
	        }
	     
## Logs

### Version 1.0

Note that, due to copyright and protection of our benchmark, this repo currently only provides the tracking and label generating parts of the proposed method.
For Odometry/LiDAR-SLAM we refer to our SuMa ([link](https://github.com/jbehley/SuMa)), refer dynamic removal to ERASOR ([link](https://github.com/LimHyungTae/ERASOR)), refer instance clustering to HDBSCAN ([link](https://github.com/scikit-learn-contrib/hdbscan)), and refer the LiDAR-MOS network to our LMNet ([link](https://github.com/PRBonn/LiDAR-MOS)).
	        
## Dependencies
Before using our code, you need to install some libraries.

- System dependencies:

  ```bash
  sudo apt-get update 
  sudo apt-get install -y python3-pip wget unzip
  sudo -H pip3 install --upgrade pip
  ```

- Python dependencies (may also work with different versions than mentioned in the requirements file)

  ```bash
  sudo -H pip3 install -r requirements.txt
  ```
	        
## How to run

### Download data and intermediate results

To run the quick demo, please first download the data ([link](https://www.ipb.uni-bonn.de/html/projects/auto-mos/kitti.zip)) extracting it to the `data` folder, and the intermediate instance results ([link](https://www.ipb.uni-bonn.de/html/projects/auto-mos/instances.zip)) extracting it to the `results` folder.

To visualize the final results, you could also directly download the mos results ([link](https://www.ipb.uni-bonn.de/html/projects/auto-mos/mos_predictions.zip)) and extract it into the `results` folder. 

You could also download the data and intermediate results using command lines as follows:

- Download kitti demo dataset:

  ```bash
  wget -P data/ https://www.ipb.uni-bonn.de/html/projects/auto-mos/kitti.zip
  unzip data/kitti.zip -d data
  rm data/kitti.zip
  ```
- Download instance predictions:

  ```bash
  wget -P results/ https://www.ipb.uni-bonn.de/html/projects/auto-mos/instances.zip
  unzip results/instances.zip -d results
  rm results/instances.zip
  ```
- Download final mos predictions:

  ```bash
  wget -P results/ https://www.ipb.uni-bonn.de/html/projects/auto-mos/mos_predictions.zip
  unzip results/mos_predictions.zip -d results
  rm results/mos_predictions.zip
  ```
  	
### Quick run

- To automatic generate the mos labels, one could directly run:
  ```bash
  python3 auto-mos-tracking.py
  ```
  
- To visualize the mos results, one could directly run:
  ```bash
  python3 vis_mos_results.py
  ```
  
  To control the visualizer:
  - press `n`: play next scan,
  - press `b`: play previous scan,
  - press `esc` or `q`: exits.
   
- To visualize the intermediate instance predictions, one could directly run:
  ```bash
  python3 vis_instances.py
  ```
  
  To control the visualizer:
  - press `esc` or `q`: exits.
            
## License
This project is free software made available under the MIT License. For details see the LICENSE file.
