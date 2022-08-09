#!/usr/bin/env python3
# Developed by Xieyuanli Chen
# Brief: visualizer based on open3D for moving object segmentation
# This file is covered by the LICENSE file in the root of this project.

import sys
import yaml
from tracking_utils.utils import *
import open3d as o3d
import pynput.keyboard as keyboard


class vis_mos_results:
  """ LiDAR moving object segmentation results (LiDAR-MOS) visualizer
  Keyboard navigation:
    n: play next
    b: play back
    esc or q: exits
  """
  def __init__(self, config):
    # specify paths
    seq = str(config['seq']).zfill(2)
    dataset_root = config['dataset_root']
    prediction_root = config['prediction_root']
    
    # specify folders
    scan_folder = os.path.join(dataset_root, 'sequences', seq, 'velodyne')
    gt_folder = os.path.join(dataset_root, 'sequences', seq, 'labels')
    prediction_folder = os.path.join(prediction_root, 'sequences', seq, 'predictions')
    
    # load files
    self.scan_files = load_files(scan_folder)
    self.gt_paths = load_files(gt_folder)
    self.predictions_files = load_files(prediction_folder)
    
    # init frame
    self.current_points = load_vertex(self.scan_files[0])[:, :3]
    self.current_preds, _ = load_labels(self.predictions_files[0])
    self.current_gt, _ = load_labels(self.gt_paths[0])
    
    # init visualizer
    self.vis = o3d.visualization.Visualizer()
    self.vis.create_window()
  
    self.pcd = o3d.geometry.PointCloud()
    self.pcd.points = o3d.utility.Vector3dVector(self.current_points)
    self.pcd.paint_uniform_color([0.5, 0.5, 0.5])
    colors = np.array(self.pcd.colors)
    tp = (self.current_preds > 200) & (self.current_gt > 200)
    fp = (self.current_preds > 200) & (self.current_gt < 200)
    fn = (self.current_preds < 200) & (self.current_gt > 200)
  
    colors[tp] = [0, 1, 0]
    colors[fp] = [1, 0, 0]
    colors[fn] = [0, 0, 1]
  
    self.pcd.colors = o3d.utility.Vector3dVector(colors)
  
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-45, -45, -5),
                                               max_bound=(45, 45, 5))
    self.pcd = self.pcd.crop(bbox)  # set view area
    self.vis.add_geometry(self.pcd)
  
    # init keyboard controller
    key_listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
    key_listener.start()
    
    # init frame index
    self.frame_idx = 0
    self.num_frames = len(self.scan_files)

  def on_press(self, key):
    try:
      if key.char == 'q':
        try:
          sys.exit(0)
        except SystemExit:
          os._exit(0)
        
      if key.char == 'n':
        if self.frame_idx < self.num_frames - 1:
          self.frame_idx += 1
          self.current_points = load_vertex(self.scan_files[self.frame_idx])[:, :3]
          self.current_preds, _ = load_labels(self.predictions_files[self.frame_idx])
          self.current_gt, _ = load_labels(self.gt_paths[self.frame_idx])
          print("frame index:", self.frame_idx)
        else:
          print('Reach the end of this sequence!')
          
      if key.char == 'b':
        if self.frame_idx > 1:
          self.frame_idx -= 1
          self.current_points = load_vertex(self.scan_files[self.frame_idx])[:, :3]
          self.current_preds, _ = load_labels(self.predictions_files[self.frame_idx])
          self.current_gt, _ = load_labels(self.gt_paths[self.frame_idx])
          print("frame index:", self.frame_idx)
        else:
          print('At the start at this sequence!')
          
    except AttributeError:
      print('special key {0} pressed'.format(key))
      
  def on_release(self, key):
    try:
      if key.char == 'n':
        self.current_points = load_vertex(self.scan_files[self.frame_idx])[:, :3]
        self.current_preds, _ = load_labels(self.predictions_files[self.frame_idx])
        self.current_gt, _ = load_labels(self.gt_paths[self.frame_idx])
    
      if key.char == 'b':
        self.current_points = load_vertex(self.scan_files[self.frame_idx])[:, :3]
        self.current_preds, _ = load_labels(self.predictions_files[self.frame_idx])
        self.current_gt, _ = load_labels(self.gt_paths[self.frame_idx])
        
    except AttributeError:
      print('special key {0} pressed'.format(key))
  
  def run(self):
    current_points = self.current_points
    current_preds = self.current_preds
    current_gt = self.current_gt
    if (len(current_points) == len(current_preds)) \
        and (len(current_points) == len(current_gt)) \
        and (len(current_preds) == len(current_gt)):
      self.pcd.points = o3d.utility.Vector3dVector(current_points)
      self.pcd.paint_uniform_color([0.5, 0.5, 0.5])
      colors = np.array(self.pcd.colors)
    
      tp = (current_preds > 200) & (current_gt > 200)
      fp = (current_preds > 200) & (current_gt < 200)
      fn = (current_preds < 200) & (current_gt > 200)
    
      colors[tp] = [0, 1, 0]
      colors[fp] = [1, 0, 0]
      colors[fn] = [0, 0, 1]
    
      self.pcd.colors = o3d.utility.Vector3dVector(colors)
      
      self.vis.update_geometry(self.pcd)
      self.vis.poll_events()
      self.vis.update_renderer()
  
  
if __name__ == '__main__':
  # load config file
  config_filename = 'config/mos_tracking.yml'
  if len(sys.argv) > 1:
    config_filename = sys.argv[1]
  
  if yaml.__version__ >= '5.1':
    config = yaml.load(open(config_filename), Loader=yaml.FullLoader)
  else:
    config = yaml.load(open(config_filename))

  # init the mos visualizer
  visualizer = vis_mos_results(config)
  
  # run the visualizer
  while True:
    visualizer.run()