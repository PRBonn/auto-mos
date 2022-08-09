#!/usr/bin/env python3
# Developed by Xieyuanli Chen
# Brief: this script provides an example of visualizing the instances in point clouds.
# This file is covered by the LICENSE file in the root of this project.

import os
import sys
import time
import yaml
import numpy as np
import open3d as o3d
from tqdm import tqdm
from colorsys import hls_to_rgb
from tracking_utils import utils
import pynput.keyboard as keyboard


def gen_color_map(n):
  """ generate color map given number of instances
  """
  colors = []
  for i in np.arange(0., 360., 360. / n):
    h = i / 360.
    l = (50 + np.random.rand() * 10) / 100.
    s = (90 + np.random.rand() * 10) / 100.
    colors.append(hls_to_rgb(h, l, s))

  return np.array(colors)


def vis_instance_online(scan_files, instance_files):
  """ visualize the instances online
  """
  vis = o3d.visualization.Visualizer()
  vis.create_window()
  
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(utils.load_vertex(scan_files[0])[:, :3])

  # instances = np.load(instance_files[0])['arr_0']
  instances, _ = utils.load_labels(instance_files[0])
  unique_instances = np.unique(instances)

  max_label = max(unique_instances)
  print(f"point cloud has {max_label + 1} clusters")
  color_map = gen_color_map(max_label + 1)
  colors = color_map[instances.astype(int)]
  colors[instances < 1] = [0.5, 0.5, 0.5]

  bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-50, -50, -5),
                                             max_bound=(50, 50, 5))
  pcd = pcd.crop(bbox)  # set view area
  
  vis.add_geometry(pcd)

  # init keyboard controller
  def on_press(key):
    try:
      if key.char == 'q':
        try:
          sys.exit(0)
        except SystemExit:
          os._exit(0)
    except AttributeError:
      pass
  key_listener = keyboard.Listener(on_press=on_press)
  key_listener.start()
  
  for frame_idx in tqdm(range(len(scan_files))):
    instances, _ = utils.load_labels(instance_files[frame_idx])
    unique_instances = np.unique(instances)
    color_map = gen_color_map(max(unique_instances) + 1)
    colors = color_map[instances.astype(int)]
    colors[instances < 1] = [0.5, 0.5, 0.5]

    pcd.points = o3d.utility.Vector3dVector(utils.load_vertex(scan_files[frame_idx])[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # pcd.crop(bbox)
    
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.05)


if __name__ == '__main__':
  # load config file
  config_filename = 'config/mos_tracking.yml'
  if len(sys.argv) > 1:
    config_filename = sys.argv[1]
  
  if yaml.__version__ >= '5.1':
    config = yaml.load(open(config_filename), Loader=yaml.FullLoader)
  else:
    config = yaml.load(open(config_filename))

  seq = str(config['seq']).zfill(2)
  dataset_root = config['dataset_root']
  instance_pred_root = config['instance_root']

  # specify folders
  scan_folder = os.path.join(dataset_root, 'sequences', seq, 'velodyne')
  prediction_folder = os.path.join(instance_pred_root, 'sequences', seq, 'predictions')

  # load files
  scan_paths = utils.load_files(scan_folder)
  instance_files = utils.load_files(prediction_folder)
  
  vis_instance_online(scan_paths, instance_files)
