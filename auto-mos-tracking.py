#!/usr/bin/env python3
# Developed by Xieyuanli Chen
# Brief: using tracking results to separate moving and non-moving objects
# This file is covered by the LICENSE file in the root of this project.

import os
import sys
import yaml
import numpy as np
from tqdm import tqdm
from tracking_utils import tracking, utils
from tracking_utils.instance_association import associate_instances, kill_tracking


class mos_tracker:
  """ A 3D multi-object tracker for LiDAR-MOS automatic label generation
  """
  
  def __init__(self, config):
    # create prediction result folder
    prediction_folder = config['prediction_root']
    if not os.path.exists(os.path.join(prediction_folder, "sequences")):
      os.makedirs(os.path.join(prediction_folder, "sequences"))
    for seq_ in range(11):
      seq_ = '{0:02d}'.format(int(seq_))
      print("train", seq_)
      if not os.path.exists(os.path.join(prediction_folder, "sequences", seq_, "predictions")):
        os.makedirs(os.path.join(prediction_folder, "sequences", seq_))
        os.makedirs(os.path.join(prediction_folder, "sequences", seq_, "predictions"))
    
    # specify seq
    self.seq = str(config['seq']).zfill(2)
    
    # specify dataset paths
    dataset_root = config['dataset_root']
    scan_folder = os.path.join(dataset_root, 'sequences', self.seq, 'velodyne')
    pose_file = os.path.join(dataset_root, 'sequences', self.seq, 'poses.txt')
    calib_file = os.path.join(dataset_root, 'sequences', self.seq, 'calib.txt')
    
    # load LiDAR scans
    self.scan_paths = utils.load_files(scan_folder)
    
    # load poses
    self.poses = utils.load_poses_kitti(pose_file, calib_file)
    
    # load instances predictions
    instance_pred_root = config['instance_root']
    instance_pred_folder = os.path.join(instance_pred_root, 'sequences', self.seq, 'predictions')
    self.instance_pred_files = utils.load_files(instance_pred_folder)
    
    # specify the output mos predifion folder
    self.prediction_root = config['prediction_root']
  
  def run_tracking(self):
    """ Run the 3D multi-object tracking on all instances through the whole LiDAR sequence
    """
    print('Using tracking to distinguish moving and non-moving objects...')
    # init tracking parameters
    last_ins_id = 0
    tracked_instances = {}
    instance_associates_list = []
    instances_voting_results = {}
    moving_instances_bounding_boxes = {}
    seq_length = len(self.scan_paths)
    
    # generate tracking labels
    for frame_idx in tqdm(range(seq_length)):
      # load scan and corresponding instance labels
      current_scan = utils.load_vertex(self.scan_paths[frame_idx])
      current_inst_raw, _ = utils.load_labels(self.instance_pred_files[frame_idx])
      
      # make sure the incoming instance ids are different from the tracked ones
      current_inst_label, rearrange_associates = self.rearrange_new_instances_ids(current_inst_raw, last_ins_id)
      
      # associate the incoming instances to the tracked ones
      assoc_ins_pred, tracked_instances, last_ins_id, instances_voting_results, rearrange_associates, \
      moving_instances_bounding_boxes = associate_instances(
        tracked_instances, last_ins_id, current_inst_label, self.poses[frame_idx], current_scan[:, :3],
        instances_voting_results, rearrange_associates, frame_idx, moving_instances_bounding_boxes)
      instance_associates_list.append(rearrange_associates)
    
    # finish tracking
    instances_voting_results, moving_instances_bounding_boxes = kill_tracking(tracked_instances,
                                                                              instances_voting_results,
                                                                              moving_instances_bounding_boxes,
                                                                              seq_length)
    
    return instance_associates_list, moving_instances_bounding_boxes, instances_voting_results
  
  def gen_mos_labels(self, instance_associates_list, moving_instances_bounding_boxes, instances_voting_results):
    """ Generate LiDAR-MOS labels based on the tracking results
    """
    print('Now generating mos labels...')
    for frame_idx in tqdm(range(len(self.scan_paths))):
      # load scan and corresponding instance labels
      base_name = int(os.path.basename(self.scan_paths[frame_idx]).replace('.bin', ''))
      current_scan = utils.load_vertex(self.scan_paths[frame_idx])
      current_inst_label, _ = utils.load_labels(self.instance_pred_files[frame_idx])
      
      # init mos predictions
      instance_associates = instance_associates_list[frame_idx]
      labels_pred = np.ones(len(current_scan), dtype=np.uint32) * 9
      
      # for each current instance, determine whether it's moving or not
      unique_labels = np.unique(current_inst_label)
      original_indices = np.arange(len(current_inst_label))
      for unique_label in unique_labels:
        if unique_label == 0:
          continue
        inds = original_indices[np.flatnonzero(current_inst_label == unique_label)]
        assoc_ins = instance_associates[unique_label]
        if assoc_ins in instances_voting_results.keys():
          if instances_voting_results[assoc_ins] > 0:
            labels_pred[inds] = 251
        else:
          print('skipping', assoc_ins, 'with number of points:', len(inds))
      
      # in case the instance is not detected in the current frame but was tracked/predicted by the ekf
      # shift points to global frame to associate with tracked_instances
      # Apply pose (without np.dot to avoid multi-threading)
      hpoints = np.hstack((current_scan[:, :3], np.ones_like(current_scan[:, :1])))
      shifted_points = np.sum(np.expand_dims(hpoints, 2) * self.poses[frame_idx].T, axis=1)
      
      # refine the instances by using the tracked bboxes
      if frame_idx in moving_instances_bounding_boxes.keys():
        bboxes = moving_instances_bounding_boxes[frame_idx]
        if len(bboxes) > 0:
          for bbox in bboxes:
            moving_indexes = tracking.find_points_in_box(shifted_points, bbox)
            labels_pred[moving_indexes] = 251
      
      # save predictions
      file_name = os.path.join(self.prediction_root, "sequences", self.seq, "predictions",
                               str(base_name).zfill(6))
      labels_pred = labels_pred.reshape((-1)).astype(np.uint32)
      labels_pred.tofile(file_name + '.label')
  
  def rearrange_new_instances_ids(self, current_inst_raw, last_ins_id):
    """ rearrange current new instances ids based on the last maximum id
    """
    current_inst_label = np.copy(current_inst_raw)
    unique_labels = np.unique(current_inst_raw)
    original_indices = np.arange(len(current_inst_label))
    rearrange_associates = {}
    
    for idx, unique_label in enumerate(unique_labels):
      if unique_label == 0:
        continue
      inds = original_indices[current_inst_raw == unique_label]
      
      last_ins_id += 1
      current_inst_label[inds] = last_ins_id
      rearrange_associates[unique_label] = last_ins_id
    
    return current_inst_label, rearrange_associates


if __name__ == '__main__':
  # load config file
  config_filename = 'config/mos_tracking.yml'
  if len(sys.argv) > 1:
    config_filename = sys.argv[1]
  
  if yaml.__version__ >= '5.1':
    config = yaml.load(open(config_filename), Loader=yaml.FullLoader)
  else:
    config = yaml.load(open(config_filename))
  
  # init the mos tracker
  tracker = mos_tracker(config)
  
  # conduct the tracking
  instance_associates_list, moving_instances_bounding_boxes, instances_voting_results = tracker.run_tracking()
  
  # generate mos labels based on tracking results
  tracker.gen_mos_labels(instance_associates_list, moving_instances_bounding_boxes, instances_voting_results)
