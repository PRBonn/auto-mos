#!/usr/bin/env python3
# Developed by Xieyuanli Chen
# Brief: functions for associating instances between two LiDAR scans
# This file is covered by the LICENSE file in the root of this project.

import numpy as np
from tracking_utils import tracking
from tracking_utils.kalman_filter import KalmanBoxTracker
from scipy.optimize import linear_sum_assignment as lsa


def associate_instances(tracked_instances, last_ins_id, ins_pred_raw,
                      pose, points, instances_voting_results, rearrange_associates,
                      frame_idx, moving_instances_bounding_boxes, min_num_points=5):
  """ Find instance associations between two LiDAR scans
  """
  ins_pred = np.copy(ins_pred_raw)
  ins_ids = np.unique(ins_pred)
  new_ids = np.array(list(rearrange_associates.values()))
  raw_ids = np.array(list(rearrange_associates.keys()))
  
  # shift points to global frame to associate with tracked_instances
  # Apply pose (without np.dot to avoid multi-threading)
  hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
  shifted_points = np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)
  
  new_instances = {}
  for _id in ins_ids:  # go over all instances in current scan
    if _id == 0:  # ignore if stuff
      continue
    ind = np.where(ins_pred == _id)
    
    # filter instances with few points
    if ind[0].shape[0] < min_num_points:
      ins_pred[ind] = 0
      continue
    
    # initialize object tracks in current frame
    bbox, kalman_bbox = tracking.get_bbox_from_points(shifted_points[ind])
    
    # init kalman bbox tracker
    tracker = KalmanBoxTracker(kalman_bbox, _id)
    center = tracking.get_median_center_from_points(shifted_points[ind])
    kalman_bboxes = []
    kalman_bboxes.append(kalman_bbox.flatten().tolist())
    
    # calculate the distance
    dist = np.linalg.norm(shifted_points[ind] - np.mean(shifted_points[ind], axis=0), axis=1)
    std = np.std(dist)
    
    # init new instance
    new_instances[_id] = {'life'         : 5,
                          'center'       : center,
                          'n_point'      : ind[0].shape[0],
                          'tracker'      : tracker,
                          'bbox'         : bbox,
                          'kalman_bbox'  : kalman_bbox,
                          'counter'      : 1,
                          'kalman_bboxes': kalman_bboxes,
                          'std'          : std,
                          }

  if len(tracked_instances) > 0:  # if tracking some instances, associate
    
    # predict previous instances new position
    for i in tracked_instances.keys():
      tracked_instances = tracked_instances.copy()
      tracked_instances[i]['kalman_bbox'] = (tracked_instances[i]['tracker'].predict()).flatten().tolist()
      tracked_instances[i]['bbox'] = tracking.kalman_box_to_eight_point(tracked_instances[i]['kalman_bbox'])
    
    # compute associations
    association_costs, associations = compute_associations(tracked_instances, new_instances)
    
    # perform associations
    for prev_id, new_id in associations:
      ins_ind = np.where((ins_pred == new_id))
      
      # assign consistent instance id to previous prediction
      ins_pred[ins_ind[0]] = prev_id
      
      # update the rearrange_associates
      rearrange_associates[int(raw_ids[new_ids == new_id])] = prev_id
      
      # update tracked_instances with the ones in current frame
      tracked_instances[prev_id]['life'] += 1
      tracked_instances[prev_id]['tracker'].update(new_instances[new_id]['kalman_bbox'], prev_id)
      tracked_instances[prev_id]['kalman_bbox'] = (tracked_instances[prev_id]['tracker'].get_state()).flatten().tolist()
      tracked_instances[prev_id]['bbox'] = tracking.kalman_box_to_eight_point(tracked_instances[prev_id]['kalman_bbox'])
      
      # new for auto label generation
      tracked_instances[prev_id]['counter'] += 1

      del new_instances[new_id]  # remove already assigned instances

    # record all kalman_bboxes history
    for i in tracked_instances.keys():
      tracked_instances[i]['kalman_bboxes'].append(tracked_instances[i]['kalman_bbox'][:7])
      
  # Manage new instances
  # add newly created instances to track
  for _id, instance in new_instances.items():
    idx = np.where(ins_pred == _id)
    if idx[0].shape[0] < min_num_points:
      continue
    if _id in tracked_instances.keys():
      continue
    tracked_instances[_id] = instance
  
  # kill instances which are not tracked for a while
  dont_track_ids = []
  for _id in tracked_instances.keys():
    if tracked_instances[_id]['life'] == 0:
      dont_track_ids.append(_id)
    else:
      tracked_instances[_id]['life'] -= 1
      
  for _id in dont_track_ids:
    # save the label for each instance
    label = 0
    if tracked_instances[_id]['counter'] > 5:
      kalman_bboxes = np.array(tracked_instances[_id]['kalman_bboxes'])
      # movement larger than the largest edge of the box
      if tracking.euclidean_dist(kalman_bboxes[-5], kalman_bboxes[0]) > np.max(kalman_bboxes[:, 4:7]):
        label = 1
        start_frame_idx = frame_idx - len(kalman_bboxes) + 1
        for idx in range(len(kalman_bboxes)):
          kalman_bbox = kalman_bboxes[idx]
          if start_frame_idx + idx in moving_instances_bounding_boxes.keys():
            moving_instances_bounding_boxes[start_frame_idx + idx].append(kalman_bbox)
          else:
            moving_instances_bounding_boxes[start_frame_idx + idx] = [kalman_bbox]
        
    instances_voting_results[_id] = label  # python will convert \n to os.linesep
    
    del tracked_instances[_id]
  
  # clean instances (in prediction) which have too few points
  for _id in np.unique(ins_pred):
    if _id == 0:
      continue
    valid_ind = np.argwhere(ins_pred == _id)[:, 0]
    
    if valid_ind.shape[0] < min_num_points:
      ins_pred[valid_ind] = 0
  
  # get last assigned instances id
  if len(tracked_instances) != 0:
    last_ins_id = max(tracked_instances)
  
  # new_ins_preds for each scan on the bacth:
  # consistent per-point instance id. id=0 for things
  # tracked_instances: dict with instances to track (and parameters)
  # last_ins_id: last id used on the tracked_instances
  return ins_pred, tracked_instances, last_ins_id, \
         instances_voting_results, rearrange_associates, moving_instances_bounding_boxes


def compute_associations(previous_instances, current_instances):
  """ compute the associations matrix and determine the instance associations
  """
  p_n = len(previous_instances.keys())
  c_n = len(current_instances.keys())
  association_costs = np.zeros((p_n, c_n))
  prev_ids = []
  current_ids = []
  
  for i, (id1, v1) in enumerate(previous_instances.items()):
    prev_ids.append(id1)
    for j, (id2, v2) in enumerate(current_instances.items()):
      if i == 0: current_ids.append(id2)
      
      cost_3d = 1 - tracking.IoU(v2['bbox'], v1['bbox'])
      if cost_3d > 0.95: cost_3d = 1e8
      
      cost_center = tracking.euclidean_dist(v2['kalman_bbox'], v1['kalman_bbox'])
      if cost_center > 2: cost_center = 1e8

      cost_volume = tracking.volume(v2['kalman_bbox'], v1['kalman_bbox'])
      if cost_volume > 0.7: cost_volume = 1e8
      
      # weight the costs
      association_costs[i, j] = cost_3d + cost_center + cost_volume
  
  idx1, idx2 = lsa(association_costs)
  associations = []
  for i1, i2 in zip(idx1, idx2):
    if association_costs[i1][i2] < 1e8:
      associations.append((prev_ids[i1], current_ids[i2]))
  
  return association_costs, associations


def kill_tracking(tracked_instances, instances_voting_results, moving_instances_bounding_boxes, frame_idx):
  """ kill the tracking and solve the border cases
  """
  for _id in tracked_instances.keys():
    label = 0
    kalman_bboxes = np.array(tracked_instances[_id]['kalman_bboxes'])
    
    if tracked_instances[_id]['counter'] > 5:
      if tracking.euclidean_dist(kalman_bboxes[-5], kalman_bboxes[0]) > np.max(kalman_bboxes[:, 4:7]):
        label = 1
        for idx, kalman_bbox in enumerate(kalman_bboxes):
          start_frame_idx = frame_idx - len(kalman_bboxes) + 1
          if start_frame_idx + idx in moving_instances_bounding_boxes.keys():
            moving_instances_bounding_boxes[start_frame_idx + idx].append(kalman_bbox)
          else:
            moving_instances_bounding_boxes[start_frame_idx + idx] = [kalman_bbox]
    
    else:
      if tracking.euclidean_dist(kalman_bboxes[-1], kalman_bboxes[0]) > np.max(kalman_bboxes[:, 4:7]):
        label = 1
        for idx, kalman_bbox in enumerate(kalman_bboxes):
          start_frame_idx = frame_idx - len(kalman_bboxes) + 1
          if start_frame_idx + idx in moving_instances_bounding_boxes.keys():
            moving_instances_bounding_boxes[start_frame_idx + idx].append(kalman_bbox)
          else:
            moving_instances_bounding_boxes[start_frame_idx + idx] = [kalman_bbox]
    
    instances_voting_results[_id] = label  # python will convert \n to os.linesep
    
  return instances_voting_results, moving_instances_bounding_boxes