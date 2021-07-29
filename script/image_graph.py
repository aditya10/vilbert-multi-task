from typing import List
import csv
import h5py
import numpy as np
import copy
import pickle
import lmdb  # install lmdb by "pip install lmdb"
import base64
import pdb
import torch

datapath = "../datasets/refcoco/refcoco+_unc/refcoco+_resnext152_faster_rcnn_genome.lmdb"

env = lmdb.open(datapath, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)

def iou(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.size(0)
    K = gt_boxes.size(0)

    gt_boxes_area = (
        (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)
    ).view(1, K)

    anchors_area = (
        (anchors[:, 2] - anchors[:, 0] + 1) * (anchors[:, 3] - anchors[:, 1] + 1)
    ).view(N, 1)

    boxes = anchors.view(N, 1, 4).expand(N, K, 4)
    query_boxes = gt_boxes.view(1, K, 4).expand(N, K, 4)

    iw = (
        torch.min(boxes[:, :, 2], query_boxes[:, :, 2])
        - torch.max(boxes[:, :, 0], query_boxes[:, :, 0])
        + 1
    )
    iw[iw < 0] = 0

    ih = (
        torch.min(boxes[:, :, 3], query_boxes[:, :, 3])
        - torch.max(boxes[:, :, 1], query_boxes[:, :, 1])
        + 1
    )
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps

with env.begin(write=False) as txn:
    image_ids = pickle.loads(txn.get("keys".encode()))
    image_id = image_ids[3]
    item = pickle.loads(txn.get(image_id))
    image_id = item["image_id"]
    image_h = int(item["image_h"])
    image_w = int(item["image_w"])
    # num_boxes = int(item['num_boxes'])

    # features = np.frombuffer(base64.b64decode(item["features"]), dtype=np.float32).reshape(num_boxes, 2048)
    # boxes = np.frombuffer(base64.b64decode(item['boxes']), dtype=np.float32).reshape(num_boxes, 4)
    features = item["features"]
    print(features.shape)
    features = item["features"].reshape(-1, 2048)
    print(features.shape)
    boxes = item["boxes"].reshape(-1, 4)
    print(boxes.shape)
    bs = torch.tensor(boxes).float()
    print(bs)
    iou_mask = iou(bs, bs)
    print(iou_mask)

    num_boxes = features.shape[0]
    g_feat = np.sum(features, axis=0) / num_boxes
    num_boxes = num_boxes + 1
    features = np.concatenate(
        [np.expand_dims(g_feat, axis=0), features], axis=0
    )

    image_location = np.zeros((boxes.shape[0], 5), dtype=np.float32)
    image_location[:, :4] = boxes
    image_location[:, 4] = (
        (image_location[:, 3] - image_location[:, 1])
        * (image_location[:, 2] - image_location[:, 0])
        / (float(image_w) * float(image_h))
    )

    image_location_ori = copy.deepcopy(image_location)
    image_location[:, 0] = image_location[:, 0] / float(image_w)
    image_location[:, 1] = image_location[:, 1] / float(image_h)
    image_location[:, 2] = image_location[:, 2] / float(image_w)
    image_location[:, 3] = image_location[:, 3] / float(image_h)

    g_location = np.array([0, 0, 1, 1, 1])
    image_location = np.concatenate(
        [np.expand_dims(g_location, axis=0), image_location], axis=0
    )

    g_location_ori = np.array([0, 0, image_w, image_h, image_w * image_h])
    image_location_ori = np.concatenate(
        [np.expand_dims(g_location_ori, axis=0), image_location_ori], axis=0
    )
    print(features.shape)
    print(num_boxes)
    print(image_location.shape)
    print(image_location_ori.shape)


