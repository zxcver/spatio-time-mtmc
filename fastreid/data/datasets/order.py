# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import os
import os.path as osp
import re
import warnings

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class Order(ImageDataset):
    """Order

    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    dataset_dir = ''
    dataset_name = "order"
    def __init__(self, root='datasets', **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'order')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "bounding_box_train" under '
                          '"Market-1501-v15.09.15".')

        self.train_dir = osp.join(self.data_dir, 'train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'gallery')

        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]

        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir)
        query = self.process_dir(self.query_dir, is_train=False)
        gallery = self.process_dir(self.gallery_dir, is_train=False)


        super(Order, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        data = []
        dict_camid = {} 
        camid_index = 0
        pids = os.listdir(dir_path)
        for pid in pids:
            pid_path = osp.join(dir_path,pid)
            if not osp.isdir(pid_path):
                continue
            if pid == 'a00':
                int_pid = 0
            else:
                int_pid = int(pid)+1
            communitys = os.listdir(pid_path)
            for community in communitys:
                community_path = osp.join(pid_path,community)
                if not osp.isdir(community_path):
                    continue
                camids = os.listdir(community_path)
                for camid in camids:
                    camid_path = osp.join(community_path,camid)
                    if not osp.isdir(camid_path):
                        continue
                    if not camid in dict_camid.keys():
                        dict_camid[camid] = camid_index
                        camid_index+=1
                    imgs = os.listdir(camid_path)
                    for img in imgs:   
                        img_path = osp.join(camid_path,img)
                        if img =='.DS_Store':
                            continue
                        if is_train:
                            fpid = self.dataset_name + "_" + str(int_pid)
                            fcamid = self.dataset_name + "_" + str(dict_camid[camid])
                        else:
                            fpid = int_pid
                            fcamid = dict_camid[camid]
                        data.append((img_path, fpid, fcamid))
        return data
