import sys
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms

from pipeline.base.baseemb import BaseEmb

from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger
from fastreid.utils.file_io import PathManager

from pipeline.embedding.fastnet.predictor import FeatureExtractionDemo

class FastEmb(BaseEmb):
    def __init__(self,emb_model,conf):
        self.config_file = conf
        self.emb_model = emb_model
        cfg = self.setup_cfg()
        print(cfg)
        self.extractor = FeatureExtractionDemo(cfg)

    def setup_cfg(self):
        # load config from file and command-line arguments
        cfg = get_cfg()
        # add_partialreid_config(cfg)
        cfg.merge_from_file(self.config_file)
        cfg.MODEL.WEIGHTS = self.emb_model
        cfg.freeze()
        return cfg


    def extract(self,croped_imgs):
        cfeatures = []
        for croped_img in croped_imgs:
            feat = self.extractor.run_on_image(croped_img)
            feat = feat.numpy()
            cfeatures.append(feat)

        features = np.array(cfeatures).reshape(len(croped_imgs),-1)

        return features
