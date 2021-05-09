import sys
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms

from pipeline.base.baseemb import BaseEmb


from pipeline.embedding.aicitynet import models
from pipeline.embedding.aicitynet.utils.torch_func import count_num_param,load_pretrained_weights


class AicityEmb(BaseEmb):
    def __init__(self,emb_model):

        print('Initializing model: {}'.format('resnet101'))
        self.extractor = models.init_model(name='resnet101', loss={'xent', 'htri'},use_gpu=True)
        print('Model size: {:.3f} M'.format(count_num_param(self.extractor)))
        #init extractor
        load_pretrained_weights(self.extractor, emb_model)
        self.extractor.eval()
        self.device = "cuda" if torch.cuda.is_available()  else "cpu"
        print("Loading weights from {}... Done!".format(emb_model))
        self.extractor.to(self.device)
        self.size = (256, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def _preprocess(self, croped_imgs):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            return (cv2.resize(im, size)).astype(np.float32)/255.
        
        im_batch = torch.cat([self.norm(_resize(im[:,:,(2,1,0)], self.size)).unsqueeze(0) for im in croped_imgs], dim=0).float()
        return im_batch

    def extract(self,croped_imgs):
        im_batch = self._preprocess(croped_imgs)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.extractor(im_batch)
        return features.cpu().numpy()
