import sys
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms

from pipeline.base.baseemb import BaseEmb

from pipeline.embedding.normalnet.normal_model import Net


class NormalnetEmb(BaseEmb):
    def __init__(self,emb_model):
        #init extractor
        self.extractor = Net(reid=True)
        self.extractor.eval()
        net_dict = self.extractor.state_dict()
        saved_state_dict = torch.load(emb_model, map_location=lambda storage, loc: storage)['net_dict']
        need_dict = {k: v for k, v in saved_state_dict.items() if k in net_dict}
        net_dict.update(need_dict)
        self.extractor.load_state_dict(net_dict)
        self.device = "cuda" if torch.cuda.is_available()  else "cpu"
        print("Loading weights from {}... Done!".format(emb_model))
        self.extractor.to(self.device)
        self.size = (64, 128)
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
            cfeatures = features.cpu().numpy()
        return cfeatures
